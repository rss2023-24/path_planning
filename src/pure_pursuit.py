#!/usr/bin/env python
from __future__ import division

import rospy
import numpy as np
import time
import utils
import tf, tf2_ros

from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from std_msgs.msg import Float32

class PurePursuit(object):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
    def __init__(self):
        self.odom_topic       = rospy.get_param("~odom_topic")
        self.lookahead        = 1.6
        self.speed            = 1.5
        self.wheelbase_length = 0.325
        self.trajectory  = utils.LineTrajectory("/followed_trajectory")
        self.trajectory_np = None
        self.position = None
        self.orientation = None
        self.points_np = []
        self.min_distances = [0,]
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)
        self.start_sub = rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.start_cb, queue_size=10)
        self.tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(self.tfBuffer)
        self.traj_sub = rospy.Subscriber("/trajectory/current", PoseArray, self.trajectory_callback, queue_size=1)
        #self.load_traj_sub = rospy.Subscriber("/loaded_trajectory/path", PoseArray, self.trajectory_callback, queue_size=1)
        self.drive_pub = rospy.Publisher("/drive", AckermannDriveStamped, queue_size=1)
        self.pose_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        self.init_pose_sub = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.pose_callback, queue_size=1)
        self.dist_pub = rospy.Publisher("/dist", Float32, queue_size=1)
    
    # def conv_load_traj_callback(self, msg):
    #     print("loading trajectory")
    #     new_pose = msg.toPoseArray()
    #     self.trajectory_callback(new_pose)


    def trajectory_callback(self, msg):
        ''' Clears the currently followed trajectory, and loads the new one from the message
        '''
        print("Receiving new trajectory:", len(msg.poses), "points")
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
        self.points_np = np.array(self.trajectory.points)
        self.last = False # on last segment indicator
        self.min_distances = [0,]

    def goal_cb(self, msg):
        self.min_distances = [0,]

    def start_cb(self, msg):
        self.min_distances = [0,]

    def odom_callback(self, msg):
        self.position = msg.pose.pose.position
        self.orientation = msg.pose.pose.orientation
        self.drive()

    def pose_callback(self, msg):
        self.last = False # setting new pose allows new goals in drive()

    def drive(self):
        if len(self.trajectory.points) == 0:
            return
        drive_cmd = AckermannDriveStamped()
        drive_header = Header()
        drive_header.stamp = rospy.Time.now() 
        drive_header.frame_id = "base_link"
        drive_cmd.header = drive_header
        
        distances = []
        pos = np.array([self.position.x, self.position.y])
        # TODO change for loop to numpy matrix ops
        for i in range(1, len(self.points_np)):
            # start and end points of a segment
            u = self.points_np[i-1]
            v = self.points_np[i]
            l2 = np.sum(np.square(v - u))
            if l2 == 0:
                dist = np.linalg.norm(pos - u)
            else:
                t = np.sum((pos - u) * (v - u)) / l2
                t = max(0, min(1, t))
                dist = np.linalg.norm(pos - (u + t * (v - u)))
            distances.append(dist)

        min_dist_ix, min_dist = min(enumerate(distances), key=lambda d:d[1])
        self.dist_pub.publish(min_dist)
        self.min_distances.append(min_dist)
        rospy.logerr("Average Error: " + str(np.average(self.min_distances)))
        if min_dist > self.lookahead:
            # too far from path, stop driving
            drive_cmd.drive.speed = 0
            self.drive_pub.publish(drive_cmd)
            return
        if min_dist_ix == len(distances) - 1:
            self.last = True
            
        goal = None
        # look at segments in forward trajectory until intersection with circle is found
        for i in range(min_dist_ix, len(self.points_np)-1):
            start = self.points_np[i]
            end = self.points_np[i + 1]
            v = end - start
            a = np.dot(v, v)
            b = 2 * np.dot(v, start - pos)
            c = np.dot(start, start) + np.dot(pos, pos) - 2*np.dot(start, pos) - self.lookahead**2
            disc =  b**2 - 4 * a * c
            if disc < 0:
                continue # line misses the circle
            sqrt_disc = disc ** (0.5)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)
            if 0 <= t2 <= 1:
                if 0 <= t1 <= 1:
                    # handles edge case with 2 intersections by using the one
                    # closer to the end of the line segment to continue forward on the path
                    intersection1 = start + t1 * v
                    intersection2 = start + t2 * v
                    if np.linalg.norm(end - intersection1) <= np.linalg.norm(end - intersection1):
                        goal = intersection1
                    else:
                        goal = intersection2
                else:
                    goal = start + t2 * v
                break 
            elif 0 <= t1 <= 1:
                goal = start + t1 * v
                break 

        if (goal is None or 
           (self.last and np.linalg.norm(self.points_np[-1] - pos) < self.lookahead)): 
            # set to the last point in the trajectory
            goal = self.points_np[-1]
        
        ## pure pursuit to goal ##

        #rospy.logerr(pos)
        #rospy.logerr(goal)
        rotation = [self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w]
        translation = [self.position.x, self.position.y, self.position.z]
        euler = tf.transformations.euler_from_quaternion(rotation)
        rot_angle = euler[2]
        baselink_wrt_map = np.array(
            [[np.cos(rot_angle), -np.sin(rot_angle), translation[0]],
             [np.sin(rot_angle),  np.cos(rot_angle), translation[1]],
             [0, 0, 1]])
        map_wrt_baselink = np.linalg.inv(baselink_wrt_map)
        relative_pos = np.matmul(map_wrt_baselink, np.append(goal, 1))
        #rospy.logerr(relative_pos)
        relative_pos = relative_pos[:2]
        if np.linalg.norm(relative_pos) < 0.8:
            # use stopping distance 0.8 to stop at goal
            drive_cmd.drive.speed = 0
        else:
            theta = np.arctan2(relative_pos[1], relative_pos[0])
            L_1 = np.linalg.norm(relative_pos)
            L = self.wheelbase_length
            R = L_1 / (2 * np.sin(theta))
            turn_angle = np.arctan(L / R)
            #rospy.logerr(turn_angle)
            drive_cmd.drive.speed = self.speed 
            drive_cmd.drive.steering_angle = turn_angle

        self.drive_pub.publish(drive_cmd)


if __name__=="__main__":
    rospy.init_node("pure_pursuit")
    pf = PurePursuit()
    rospy.spin()
