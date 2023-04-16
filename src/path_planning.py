#!/usr/bin/env python

import rospy
import tf
import numpy as np
from scipy import ndimage
from geometry_msgs.msg import PoseStamped, PoseArray, PoseWithCovarianceStamped, Point
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory
import heapq

np.set_printoptions(suppress=True)

class PathPlan(object):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic")
        self.start_topic = '/initialpose'

        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.start_sub = rospy.Subscriber(self.start_topic, PoseWithCovarianceStamped, self.start_cb, queue_size=10)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)

        self.trajectory = LineTrajectory("/planned_trajectory")
        self.traj_pub = rospy.Publisher("/trajectory/current", PoseArray, queue_size=10)
        # self.path_pub = rospy.Publisher("/path/points", PointArray, queue_size=10)

        self.map_grid = None
        self.map_info = None
        self.rot_matrix = None
        self.start_loc = None
        self.goal_loc = None
        # self.start_loc = (0, 0)
        # self.goal_loc = (0, 0)



    def map_cb(self, msg):
        msg_map = msg.data
        grid_dimensions = (msg.info.height, msg.info.width)
        grid = np.reshape(msg_map, grid_dimensions)
        grid = ndimage.binary_dilation(grid, iterations=14)


        self.map_grid = grid
        self.map_info = msg.info

        map_orientation = self.map_info.origin.orientation
        rot_matrix = tf.transformations.quaternion_matrix([map_orientation.x, map_orientation.y, map_orientation.z, map_orientation.w])
        rot_matrix[0:3, 3] = np.array( [self.map_info.origin.position.x, self.map_info.origin.position.y, self.map_info.origin.position.z] ) # Include translation
        self.rot_matrix = rot_matrix
        self.inverse_rot_matrix = np.linalg.inv(self.rot_matrix)

        print("Map obtained")

    def start_cb(self, msg):
        self.start_loc = (msg.pose.pose.position.y, msg.pose.pose.position.x)

    def goal_cb(self, msg):
        self.goal_loc = (msg.pose.position.y, msg.pose.position.x)

        if self.start_loc != None:
            self.plan_path()

    def odom_cb(self, msg):
        self.start_loc = (msg.pose.pose.position.y, msg.pose.pose.position.x)

    def compute_real_coordinates(self, pixel_point):
        py, px = pixel_point
        spy, spx = py * self.map_info.resolution, px * self.map_info.resolution

        point_vec =  np.array( [spx, spy, 0.0, 1.0] ).T  # x, y, z, w
        prenorm_vec = np.matmul(self.rot_matrix, point_vec)

        coord_x, coord_y, coord_z, _ = prenorm_vec / prenorm_vec[3]

        return (coord_y, coord_x)

    def compute_pixel_point(self, real_coordinates):
        ry, rx = real_coordinates
        point_vec = np.array( [rx, ry, 0.0, 1.0] )

        pre_resolution_vec =  np.matmul(self.inverse_rot_matrix, point_vec)
        coord_x, coord_y, coord_z, _ = np.rint(pre_resolution_vec / self.map_info.resolution).astype(int)

        return (coord_y, coord_x)
    

    def compute_distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def estimate_heuristic(self, p1, goal):
        return self.compute_distance(p1, goal)

    def get_valid_neighbors(self, point):
        grid = self.map_grid
        height, width = grid.shape
        py, px = point

        neighbors = set()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                new_y, new_x = py + dy, px + dx
                valid_coord = 0 <= new_y < height and 0 <= new_x < width and (dy, dx) != (0, 0)
                if valid_coord and grid[new_y, new_x] == 0:
                    neighbors.add( (new_y, new_x) )
        
        return neighbors

    def make_trajectory(self, path):
        traj = LineTrajectory()
        for py, px in path:
            ry, rx = self.compute_real_coordinates( (py, px) )

            traj_point = Point()
            traj_point.x = rx
            traj_point.y = ry
            traj_point.z = 0

            traj.addPoint(traj_point)
        
        return traj

    def plan_path(self):
        ## Uses A* ##

        start_point = self.compute_pixel_point(self.start_loc)
        goal_point = self.compute_pixel_point(self.goal_loc)

        queue = [ (self.estimate_heuristic(start_point, goal_point), 0.0, start_point) ]  # Use as a heap (estimated_cost, current_weight, point)
        parents = { start_point: None }  # Also use for identifying visited cells

        while queue:
            estimated_cost, current_weight, current_point = heapq.heappop(queue)

            if current_point == goal_point:
                
                path = [goal_point]
                parent = goal_point
                while parents[parent] != None:
                    parent = parents[parent]
                    path.append(parent)
                path = path[::-1]

                self.trajectory = self.make_trajectory(path)

                # publish trajectory
                self.traj_pub.publish(self.trajectory.toPoseArray())

                # visualize trajectory Markers
                self.trajectory.publish_viz()

                return
            
            neighbors = self.get_valid_neighbors(current_point)
            for neighbor_point in neighbors:
                if neighbor_point not in parents:
                    edge_weight = self.compute_distance(current_point, neighbor_point)
                    new_weight = edge_weight + current_weight
                    new_estimate = new_weight + self.estimate_heuristic(neighbor_point, goal_point)

                    parents[neighbor_point] = current_point
                    queue_item = (new_estimate, new_weight, neighbor_point)
                    heapq.heappush(queue, queue_item)


if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
