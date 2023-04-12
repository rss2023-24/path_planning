#!/usr/bin/env python

import rospy
import tf
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray, PoseWithCovarianceStamped
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

        self.map_grid = None
        self.map_info = None
        self.start_loc = None
        self.goal_loc = None
        # self.start_loc = (0, 0)
        # self.goal_loc = (0, 0)



    def map_cb(self, msg):
        msg_map = msg.data
        grid_dimensions = (msg.info.height, msg.info.width)
        grid = np.reshape(msg_map, grid_dimensions) == 0

        self.map_grid = grid
        self.map_info = msg.info

        print(grid)

    def start_cb(self, msg):
        self.start_pos = (msg.pose.pose.position.y, msg.pose.pose.position.x)
        # print(self.start_pos)

        # self.compute_real_coordinates((0.0, 0.0))

    def odom_cb(self, msg):
        pass  ## REMOVE AND FILL IN ##

    def goal_cb(self, msg):
        self.goal_loc = (msg.pose.position.y, msg.pose.position.x)

        if self.start_loc == None:
            # self.plan_path(self.map_grid, self.start_loc, self.goal_loc)
            pass
        print(self.goal_loc)


    def compute_real_coordinates(self, pixel_point):
        py, px = pixel_point
        spy, spx = py * self.map_info.resolution, px * self.map_info.resolution

        map_orientation = self.map_info.origin.orientation
        rot_matrix = tf.transformations.quaternion_matrix([map_orientation.x, map_orientation.y, map_orientation.z, map_orientation.w])
        rot_matrix[0:3, 3] = np.array( [self.map_info.origin.position.x, self.map_info.origin.position.y, self.map_info.origin.position.z] ) # Include translation

        point_vec =  np.array( [spx, spy, 0.0, 1.0] )
        coord_x, coord_y, coord_z = point_vec / point_vec[2]

        return (coord_y, coord_x)

    def compute_pixel_point(self, real_coordinates):
        pass




    def compute_distance(self, p1, p2):
        y1, x1 = p1
        y2, x2 = p2
        return ( (y2 - y1) ** 2.0 + (x2 - x1) ** 2.0  ) ** (1.0/2.0)
    
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
                if valid_coord and grid[new_y, new_x]:
                    neighbors.add( (new_x, new_y) )
        
        return neighbors

    def plan_path(self, start_point, goal_point):
        ## Uses A* ##

        queue = [ (self.estimate_heuristic(start_point, goal_point), 0.0, start_point) ]  # Use as a heap (estimated_cost, current_weight, point)
        parents = {}  # Also use for identifying visited cells

        # cost_map = { start_point: 0 }
        # cost_map_with_heuristic = { start_point: self.estimate_heuristic(start_point, goal_point) }

        # weight_map = { start_point: 0.0 }

        while queue:
            estimated_cost, current_weight, current_point = heapq.heappop(queue)
            # current_weight = weight_map[current_point]

            if current_point == goal_point:
                print("Yeet")
                return
            
            neighbors = self.get_valid_neighbors(current_point)
            for neighbor_point in neighbors:
                if neighbor_point not in parents:
                    edge_weight = self.compute_distance(current_point, neighbor_point)
                    new_weight = edge_weight + current_weight
                    new_estimate = new_weight + self.estimate_heuristic(neighbor_point, goal_point)

                    # prev_weight = weight_map[neighbor_point]

                    parents[neighbor_point] = current_point
                    queue_item = (new_estimate, new_weight, neighbor_point)
                    heapq.heappush(queue, queue_item)

        # publish trajectory
        self.traj_pub.publish(self.trajectory.toPoseArray())

        # visualize trajectory Markers
        self.trajectory.publish_viz()


if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
