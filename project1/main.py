import numpy as np

from rrt import RRTPlanner

initial_state = np.array([0.1, 0.1])
goal_state = np.array([0.9, 0.9])
num_obstacles = 20
max_obstacle_radius = 0.1

rrt = RRTPlanner(initial_state, goal_state, num_obstacles, max_obstacle_radius)
rrt.run()
