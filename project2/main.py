import pybullet
import time
import numpy as np

from arena import Arena
from rrt import JointSpaceRRT


x_limits = (-0.8, 0.8)
y_limits = (-0.8, 0.8)
z_limits = (0, 0.8)  # don't want the box underground
number_of_boxes = 10
start_joint_configuration = np.zeros(7)
number_of_joint_space_collision_nodes = 100

# TODO: VARY THESE IN YOUR EXPERIMENTS!
max_iterations = 1000
goal_sample_probability = 0.02
use_angular_difference = True
norm_type = 2
distance_threshold = 4

arena = Arena(number_of_boxes, x_limits, y_limits, z_limits,
              use_angular_difference=use_angular_difference)

# Populate boxes
arena.populate()

# Initialise RRT
rrt = JointSpaceRRT(arena, max_iterations, goal_sample_probability,
                    number_of_joint_space_collision_nodes, use_angular_difference=use_angular_difference, playback_results=False)

# Run RRT
vertices_to_goal, total_rrt_time, total_find_valid_joint_configuration_time = rrt.run()
proportion_of_time_spent_finding_valid_joint_configuration = total_find_valid_joint_configuration_time / total_rrt_time
print(f"Vertices to goal:\n{vertices_to_goal}")
print(f"Total RRT time: {total_rrt_time}")
print(f"Total time spent finding valid joint configurations: {total_find_valid_joint_configuration_time}")
print(f"Proportion of time spent finding valid joint configurations: {proportion_of_time_spent_finding_valid_joint_configuration}")
