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
kuka = arena.robot
plane = arena.plane

# Populate boxes
arena.populate()

# Initialise RRT
rrt = JointSpaceRRT(arena, max_iterations, goal_sample_probability,
                    number_of_joint_space_collision_nodes, use_angular_difference=use_angular_difference)

rrt.run()

# Pause for a while so you can observe result
time.sleep(5)
pybullet.disconnect()

# TODO: project questions
#   * how to do collision checking in a joint space RRT? Impossible directly -- will need to do forward kinematics each time and do a collision check, interesting to check how this goes vs. the inverse kinematics required for task space rrt
