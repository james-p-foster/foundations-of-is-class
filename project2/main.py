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
max_rrt_iterations = 20  # TODO: for now, to be upped when doing stat studies
max_sample_iterations = 100
goal_sample_probability = 0.02
goal_eps = 1e-1
number_of_joint_space_collision_nodes = 100
# TODO: play around with a lot of different norm types and thresholds, it's a default argument atm

arena = Arena(number_of_boxes, x_limits, y_limits, z_limits)
kuka = arena.robot
plane = arena.plane

# Populate boxes
arena.populate()

# Initialise RRT
rrt = JointSpaceRRT(arena, max_rrt_iterations, max_sample_iterations,
                    goal_sample_probability, goal_eps,
                    number_of_joint_space_collision_nodes)

rrt.run()

# TODO: project questions
#   * when calculating the nearest vertex in RRT to find what vertex to link to, what is a good distance metric in joint space? 2 norm, 1 norm, inf norm?
#   * will probably need some angle wrapping capability, e.g. -pi/2 and +pi/2 are actually the same angle
#   * two seperate sims? One for visualising the result and the other for collision checking? Check GUI and DIRECT server options
#   * how to do collision checking in a joint space RRT? Impossible directly -- will need to do forward kinematics each time and do a collision check, interesting to check how this goes vs. the inverse kinematics required for task space rrt

# End
# Pause for a while so you can observe result
time.sleep(5)
pybullet.disconnect()

