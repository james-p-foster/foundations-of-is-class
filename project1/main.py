import numpy as np
import matplotlib.pyplot as plt

from arena import Arena2D
from rrt import RRTPlanner

initial_state = np.array([0.1, 0.1])
goal_state = np.array([0.9, 0.9])

arena = Arena2D()
rrt = RRTPlanner(initial_state, goal_state, arena)
rrt.run()
vertices = rrt.vertices
plt.scatter([vertex[0] for vertex in vertices], [vertex[1] for vertex in vertices], c='r', s=5)
edges = rrt.edges
for edge in edges:
    plt.plot([edge[0], edge[2]], [edge[1], edge[3]], c='b')
plt.scatter(goal_state[0], goal_state[1], c='g', s=25)
plt.show()
