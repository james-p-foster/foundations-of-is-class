import numpy as np
import matplotlib.pyplot as plt

from arena import Arena2D
from rrt import RRTPlanner

initial_state = np.array([0.1, 0.1])
goal_state = np.array([0.9, 0.9])

arena = Arena2D()
rrt = RRTPlanner(initial_state, goal_state, arena)
rrt.run()
fig, ax = plt.subplots()
fig, ax = rrt.plot(fig, ax)
plt.show()
