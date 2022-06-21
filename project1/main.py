import matplotlib.pyplot as plt
import numpy as np

from rrt import RRTPlanner


def run_multiple_unconstrained_rrt(initial_state, goal_state, number_of_runs,
                                   maximum_distance_between_vertices, goal_sample_probability):
    success_array = np.zeros(number_of_runs, dtype=bool)
    iterations_array = np.zeros(number_of_runs)
    for i in range(number_of_runs):
        rrt = RRTPlanner(initial_state, goal_state, 0, 0,
                         maximum_distance_between_vertices=maximum_distance_between_vertices,
                         goal_sample_probability=goal_sample_probability)
        success, iterations = rrt.run()
        success_array[i] = success
        iterations_array[i] = iterations
    return success_array, iterations_array


def run_multiple_constrained_rrt(initial_state, goal_state, number_of_runs, number_of_obstacles, max_obstacle_radius):
    success_array = np.zeros(number_of_runs, dtype=bool)
    iterations_array = np.zeros(number_of_runs)
    for i in range(number_of_runs):
        rrt = RRTPlanner(initial_state, goal_state, number_of_obstacles, max_obstacle_radius)
        success, iterations = rrt.run()
        success_array[i] = success
        iterations_array[i] = iterations
    return success_array, iterations_array


# First set of experiments: unconstrained RRT.
# Vary RRT parameters to see their effect:
#   * maximum distance between vertices (0.1, 0.2, 0.3, 0.4, 0.5)
#   * goal sample probability (0.01, 0.02, ...,  0.1)
# Run 50 RRTs for each configuration.
# Fixed initial and goal states.
# No obstacles.
# Collision checking stays at same resolution of 1e-3.
# Goal detection eps stays fixed at 1e-2.
initial_state = np.array([0.1, 0.1])
goal_state = np.array([0.9, 0.9])
number_of_runs = 20

max_distance_values = [0.1, 0.2, 0.3, 0.4, 0.5]
goal_sample_probability_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
# Represent results as a matrix, with max_distance_values as rows, and goal_sample_probability_values as columns
number_of_successes_array = np.zeros((len(max_distance_values), len(goal_sample_probability_values)))
average_iterations_array = np.zeros((len(max_distance_values), len(goal_sample_probability_values)))
dataset_number = 0
for (i, _) in enumerate(max_distance_values):
    for (j, _) in enumerate(goal_sample_probability_values):
        print(f"Dataset {dataset_number}")
        success_array, iterations_array = run_multiple_unconstrained_rrt(initial_state, goal_state, number_of_runs,
                                                                         maximum_distance_between_vertices=
                                                                         max_distance_values[i],
                                                                         goal_sample_probability=
                                                                         goal_sample_probability_values[j])
        print(success_array)
        print(iterations_array)
        # To get a scalar related to successes, we simply count the number of successes
        number_of_successes = np.count_nonzero(success_array)
        number_of_successes_array[i, j] = number_of_successes
        # To get a scalar related to required iterations, we take the average number of iterations
        average_iterations = np.mean(iterations_array)
        average_iterations_array[i, j] = average_iterations
        dataset_number += 1
print("DONE!")
print(number_of_successes_array)
print(average_iterations_array)
fig, ax = plt.subplots()
ax.matshow(number_of_successes_array)
for (i, _) in enumerate(goal_sample_probability_values):
    for (j, _) in enumerate(max_distance_values):
        score = number_of_successes_array[j, i]
        ax.text(i, j, int(score), va='center', ha='center', c='w')
ax.set_title("Number of successes")
ax.set_xlabel("goal sample probability")
ax.set_xticks(np.arange(len(goal_sample_probability_values)))
ax.set_xticklabels([str(value) for value in goal_sample_probability_values])
ax.set_ylabel("maximum distance between vertices")
ax.set_yticks(np.arange(len(max_distance_values)))
ax.set_yticklabels([str(value) for value in max_distance_values])
plt.show()
fig, ax = plt.subplots()
ax.matshow(average_iterations_array)
for (i, _) in enumerate(goal_sample_probability_values):
    for (j, _) in enumerate(max_distance_values):
        score = average_iterations_array[j, i]
        ax.text(i, j, f"{score:.2f}", va='center', ha='center', c='w')
ax.set_title("Average iterations")
ax.set_xlabel("goal sample probability")
ax.set_xticks(np.arange(len(goal_sample_probability_values)))
ax.set_xticklabels([str(value) for value in goal_sample_probability_values])
ax.set_ylabel("maximum distance between vertices")
ax.set_yticks(np.arange(len(max_distance_values)))
ax.set_yticklabels([str(value) for value in max_distance_values])
plt.show()

# Second set of experiments: RRT with obstacles.
# Vary number of obstacles and their maximum radius to see the effect:
#   * number of obstacles (5, 10, 15, 20, 25)
#   * max obstacle radius (0.01, 0.02, ..., 0.1)
# Run 50 RRTs for each configuration.
# Initial and goal states close to (0.1, 0.1) and (0.9, 0.9) -- perturbed to free space if inside an obstacle.
# Maximum distance between vertices fixed at 0.1.
# Collision checking stays at same resolution of 1e-3.
# Goal detection eps stays fixed at 1e-2.
# Goal sample probability fixed at 0.01
initial_state = np.array([0.1, 0.1])
goal_state = np.array([0.9, 0.9])
number_of_runs = 20

number_of_obstacles_values = [5, 10, 15, 20, 25]
max_obstacle_radius_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
# Represent results as a matrix, with number_of_obstacles_values as rows, and max_obstacles_radius_values as columns
number_of_successes_array = np.zeros((len(number_of_obstacles_values), len(max_obstacle_radius_values)))
average_iterations_array = np.zeros((len(number_of_obstacles_values), len(max_obstacle_radius_values)))
dataset_number = 0
for (i, _) in enumerate(number_of_obstacles_values):
    for (j, _) in enumerate(max_obstacle_radius_values):
        print(f"Dataset {dataset_number}")
        success_array, iterations_array = run_multiple_constrained_rrt(initial_state, goal_state, number_of_runs,
                                                                       number_of_obstacles_values[i],
                                                                       max_obstacle_radius_values[j])
        print(success_array)
        print(iterations_array)
        # To get a scalar related to successes, we simply count the number of successes
        number_of_successes = np.count_nonzero(success_array)
        number_of_successes_array[i, j] = number_of_successes
        # To get a scalar related to required iterations, we take the average number of iterations
        average_iterations = np.mean(iterations_array)
        average_iterations_array[i, j] = average_iterations
        dataset_number += 1
print("DONE!")
print(number_of_successes_array)
print(average_iterations_array)
fig, ax = plt.subplots()
ax.matshow(number_of_successes_array)
for (i, _) in enumerate(max_obstacle_radius_values):
    for (j, _) in enumerate(number_of_obstacles_values):
        score = number_of_successes_array[j, i]
        ax.text(i, j, int(score), va='center', ha='center', c='w')
ax.set_title("Number of successes")
ax.set_xlabel("maximum obstacle radius")
ax.set_xticks(np.arange(len(max_obstacle_radius_values)))
ax.set_xticklabels([str(value) for value in max_obstacle_radius_values])
ax.set_ylabel("number of obstacles")
ax.set_yticks(np.arange(len(number_of_obstacles_values)))
ax.set_yticklabels([str(value) for value in number_of_obstacles_values])
plt.show()
fig, ax = plt.subplots()
ax.matshow(average_iterations_array)
for (i, _) in enumerate(max_obstacle_radius_values):
    for (j, _) in enumerate(number_of_obstacles_values):
        score = average_iterations_array[j, i]
        ax.text(i, j, f"{score:.2f}", va='center', ha='center', c='w')
ax.set_title("Average iterations")
ax.set_xlabel("maximum obstacle radius")
ax.set_xticks(np.arange(len(max_obstacle_radius_values)))
ax.set_xticklabels([str(value) for value in max_obstacle_radius_values])
ax.set_ylabel("number of obstacles")
ax.set_yticks(np.arange(len(number_of_obstacles_values)))
ax.set_yticklabels([str(value) for value in number_of_obstacles_values])
plt.show()
