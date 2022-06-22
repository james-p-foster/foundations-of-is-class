import matplotlib.pyplot as plt
import numpy as np

from arena import Arena2D
from convex_obstacle import ConvexObstacle
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


def run_multiple_flytrap_rrt(initial_state, goal_state, number_of_runs, gap_size):
    upper_border_vertices = np.array([[0.2, 0.8],
                                      [0.8, 0.8],
                                      [0.8, 0.7],
                                      [0.2, 0.7]])
    upper_border = ConvexObstacle(upper_border_vertices)
    left_border_vertices = np.array([[0.2, 0.8],
                                     [0.3, 0.8],
                                     [0.3, 0.2],
                                     [0.2, 0.2]])
    left_border = ConvexObstacle(left_border_vertices)
    bottom_border_vertices = np.array([[0.2, 0.3],
                                       [0.8, 0.3],
                                       [0.8, 0.2],
                                       [0.2, 0.2]])
    bottom_border = ConvexObstacle(bottom_border_vertices)
    right_border_above_vertices = np.array([[0.7, 0.8],
                                            [0.8, 0.8],
                                            [0.8, 0.5 + gap_size / 2],
                                            [0.7, 0.5 + gap_size / 2]])
    right_border_above = ConvexObstacle(right_border_above_vertices)
    right_border_below_vertices = np.array([[0.7, 0.5 - gap_size / 2],
                                            [0.8, 0.5 - gap_size / 2],
                                            [0.8, 0.2],
                                            [0.7, 0.2]])
    right_border_below = ConvexObstacle(right_border_below_vertices)
    arena = Arena2D()
    arena.obstacles = [upper_border, left_border, bottom_border, right_border_above, right_border_below]

    success_array = np.zeros(number_of_runs, dtype=bool)
    iterations_array = np.zeros(number_of_runs)
    for i in range(number_of_runs):
        rrt = RRTPlanner(initial_state, goal_state, 0, 0, arena=arena)
        success, iterations = rrt.run()
        success_array[i] = success
        iterations_array[i] = iterations
    return success_array, iterations_array


# First set of experiments: unconstrained RRT.
# Vary RRT parameters to see their effect:
#   * maximum distance between vertices (0.1, 0.2, 0.3, 0.4, 0.5)
#   * goal sample probability (0.01, 0.02, ...,  0.1)
# Run 20 RRTs for each configuration.
# Fixed initial and goal states.
# No obstacles.
# Collision checking stays at same resolution of 1e-3.
# Goal detection eps stays fixed at 1e-2.
def unconstrained_experiments():
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
# Run 20 RRTs for each configuration.
# Initial and goal states close to (0.1, 0.1) and (0.9, 0.9) -- perturbed to free space if inside an obstacle.
# Maximum distance between vertices fixed at 0.1.
# Collision checking stays at same resolution of 1e-3.
# Goal detection eps stays fixed at 1e-2.
# Goal sample probability fixed at 0.01.
def constrained_experiments():
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


# Third set of experiments: adversarial environments.
# Create a flytrap environment where the initial state is inside a box, with a thin escape corridor to the outside of
# the box, where the goal state is located. Vary the size of the escape corridor:
#   * gap size (0.02, 0.04, 0.06, 0.08, 0.1)
# Run 20 RRTs for each configuration.
# Initial state = (0.5, 0.5) (middle of arena), with escape corridor to the right, and goal state = (0.1, 0.5) (back
# over on the left side of the arena).
# Maximum distance between vertices fixed at 0.1.
# Collision checking stays at same resolution of 1e-3.
# Goal detection eps stays fixed at 1e-2.
# Goal sample probability fixed at 0.01.
def adversarial_experiments():
    initial_state = np.array([0.5, 0.5])
    goal_state = np.array([0.1, 0.5])
    number_of_runs = 20

    gap_size_values = [0.02, 0.04, 0.06, 0.08, 0.1]
    # Represent results as a vector
    number_of_successes_array = np.zeros(len(gap_size_values))
    average_iterations_array = np.zeros(len(gap_size_values))
    dataset_number = 0
    for (i, _) in enumerate(gap_size_values):
        print(f"Dataset {dataset_number}")
        success_array, iterations_array = run_multiple_flytrap_rrt(initial_state, goal_state, number_of_runs,
                                                                   gap_size_values[i])
        print(success_array)
        print(iterations_array)
        # To get a scalar related to successes, we simply count the number of successes
        number_of_successes = np.count_nonzero(success_array)
        number_of_successes_array[i] = number_of_successes
        # To get a scalar related to required iterations, we take the average number of iterations
        average_iterations = np.mean(iterations_array)
        average_iterations_array[i] = average_iterations
        dataset_number += 1
    print("DONE!")
    print(number_of_successes_array)
    print(average_iterations_array)
    fig, ax = plt.subplots()
    ax.scatter(gap_size_values, number_of_successes_array, c='b')
    ax.grid()
    ax.set_title("Number of successes")
    ax.set_xlabel("gap size")
    ax.set_ylabel("number of successes")
    ax.set_yticks(np.arange(0, 21))
    plt.show()
    fig, ax = plt.subplots()
    ax.scatter(gap_size_values, average_iterations_array, c='b')
    ax.grid()
    ax.set_title("Average iterations")
    ax.set_xlabel("gap size")
    ax.set_ylabel("average iterations")
    plt.show()


# Finally, visualise some examples from the various experiments.
# Unconstrained:
def visualise_unconstrained_examples():
    initial_state = np.array([0.1, 0.1])
    goal_state = np.array([0.9, 0.9])

    maximum_distance_between_vertices = [0.1, 0.5]
    goal_sample_probability = [0.01, 0.1]
    for i in range(2):
        rrt = RRTPlanner(initial_state, goal_state, 0, 0,
                         maximum_distance_between_vertices=maximum_distance_between_vertices[i],
                         goal_sample_probability=goal_sample_probability[i],
                         plotting=True)
        rrt.run()


visualise_unconstrained_examples()
