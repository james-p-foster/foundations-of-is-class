import pybullet
import time
import numpy as np
import matplotlib.pyplot as plt

from arena import Arena
from rrt import JointSpaceRRT


# TODO: VARY THESE IN YOUR EXPERIMENTS!
# max_iterations = 1000
# goal_sample_probability = 0.02
# use_angular_difference = True
# norm_type = 2
# distance_threshold = 4


def run_multiple_joint_space_rrt(number_of_runs, number_of_boxes, goal_sample_probability, norm_type, distance_threshold, use_angular_difference):
    success_array = []
    total_rrt_time_array = []
    total_find_valid_joint_configuration_time_array = []
    proportion_of_time_spent_finding_valid_joint_configuration_array = []
    for run in range(number_of_runs):
        print(f"RUN {run} OF {number_of_runs}")
        arena = Arena(number_of_boxes, use_angular_difference=use_angular_difference, visualise_sim=False)
        arena.populate()

        rrt = JointSpaceRRT(arena,
                            goal_sample_probability=goal_sample_probability,
                            norm_for_distance_checking=norm_type, distance_threshold=distance_threshold,
                            use_angular_difference=use_angular_difference,
                            enable_smoothing=False,  # want to see ALL the vertices
                            playback_results=False)
        vertices_to_goal, total_rrt_time, total_find_valid_joint_configuration_time = rrt.run()
        proportion_of_time_spent_finding_valid_joint_configuration = total_find_valid_joint_configuration_time / total_rrt_time

        if vertices_to_goal is None:
            success_array.append(False)
        else:
            success_array.append(True)

        total_rrt_time_array.append(total_rrt_time)
        total_find_valid_joint_configuration_time_array.append(total_find_valid_joint_configuration_time)
        proportion_of_time_spent_finding_valid_joint_configuration_array.append(proportion_of_time_spent_finding_valid_joint_configuration)

    # Report the number of successes
    number_of_successes = success_array.count(True)
    # For timing information, find the means
    mean_total_rrt_time = np.mean(total_rrt_time)
    mean_total_find_valid_joint_configuration_time = np.mean(total_find_valid_joint_configuration_time)
    mean_proportion_of_time_spent_finding_valid_joint_configuration = np.mean(proportion_of_time_spent_finding_valid_joint_configuration_array)
    return number_of_successes, mean_total_rrt_time, mean_total_find_valid_joint_configuration_time, mean_proportion_of_time_spent_finding_valid_joint_configuration


def investigate_effect_of_goal_sample_probability(goal_sample_probability_values):
    number_of_boxes = 10
    norm_type = 2
    distance_threshold = 4
    use_angular_difference = True

    number_of_runs = 20
    number_of_successes_array = []
    mean_total_rrt_time_array = []
    mean_total_find_valid_joint_configuration_time_array = []
    mean_proportion_of_time_spent_finding_valid_joint_configuration_array = []
    for value in goal_sample_probability_values:
        print(f"GOAL SAMPLE PROBABILITY: {value}")
        run_stats = run_multiple_joint_space_rrt(number_of_runs, number_of_boxes, value, norm_type, distance_threshold, use_angular_difference)
        number_of_successes_array.append(run_stats[0])
        mean_total_rrt_time_array.append(run_stats[1])
        mean_total_find_valid_joint_configuration_time_array.append(run_stats[2])
        mean_proportion_of_time_spent_finding_valid_joint_configuration_array.append(run_stats[3])

    fig, ax = plt.subplots()
    ax.scatter(goal_sample_probability_values, number_of_successes_array, c='b')
    ax.grid()
    ax.set_title("Number of successes")
    ax.set_xlabel("goal sample probability")
    ax.set_ylabel("number of successes")
    ax.set_yticks(np.arange(0, number_of_runs+1))
    ax.axhline(number_of_runs, linestyle='-.', c='k')
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(goal_sample_probability_values, mean_total_rrt_time_array, c='b')
    ax.grid()
    ax.set_title("Average total rrt time")
    ax.set_xlabel("goal sample probability")
    ax.set_ylabel("average total rrt time [s]")
    plt.show()

    fig, ax = plt.subplots()
    table_data = [["Goal sample probability", "Proportion of time"]]
    for i in range(len(goal_sample_probability_values)):
        table_data.append([goal_sample_probability_values[i], round(mean_proportion_of_time_spent_finding_valid_joint_configuration_array[i], 2)])
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    ax.axis('off')
    fig.tight_layout()
    plt.show()

    print(number_of_successes_array)
    print(mean_total_rrt_time_array)
    print(mean_total_find_valid_joint_configuration_time_array)
    print(mean_proportion_of_time_spent_finding_valid_joint_configuration_array)


# Investigate varying of goal sample probability
goal_sample_probability_values = [0.02, 0.04, 0.06, 0.08, 0.1]
goal_sample_probability_stats = investigate_effect_of_goal_sample_probability(goal_sample_probability_values)

