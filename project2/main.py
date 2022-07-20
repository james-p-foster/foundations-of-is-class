import numpy as np
import matplotlib.pyplot as plt

from arena import Arena
from rrt import JointSpaceRRT


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
    median_total_rrt_time = np.median(total_rrt_time)
    mean_total_find_valid_joint_configuration_time = np.mean(total_find_valid_joint_configuration_time)
    mean_proportion_of_time_spent_finding_valid_joint_configuration = np.mean(proportion_of_time_spent_finding_valid_joint_configuration_array)
    return number_of_successes, mean_total_rrt_time, median_total_rrt_time, mean_total_find_valid_joint_configuration_time, mean_proportion_of_time_spent_finding_valid_joint_configuration


def investigate_effect_of_goal_sample_probability(goal_sample_probability_values):
    number_of_boxes = 10
    norm_type = 2
    distance_threshold = 4
    use_angular_difference = True

    number_of_runs = 20
    number_of_successes_array = []
    mean_total_rrt_time_array = []
    median_total_rrt_time_array = []
    mean_total_find_valid_joint_configuration_time_array = []
    mean_proportion_of_time_spent_finding_valid_joint_configuration_array = []
    for value in goal_sample_probability_values:
        print(f"GOAL SAMPLE PROBABILITY: {value}")
        run_stats = run_multiple_joint_space_rrt(number_of_runs, number_of_boxes, value, norm_type, distance_threshold, use_angular_difference)
        number_of_successes_array.append(run_stats[0])
        mean_total_rrt_time_array.append(run_stats[1])
        median_total_rrt_time_array.append(run_stats[2])
        mean_total_find_valid_joint_configuration_time_array.append(run_stats[3])
        mean_proportion_of_time_spent_finding_valid_joint_configuration_array.append(run_stats[4])

    fig, ax = plt.subplots()
    ax.scatter(goal_sample_probability_values, number_of_successes_array, c='b')
    ax.grid()
    ax.set_title("Number of successes")
    ax.set_xlabel("goal sample probability")
    ax.set_ylabel("number of successes")
    ax.set_yticks(np.arange(0, number_of_runs+5))
    ax.axhline(number_of_runs, linestyle='-.', c='k')
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(goal_sample_probability_values, mean_total_rrt_time_array, c='b')
    ax.grid()
    ax_mirror = ax.twinx()
    ax_mirror.scatter(goal_sample_probability_values, median_total_rrt_time_array, c='r', marker='x')
    ax.set_title("Total rrt time statistics")
    ax.set_xlabel("goal sample probability")
    ax.set_ylabel("average total rrt time [s]", c='b')
    ax_mirror.set_ylabel("median total rrt time [s]", c='r')
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


def investigate_effect_of_angular_difference():
    number_of_boxes = 10
    goal_sample_probability = 0.02
    norm_type = 2
    distance_threshold = 4

    angular_difference_settings = [True, False]

    number_of_runs = 20
    number_of_successes_array = []
    mean_total_rrt_time_array = []
    median_total_rrt_time_array = []
    mean_total_find_valid_joint_configuration_time_array = []
    mean_proportion_of_time_spent_finding_valid_joint_configuration_array = []
    for setting in angular_difference_settings:
        print(f"USING ANGULAR DIFFERENCE: {setting}")
        run_stats = run_multiple_joint_space_rrt(number_of_runs, number_of_boxes, goal_sample_probability, norm_type, distance_threshold, setting)
        number_of_successes_array.append(run_stats[0])
        mean_total_rrt_time_array.append(run_stats[1])
        median_total_rrt_time_array.append(run_stats[2])
        mean_total_find_valid_joint_configuration_time_array.append(run_stats[3])
        mean_proportion_of_time_spent_finding_valid_joint_configuration_array.append(run_stats[4])

    fig, ax = plt.subplots()
    ax.scatter(angular_difference_settings, number_of_successes_array, c='b')
    ax.grid()
    ax.set_title("Number of successes")
    ax.set_xlabel("use angular difference")
    ax.set_ylabel("number of successes")
    ax.set_yticks(np.arange(0, number_of_runs+5))
    ax.axhline(number_of_runs, linestyle='-.', c='k')
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(angular_difference_settings, mean_total_rrt_time_array, c='b')
    ax.grid()
    ax_mirror = ax.twinx()
    ax_mirror.scatter(angular_difference_settings, median_total_rrt_time_array, c='r', marker='x')
    ax.set_title("Total rrt time statistics")
    ax.set_xlabel("use angular difference")
    ax.set_ylabel("average total rrt time [s]", c='b')
    ax_mirror.set_ylabel("median total rrt time [s]", c='r')
    plt.show()

    fig, ax = plt.subplots()
    table_data = [["Use angular difference", "Proportion of time"]]
    for i in range(len(angular_difference_settings)):
        table_data.append([angular_difference_settings[i], round(mean_proportion_of_time_spent_finding_valid_joint_configuration_array[i], 2)])
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    ax.axis('off')
    fig.tight_layout()
    plt.show()

    print(number_of_successes_array)
    print(mean_total_rrt_time_array)
    print(mean_total_find_valid_joint_configuration_time_array)
    print(mean_proportion_of_time_spent_finding_valid_joint_configuration_array)


def investigate_effect_of_distance_threshold_for_given_norm(norm_type, distance_threshold_values):
    number_of_boxes = 10
    goal_sample_probability = 0.02
    use_angular_difference = True

    number_of_runs = 20
    number_of_successes_array = []
    mean_total_rrt_time_array = []
    median_total_rrt_time_array = []
    mean_total_find_valid_joint_configuration_time_array = []
    mean_proportion_of_time_spent_finding_valid_joint_configuration_array = []
    for value in distance_threshold_values:
        print(f"DISTANCE THRESHOLD: {value}")
        run_stats = run_multiple_joint_space_rrt(number_of_runs, number_of_boxes, goal_sample_probability, norm_type, value, use_angular_difference)
        number_of_successes_array.append(run_stats[0])
        mean_total_rrt_time_array.append(run_stats[1])
        median_total_rrt_time_array.append(run_stats[2])
        mean_total_find_valid_joint_configuration_time_array.append(run_stats[3])
        mean_proportion_of_time_spent_finding_valid_joint_configuration_array.append(run_stats[4])

    fig, ax = plt.subplots()
    ax.scatter(distance_threshold_values, number_of_successes_array, c='b')
    ax.grid()
    ax.set_title("Number of successes")
    ax.set_xlabel("distance threshold")
    ax.set_ylabel("number of successes")
    ax.set_yticks(np.arange(0, number_of_runs+5))
    ax.axhline(number_of_runs, linestyle='-.', c='k')
    plt.xscale("log")
    ax.set_xticks(distance_threshold_values)
    ax.set_xticklabels(distance_threshold_values)
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(distance_threshold_values, mean_total_rrt_time_array, c='b')
    ax.grid()
    ax_mirror = ax.twinx()
    ax_mirror.scatter(distance_threshold_values, median_total_rrt_time_array, c='r', marker='x')
    ax.set_title("Total rrt time statistics")
    ax.set_xlabel("distance threshold")
    ax.set_ylabel("average total rrt time [s]", c='b')
    ax_mirror.set_ylabel("median total rrt time [s]", c='r')
    plt.xscale("log")
    ax.set_xticks(distance_threshold_values)
    ax.set_xticklabels(distance_threshold_values)
    plt.show()

    fig, ax = plt.subplots()
    table_data = [["Distance threshold", "Proportion of time"]]
    for i in range(len(distance_threshold_values)):
        table_data.append([distance_threshold_values[i], round(mean_proportion_of_time_spent_finding_valid_joint_configuration_array[i], 2)])
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    ax.axis('off')
    fig.tight_layout()
    plt.show()

    print(number_of_successes_array)
    print(mean_total_rrt_time_array)
    print(mean_total_find_valid_joint_configuration_time_array)
    print(mean_proportion_of_time_spent_finding_valid_joint_configuration_array)


# Investigate varying of goal sample probability
# goal_sample_probability_values = [0.02, 0.04, 0.06, 0.08, 0.1]
# investigate_effect_of_goal_sample_probability(goal_sample_probability_values)

# Investigate varying use of angular difference
# investigate_effect_of_angular_difference()

# Investigate the use of the 2 norm in distance thresholding
distance_threshold_values = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
investigate_effect_of_distance_threshold_for_given_norm(2, distance_threshold_values)

# Investigate the use of the 1 norm in distance thresholding
distance_threshold_values = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
investigate_effect_of_distance_threshold_for_given_norm(1, distance_threshold_values)

# Investigate the use of the inf norm in distance thresholding
distance_threshold_values = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
investigate_effect_of_distance_threshold_for_given_norm(np.inf, distance_threshold_values)
