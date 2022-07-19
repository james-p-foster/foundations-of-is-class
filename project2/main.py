import pybullet
import time
import numpy as np

from arena import Arena


def sample_random_joint_configuration(robot):
    number_of_joints = pybullet.getNumJoints(robot)
    random_joint_configuration = []
    for joint in range(number_of_joints):
        # Find upper and lower limits of the joint under consideration
        lower_limit = pybullet.getJointInfo(robot, joint)[8]  # check pybullet docs for getJointInfo(), [8] returns the lower limit from the list
        upper_limit = pybullet.getJointInfo(robot, joint)[9]  # check pybullet docs for getJointInfo(), [9] returns the upper limit from the list
        # Now, randomly sample a joint position within those limits
        position = np.random.uniform(lower_limit, upper_limit)
        random_joint_configuration.append(position)
    return np.array(random_joint_configuration)

def angular_difference(angle_array1, angle_array2):
    # TODO: do some explaining of the whole enumerate zip and the maths behind this angular difference
    assert angle_array1.shape == angle_array2.shape

    length = angle_array1.shape[0]
    difference = np.zeros(length)
    for i, (angle1, angle2) in enumerate(zip(angle_array1, angle_array2)):
        if np.abs(angle1) + np.abs(angle2) > np.pi:
            difference[i] = 2*np.pi - angle1 - angle2
        else:
            difference[i] = angle2 - angle1
    return difference


def check_for_intermediate_collisions(robot, start_state, end_state, number_of_collision_checking_nodes):
    # IF USING ANGULAR DIFFERENCE:
    # difference = angular_difference(start_state, end_state)
    # NOT:
    difference = end_state - start_state
    collision_node_increments = difference / number_of_collision_checking_nodes
    for i in range(number_of_collision_checking_nodes):
        collision_checking_node = start_state + (i + 1) * collision_node_increments
        set_joint_configuration(robot, collision_checking_node)
        update_simulation()
        if check_collisions(robot, plane, boxes):
            return True
    return False


def draw_task_space_line_with_joint_space_inputs(robot, start_joint_configuration, end_joint_configuration, colour, thickness):
    set_joint_configuration(robot, start_joint_configuration)
    update_simulation()
    end_effector_location_start = np.array(pybullet.getLinkState(robot, 6)[0])
    set_joint_configuration(kuka, end_joint_configuration)
    update_simulation()
    end_effector_location_end = np.array(pybullet.getLinkState(robot, 6)[0])
    pybullet.addUserDebugLine(end_effector_location_start, end_effector_location_end, colour, thickness)


x_limits = (-0.8, 0.8)
y_limits = (-0.8, 0.8)
z_limits = (0, 0.8)  # don't want the box underground
number_of_boxes = 10

arena = Arena(number_of_boxes, x_limits, y_limits, z_limits)
kuka = arena.robot
plane = arena.plane

# Populate boxes
arena.populate_with_boxes()

# Populate with goal
arena.populate_with_goal()

# TODO: WORKING UP TO HERE!!!!!!!!!!!!! PUT BREAKPOINT BELOW
vertices = [start_joint_configuration]
edges = []
parent_vertex_indices = []

# main rrt loop
is_finished = False
max_rrt_iterations = 20  # TODO: for now, to be upped when doing stat studies
max_sample_iterations = 100
goal_sample_probability = 0.02  # TODO: make this parameter
for rrt_iter in range(max_rrt_iterations):
    if is_finished:
        break
    print(f"RRT ITERATION: {rrt_iter}")

    sampling_iter = 0
    for sampling_iter in range(max_sample_iterations):
        print(f"SAMPLING ITERATION: {sampling_iter}")

        if np.random.uniform() < 0.1:
            sampled_joint_state = goal_joint_configuration
            print(f"GOAL STATE: {sampled_joint_state}")
        else:
            sampled_joint_state = sample_random_joint_configuration(kuka)
            print(f"RANDOM STATE: {sampled_joint_state}")

        # Collision checking
        set_joint_configuration(kuka, sampled_joint_state)
        update_simulation()
        if check_collisions(kuka, plane, boxes):
            # TODO: possibly add number of rejections counter for plotting?
            sampling_iter += 1
            continue
        else:
            print("FOUND VALID JOINT CONFIGURATION!")

        # Find nearest vertex in RRT graph according to a chosen norm
        distances = np.zeros(len(vertices))
        norm_type = 2  # TODO: should be a parameter
        # norm_type = 1
        # norm_type = np.inf
        for i, vertex in enumerate(vertices):
            # USING ANGULAR DIFFERENCE:
            # distances[i] = np.linalg.norm(angular_difference(sampled_joint_state, vertex), norm_type)
            # NOT:
            distances[i] = np.linalg.norm(vertex - sampled_joint_state, norm_type)
        nearest_vertex_index = np.argmin(distances)
        # TODO: play around with a lot of different norm types and thresholds
        # Distance thresholding
        threshold_2_norm = 4  # average distance seems to be about 4, set it lower to limit growth more # TODO: should be a parameter
        # threshold_1_norm = 10
        # threshold_inf_norm = 1
        distance_to_nearest_vertex = distances[nearest_vertex_index]
        if distance_to_nearest_vertex > threshold_2_norm:
            # USING ANGULAR DIFFERENCE:
            # difference_to_nearest_vertex = angular_difference(sampled_joint_state, vertices[nearest_vertex_index])
            # NOT:
            difference_to_nearest_vertex = vertices[nearest_vertex_index] - sampled_joint_state
            unit_vector_to_nearest_vertex = difference_to_nearest_vertex / distance_to_nearest_vertex
            sampled_joint_state = unit_vector_to_nearest_vertex * threshold_2_norm
            print("THRESHOLD APPLIED TO SAMPLED STATE!")
            is_threshold_applied = True  # TODO: not used, just for debugging purposes

        # We know by assumption that nearest vertex already in RRT graph has no collisions, and we've checked in this
        # iteration that there are no collisions on the sampled joint state, but what about the path in joint space
        # between them?
        # Solution: discretise distance between them into nodes and do forward kinematics collision checking on each one
        number_of_collision_checking_nodes = 100  # TODO: magic number
        nearest_vertex = vertices[nearest_vertex_index]
        if check_for_intermediate_collisions(kuka, sampled_joint_state, nearest_vertex, number_of_collision_checking_nodes):
            sampling_iter += 1
            continue
        else:
            print("COLLISION FREE PATH FOUND!")

        # Now it's verified to be collision free, add parent vertex information, and add to RRT graph
        parent_vertex_indices.append(nearest_vertex_index)
        vertices.append(sampled_joint_state)

        # Add lines in task space showing RRT evolution
        draw_task_space_line_with_joint_space_inputs(kuka, vertices[nearest_vertex_index], sampled_joint_state, [0, 1, 0], 2)

        # Check if within tolerance of goal
        set_joint_configuration(kuka, sampled_joint_state)
        if check_if_goal_is_reached_in_task_space(kuka, goal_location):
            print("DONE!")
            is_finished = True
            break

        sampling_iter += 1

if is_finished:
    # Now RRT is finished, highlight the line in red. Start with final vertex in vertices and work backwards to start
    vertex_index = len(vertices)-1
    vertex = vertices[vertex_index]
    vertices_backward_from_goal = [vertex]
    set_joint_configuration(kuka, vertex)
    update_simulation()
    end_effector_locations_backward_from_goal = [np.array(pybullet.getLinkState(kuka, 6)[0])]
    is_first_vertex_reached = False
    while not is_first_vertex_reached:
        parent_index = parent_vertex_indices[vertex_index-1]  # -1 because the list of parent indices is always 1 shorter than the list of vertices
        parent_vertex = vertices[parent_index]
        set_joint_configuration(kuka, parent_vertex)
        update_simulation()
        end_effector_location_parent_vertex = np.array(pybullet.getLinkState(kuka, 6)[0])
        vertices_backward_from_goal.append(parent_vertex)
        end_effector_locations_backward_from_goal.append(end_effector_location_parent_vertex)
        # if not enable_smoothing:
        draw_task_space_line_with_joint_space_inputs(kuka, parent_vertex, vertex, [1, 0, 0], 5.0)
        vertex_index = parent_index
        vertex = parent_vertex
        if parent_index == 0:
            print("FOUND PATH BACKWARD FROM GOAL VERTEX TO START VERTEX!")
            is_first_vertex_reached = True

    enable_smoothing = True
    if enable_smoothing and len(end_effector_locations_backward_from_goal) >= 3:  # Must have 3 or more vertices (2 or more path segments) in order to smooth anything
        # Iterate through end effector locations and clip ones that have no collisions between them
        is_smoothing_finished = False
        i = 0
        while not is_smoothing_finished:
            ray_data = pybullet.rayTest(end_effector_locations_backward_from_goal[i],
                                        end_effector_locations_backward_from_goal[i+2])
            if ray_data[0][0] <= kuka:  # If first return value of ray test is an id smaller than or equal to robot's, we know it's either "hit" itself or is collision-free, so we can smooth
                # Remove both inner task space location and the joint space vertex it corresponds to
                end_effector_locations_backward_from_goal.pop(i+1)
                vertices_backward_from_goal.pop(i+1)
            else:
                i += 1
            # Finally, check if we're at the penultimate (due to indexing) element in the list. If so, we're finished.
            # Otherwise, increment.
            if len(end_effector_locations_backward_from_goal) < 2 or i == len(end_effector_locations_backward_from_goal)-2:
                is_smoothing_finished = True

        # Now we've smoothed the vertices, draw lines
        for i in range(len(vertices_backward_from_goal)-1):
            draw_task_space_line_with_joint_space_inputs(kuka,
                                                        vertices_backward_from_goal[i],
                                                        vertices_backward_from_goal[i+1],  # it's +1 this time instead of +2 as we've clipped the redundant middle!
                                                        [0, 0, 1], 10)

    # To get the vertices from start to goal, reverse the list
    vertices_to_goal = list(reversed(vertices_backward_from_goal))

    # Finally, send vertex targets to sim and use position control to navigate from start to goal
    num_timesteps = 50
    set_joint_configuration(kuka, start_joint_configuration)
    for vertex in vertices_to_goal:
        pybullet.setJointMotorControlArray(kuka, range(number_of_joints), pybullet.POSITION_CONTROL, vertex)
        for timestep in range(num_timesteps):
            update_simulation()
            time.sleep(0.02)

else:
    print("RRT FAILED!")


# TODO: project questions
#   * do both joint space and task space RRT? Joint space will be 7 dof, task space will be 3 dof but will need inverse kinematics
#   * when calculating the nearest vertex in RRT to find what vertex to link to, what is a good distance metric in joint space? 2 norm, 1 norm, inf norm?
#   * will probably need some angle wrapping capability, e.g. -pi/2 and +pi/2 are actually the same angle
#   * two seperate sims? One for visualising the result and the other for collision checking? Check GUI and DIRECT server options
#   * when making RRT classes (both task space and joint space), be sure to create a list denoting the parent of each node that is added to the RRT graph -- this will et you easily backtrack from the goal to form a path
#   * do smoothing on the RRT result? Draw a ray between consecutive nodes on the goal path, see if it is collision-free. If it is, by the definition of the triangle inequality it is shortwer, so replace it as the path (only works with task space RRT where collision checking is easy?)
#   * how to do collision checking in a joint space RRT? Impossible directly -- will need to do forward kinematics each time and do a collision check, interesting to check how this goes vs. the inverse kinematics required for task space rrt


# End
# Pause for a while so you can observe result
time.sleep(5)
pybullet.disconnect()

