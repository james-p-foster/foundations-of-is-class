import pybullet
import pybullet_data
import time
import numpy as np


def create_list_of_boxes(number_of_boxes):
    boxes = []
    for i in range(number_of_boxes):
        # Randomly sample a position for the box
        x_limits = (-1.0, 1.0)
        y_limits = (-1.0, 1.0)
        z_limits = (0, 1)  # don't want the box underground
        x = np.random.uniform(x_limits[0], x_limits[1])
        y = np.random.uniform(y_limits[0], y_limits[1])
        z = np.random.uniform(z_limits[0], z_limits[1])
        box = pybullet.loadURDF("assets/box/box.urdf", basePosition=[x, y, z], useFixedBase=True)
        boxes.append(box)
    return boxes


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


def check_collision_with_boxes(robot, boxes):
    collision_with_boxes = []
    for box in range(number_of_boxes):
        collision_data = pybullet.getContactPoints(robot, boxes[box])
        if len(collision_data) == 0:  # empty tuple, no collision
            collision_with_boxes.append(False)
        else:
            collision_with_boxes.append(True)
    return collision_with_boxes


def check_collision_with_ground(robot, plane):
    collision_data = pybullet.getContactPoints(robot, plane)
    if len(collision_data) == 0:  # empty tuple, no collision
        return False
    else:
        return True


def check_collision_with_self(robot):
    collision_data = pybullet.getContactPoints(robot, robot)
    if len(collision_data) == 0:  # empty tuple, no collision
        return False
    else:
        return True


def check_if_goal_is_reached_in_task_space(robot, goal_location):
    eps = 1e-1
    end_effector_in_world = np.array(pybullet.getLinkState(robot, 6)[0])
    if np.linalg.norm(end_effector_in_world - goal_location) > eps:
        return False
    else:
        return True


def sample_goal_location():
    x_limits = (-0.5, 0.5)
    y_limits = (-0.5, 0.5)
    z_limits = (0, 1)  # don't want the box underground
    x = np.random.uniform(x_limits[0], x_limits[1])
    y = np.random.uniform(y_limits[0], y_limits[1])
    z = np.random.uniform(z_limits[0], z_limits[1])
    return np.array([x, y, z])


def create_goal_marker(position):
    visual = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 1])
    marker = pybullet.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visual)
    return marker


# Set up pybullet instance
physicsClient = pybullet.connect(pybullet.GUI)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
pybullet.setPhysicsEngineParameter(enableFileCaching=0)
pybullet.setGravity(0, 0, -9.8)
pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, False)
pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, True)
pybullet.resetDebugVisualizerCamera(cameraDistance=1.400, cameraYaw=58.000, cameraPitch=-42.200,
                                    cameraTargetPosition=(0.0, 0.0, 0.0))
kuka = pybullet.loadURDF("assets/kuka/kuka.urdf", basePosition=[0, 0, 0.02], useFixedBase=True)
plane = pybullet.loadURDF("plane.urdf")

number_of_joints = pybullet.getNumJoints(kuka)

# Start joint configuration
start_joint_configuration = np.array([0, 0, 0, 0, 0, 0, 0])

# Populate simulation with randomly positioned boxes -- they must not cause collision with the starting configuration
# of the robot however
number_of_boxes = 10
is_box_placement_valid = False
while not is_box_placement_valid:
    boxes = create_list_of_boxes(number_of_boxes)
    # TODO: make below a function
    for joint in range(number_of_joints):
        pybullet.resetJointState(kuka, joint, start_joint_configuration[joint])
    pybullet.stepSimulation()
    # TODO: make below a function
    ground_collision = check_collision_with_ground(kuka, plane)
    self_collision = check_collision_with_self(kuka)
    box_collision = any(check_collision_with_boxes(kuka, boxes))
    if any([ground_collision, self_collision, box_collision]):
        # TODO: possibly add number of rejections counter for plotting?
        # TODO: make below a function
        for box in boxes:
            pybullet.removeBody(box)
        continue
    else:
        print("FOUND VALID BOX PLACEMENT!")
        is_box_placement_valid = True

is_goal_location_valid = False
while not is_goal_location_valid:
    # Sample goal location
    goal_location = sample_goal_location()
    goal_joint_configuration = np.array(pybullet.calculateInverseKinematics(kuka, number_of_joints - 1, goal_location))
    # TODO: make below a function
    for joint in range(number_of_joints):
        pybullet.resetJointState(kuka, joint, goal_joint_configuration[joint])
    if not check_if_goal_is_reached_in_task_space(kuka, goal_location):
        continue
    # Check if joint angles that result in goal location cause a collision
    # TODO: make below a function
    pybullet.stepSimulation()
    ground_collision = check_collision_with_ground(kuka, plane)
    self_collision = check_collision_with_self(kuka)
    box_collision = any(check_collision_with_boxes(kuka, boxes))
    if any([ground_collision, self_collision, box_collision]):
        # TODO: possibly add number of rejections counter for plotting?
        continue
    print("FOUND VALID GOAL LOCATION!")
    print(f"Goal location: {goal_location}")
    print(f"Goal joint configuration: {goal_joint_configuration}")
    goal = create_goal_marker(goal_location)
    is_goal_location_valid = True

vertices = [start_joint_configuration]
edges = []
parent_vertex_indices = []

# main rrt loop
is_finished = False
max_iterations = 100
goal_sample_probability = 0.1  # TODO: make this parameter
for i in range(max_iterations):
    if is_finished:
        break
    print(f"OUTER ITERATION: {i}")
    # Sample new joint state or sample goal joint configuration (found via IK)
    j = 0
    to_finish_array = []  # TODO: just for viewing how close to goal are we, remove when good goal tolerance is set
    valid_joint_configuration_found = False
    while valid_joint_configuration_found is False:
        print(f"INNER ITERATION: {j}")
        if np.random.uniform() < 0.1:
            sampled_joint_state = goal_joint_configuration
            print(f"GOAL STATE: {sampled_joint_state}")
        else:
            sampled_joint_state = sample_random_joint_configuration(kuka)
            print(f"RANDOM STATE: {sampled_joint_state}")

        # Collision checking
        # TODO: make below a function
        for joint in range(number_of_joints):
            pybullet.resetJointState(kuka, joint, sampled_joint_state[joint])
        pybullet.stepSimulation()
        # TODO: make below a function
        ground_collision = check_collision_with_ground(kuka, plane)
        self_collision = check_collision_with_self(kuka)
        box_collision = any(check_collision_with_boxes(kuka, boxes))
        if any([ground_collision, self_collision, box_collision]):
            # TODO: possibly add number of rejections counter for plotting?
            j += 1
            continue
        else:
            print("FOUND VALID JOINT CONFIGURATION!")

        # Distance checking -- find nearest vertex in graph DO DIFFERENT NORMS
        # TODO: NEED TO CONSIDER CIRCULAR DISTANCES HERE! WRAP THE ANGLES!
        def angular_difference(angle_array1, angle_array2):
            assert angle_array1.shape == angle_array2.shape

            length = angle_array1.shape[0]
            difference = np.zeros(length)
            for i, (angle1, angle2) in enumerate(zip(angle_array1, angle_array2)):
                if np.abs(angle1) + np.abs(angle2) > np.pi:
                    difference[i] = 2*np.pi - angle1 - angle2
                else:
                    difference[i] = angle2 - angle1
            return difference

        distances = np.zeros(len(vertices))
        norm_type = 2
        # norm_type = 1
        # norm_type = np.inf
        for i, vertex in enumerate(vertices):
            distances[i] = np.linalg.norm(angular_difference(sampled_joint_state, vertex), norm_type)
        nearest_vertex_index = np.argmin(distances)
        nearest_vertex_difference = angular_difference(sampled_joint_state, vertices[nearest_vertex_index])

        # We know by assumption that nearest vertex already in RRT graph has no collisions, and we've checked in this
        # iteration that there are no collisions on the sampled joint state, but what about the path in joint space between
        # them?
        # Solution: discretise distance between them and do forward kinematics collision checking
        def check_for_intermediate_collisions(sampled_joint_state, nearest_vertex, number_of_collision_checking_nodes):
            nearest_vertex_difference = angular_difference(sampled_joint_state, nearest_vertex)
            collision_node_increments = nearest_vertex_difference / number_of_collision_checking_nodes
            for i in range(number_of_collision_checking_nodes):
                collision_checking_node = sampled_joint_state + (i+1) * collision_node_increments
                # TODO: make below a function
                for joint in range(number_of_joints):
                    pybullet.resetJointState(kuka, joint, collision_checking_node[joint])
                pybullet.stepSimulation()
                # TODO: make below a function
                ground_collision = check_collision_with_ground(kuka, plane)
                self_collision = check_collision_with_self(kuka)
                box_collision = any(check_collision_with_boxes(kuka, boxes))
                if any([ground_collision, self_collision, box_collision]):
                    return True
            return False

        number_of_collision_checking_nodes = 100  # TODO: magic number
        nearest_vertex = vertices[nearest_vertex_index]
        if check_for_intermediate_collisions(sampled_joint_state, nearest_vertex, number_of_collision_checking_nodes):
            j += 1
            continue
        else:
            print("COLLISION FREE PATH FOUND!")

        # Now it's verified to be collision free add parent vertex information and add to RRT graph
        parent_vertex_indices.append(nearest_vertex_index)
        vertices.append(sampled_joint_state)

        # Check if within tolerance of goal
        for joint in range(number_of_joints):
            pybullet.resetJointState(kuka, joint, sampled_joint_state[joint])
        if check_if_goal_is_reached_in_task_space(kuka, goal_location):
            print("DONE!")
            is_finished = True
            break
        j += 1
        # TODO: use debug lines to plot how the RRT expands in task space, even though it searches in joint space!


# TODO: project questions
#   * do both joint space and task space RRT? Joint space will be 7 dof, task space will be 3 dof but will need inverse kinematics
#   * when calculating the nearest vertex in RRT to find what vertex to link to, what is a good distance metric in joint space? 2 norm, 1 norm, inf norm?
#   * will probably need some angle wrapping capability, e.g. -pi/2 and +pi/2 are actually the same angle
#   * two seperate sims? One for visualising the result and the other for collision checking? Check GUI and DIRECT server options
#   * when making RRT classes (both task space and joint space), be sure to create a list denoting the parent of each node that is added to the RRT graph -- this will et you easily backtrack from the goal to form a path
#   * do smoothing on the RRT result? Draw a ray between consecutive nodes on the goal path, see if it is collision-free. If it is, by the definition of the triangle inequality it is shortwer, so replace it as the path (only works with task space RRT where collision checking is easy?)
#   * how to do collision checking in a joint space RRT? Impossible directly -- will need to do forward kinematics each time and do a collision check, interesting to check how this goes vs. the inverse kinematics required for task space rrt


# End
pybullet.disconnect()

