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


def create_goal():
    # TODO: copied from the create_boxes() function, probably want the smae limits shared between the two! Think about
    #   this when you put these functions in a class
    x_limits = (-1.0, 1.0)
    y_limits = (-1.0, 1.0)
    z_limits = (0, 1)  # don't want the box underground
    x = np.random.uniform(x_limits[0], x_limits[1])
    y = np.random.uniform(y_limits[0], y_limits[1])
    z = np.random.uniform(z_limits[0], z_limits[1])
    visual = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 1])
    marker = pybullet.createMultiBody(basePosition=[x, y, z], baseCollisionShapeIndex=-1, baseVisualShapeIndex=visual)
    return marker

def get_goal_location(goal_id):
    goal_location = np.array(pybullet.getBasePositionAndOrientation(goal_id)[0])
    return goal_location


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
# box1 = pybullet.loadURDF("assets/box/box.urdf", basePosition=[1, 1, 0.5], useFixedBase=True)
# box2 = pybullet.loadURDF("assets/box/box.urdf", basePosition=[2, 0, 1.0], useFixedBase=True)

start_time = 0.0
end_time = 10.0
time_step = 0.01
number_of_simulation_steps = int((end_time - start_time) / time_step)

# Populate simulation with randomly positioned boxes
number_of_boxes = 10
boxes = create_list_of_boxes(number_of_boxes)

# Find joint angles need for collision with first box, and use them set the arm to be in collision
box_position = pybullet.getBasePositionAndOrientation(boxes[0])[0]  # check pybullet docs for getBasePositionAndOrientation(), [] returns the position from the tuple
print(box_position)
number_of_joints = pybullet.getNumJoints(kuka)
desired_joint_configuration_for_collision = pybullet.calculateInverseKinematics(kuka, number_of_joints-1, np.array(box_position))  # -1 because of 0 indexing
for joint in range(number_of_joints):
    pybullet.resetJointState(kuka, joint, desired_joint_configuration_for_collision[joint])
# pybullet.setJointMotorControlArray(kuka, list(range(number_of_joints)), pybullet.POSITION_CONTROL, desired_joint_configuration_for_collision)

# Create goal marker
goal = create_goal()
goal_location = get_goal_location(goal)
print(f"Goal location: {goal_location}")

# Start joint configuration
start_joint_configuration = np.array([0, 0, 0, 0, 0, 0, 0])

vertices = [start_joint_configuration]
edges = []
parent_vertex_indices = []

# main rrt loop
max_iterations = 100
goal_sample_probability = 0.1  # TODO: make this parameter
goal_joint_configuration = np.array(pybullet.calculateInverseKinematics(kuka, number_of_joints-1, goal_location))
for i in range(max_iterations):
    # Sample new joint state or sample goal joint configuration (found via IK)
    valid_joint_configuration_found = False
    while valid_joint_configuration_found is False:
        if np.random.uniform() < 0.1:
            sampled_joint_state = goal_joint_configuration
            print(f"GOAL STATE: {sampled_joint_state}")
        else:
            sampled_joint_state = sample_random_joint_configuration(kuka)
            print(f"RANDOM STATE: {sampled_joint_state}")

        # Collision checking
        for joint in range(number_of_joints):
            pybullet.resetJointState(kuka, joint, sampled_joint_state[joint])
        pybullet.stepSimulation()
        ground_collision = check_collision_with_ground(kuka, plane)
        self_collision = check_collision_with_self(kuka)
        box_collision = any(check_collision_with_boxes(kuka, boxes))
        if any([ground_collision, self_collision, box_collision]):
            # TODO: possibly add number of rejections counter for plotting?
            continue
        else:
            valid_joint_configuration_found = True
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



    # Now it's verified to be collision free add parent vertex information and add to RRT graph
    parent_vertex_indices.append(nearest_vertex_index)
    vertices.append(sampled_joint_state)

    # Check if within tolerance of goal
    if np.linalg.norm(sampled_joint_state - goal_joint_configuration) < 1e-2:  # TODO: magic number
        print("DONE!")



# TODO: project questions
#   * do both joint space and task space RRT? Joint space will be 7 dof, task space will be 3 dof but will need inverse kinematics
#   * when calculating the nearest vertex in RRT to find what vertex to link to, what is a good distance metric in joint space? 2 norm, 1 norm, inf norm?
#   * will probably need some angle wrapping capability, e.g. -pi/2 and +pi/2 are actually the same angle
#   * two seperate sims? One for visualising the result and the other for collision checking? Check GUI and DIRECT server options
#   * when making RRT classes (both task space and joint space), be sure to create a list denoting the parent of each node that is added to the RRT graph -- this will et you easily backtrack from the goal to form a path
#   * do smoothing on the RRT result? Draw a ray between consecutive nodes on the goal path, see if it is collision-free. If it is, by the definition of the triangle inequality it is shortwer, so replace it as the path (only works with task space RRT where collision checking is easy?)
#   * how to do collision checking in a joint space RRT? Impossible directly -- will need to do forward kinematics each time and do a collision check, interesting to check how this goes vs. the inverse kinematics required for task space rrt

for t in range(number_of_simulation_steps):
    print(f"Simulation step: {t} of {number_of_simulation_steps}")
    # # Can either step the simulation (nots sure we need to do this, as only using kinematics)
    pybullet.stepSimulation()
    box_collisions = check_collision_with_boxes(kuka, boxes)
    print(f"Collision with boxes: {box_collisions}")
    ground_collisions = check_collision_with_ground(kuka, plane)
    print(f"Collision with ground: {ground_collisions}")
    self_collisions = check_collision_with_self(kuka)
    print(f"Collision with self: {self_collisions}")
    time.sleep(time_step)
    # # Or can just sleep
    # time.sleep(5)

    # Sample random joint configurations
    # random_joint_configuration = sample_random_joint_configuration(kuka)
    # number_of_joints = pybullet.getNumJoints(kuka)
    # for joint in range(number_of_joints):
    #     pybullet.resetJointState(kuka, joint, random_joint_configuration[joint])
    # time.sleep(0.1)

# End
pybullet.disconnect()

