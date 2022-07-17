import pybullet
import pybullet_data
import time
import numpy as np

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
    print(f"Number of joints: {number_of_joints}")
    random_joint_configuration = []
    for joint in range(number_of_joints):
        # Find upper and lower limits of the joint under consideration
        lower_limit = pybullet.getJointInfo(robot, joint)[8]  # check pybullet docs for getJointInfo(), [8] returns the lower limit from the list
        upper_limit = pybullet.getJointInfo(robot, joint)[9]  # check pybullet docs for getJointInfo(), [9] returns the upper limit from the list
        print(f"(Lower, Upper) limits of joint {joint}: ({lower_limit}, {upper_limit})")
        # Now, randomly sample a joint position within those limits
        position = np.random.uniform(lower_limit, upper_limit)
        print(position)
        random_joint_configuration.append(position)
    return random_joint_configuration

start_time = 0.0
end_time = 10.0
time_step = 0.01
number_of_simulation_steps = int((end_time - start_time) / time_step)

# Populate simulation with randomly positioned boxes
number_of_boxes = 10
boxes = create_list_of_boxes(number_of_boxes)

for t in range(number_of_simulation_steps):
    print(f"Simulation step: {t} of {number_of_simulation_steps}")
    # # Can either step the simulation (nots sure we need to do this, as only using kinematics)
    # pybullet.stepSimulation()
    # time.sleep(time_step)
    # # Or can just sleep
    # time.sleep(5)

    # Sample random joint configurations
    random_joint_configuration = sample_random_joint_configuration(kuka)
    number_of_joints = pybullet.getNumJoints(kuka)
    for joint in range(number_of_joints):
        pybullet.resetJointState(kuka, joint, random_joint_configuration[joint])
    time.sleep(0.5)

# End
pybullet.disconnect()

