import pybullet
import pybullet_data
import numpy as np
import time

from utils import angular_difference

class Arena:
    def __init__(self, number_of_boxes, x_limits, y_limits, z_limits,
                 playback_number_of_timesteps=50, playback_dt=0.02, use_angular_difference=False):
        # Pybullet plumbing
        physics_client = pybullet.connect(pybullet.GUI)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setPhysicsEngineParameter(enableFileCaching=0)
        pybullet.setGravity(0, 0, -9.8)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, False)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, True)
        pybullet.resetDebugVisualizerCamera(cameraDistance=1.400, cameraYaw=58.000, cameraPitch=-42.200,
                                            cameraTargetPosition=(0.0, 0.0, 0.0))

        # Robot
        self.robot = pybullet.loadURDF("assets/kuka/kuka.urdf", basePosition=[0, 0, 0.02], useFixedBase=True)
        self.number_of_joints = pybullet.getNumJoints(self.robot)
        self.start_joint_configuration = np.zeros(self.number_of_joints)

        # Add ground plane
        self.plane = pybullet.loadURDF("plane.urdf")

        # Limits
        self.x_min = x_limits[0]
        self.x_max = x_limits[1]
        self.y_min = y_limits[0]
        self.y_max = y_limits[1]
        self.z_min = z_limits[0]
        self.z_max = z_limits[1]

        # Boxes
        self.number_of_boxes = number_of_boxes
        self.boxes = []

        # Goal
        self.goal_location = None
        self.goal_joint_configuration = None

        # For simulating results
        self.playback_number_of_timesteps = playback_number_of_timesteps
        self.playback_dt = playback_dt

        self.use_angular_difference = use_angular_difference

    def update_simulation(self):
        pybullet.stepSimulation()

    def check_collision_with_boxes(self):
        collision_with_boxes = []
        for box in range(len(self.boxes)):
            collision_data = pybullet.getContactPoints(self.robot, self.boxes[box])
            if len(collision_data) == 0:  # empty tuple, no collision
                collision_with_boxes.append(False)
            else:
                collision_with_boxes.append(True)
        return collision_with_boxes

    def check_collision_with_ground(self):
        collision_data = pybullet.getContactPoints(self.robot, self.plane)
        if len(collision_data) == 0:  # empty tuple, no collision
            return False
        else:
            return True

    def check_collision_with_self(self):
        collision_data = pybullet.getContactPoints(self.robot, self.robot)
        if len(collision_data) == 0:  # empty tuple, no collision
            return False
        else:
            return True

    def check_collisions(self):
        is_self_collision = self.check_collision_with_self()
        is_ground_collision = self.check_collision_with_ground()
        is_box_collision = any(self.check_collision_with_boxes())
        if any([is_self_collision, is_ground_collision, is_box_collision]):
            return True
        else:
            return False

    def check_for_intermediate_collisions_in_joint_space(self, start_state, end_state, number_of_collision_checking_nodes):
        if self.use_angular_difference:
            difference = angular_difference(start_state, end_state)
        else:
            difference = end_state - start_state
        collision_node_increments = difference / number_of_collision_checking_nodes
        for i in range(number_of_collision_checking_nodes):
            collision_checking_node = start_state + (i + 1) * collision_node_increments
            self.set_joint_configuration(collision_checking_node)
            self.update_simulation()
            if self.check_collisions():
                return True
        return False

    def create_boxes(self):
        boxes = []
        for i in range(self.number_of_boxes):
            # Randomly sample a position for the box
            x = np.random.uniform(self.x_min, self.x_max)
            y = np.random.uniform(self.y_min, self.y_max)
            z = np.random.uniform(self.z_min, self.z_max)
            box = pybullet.loadURDF("assets/box/box.urdf", basePosition=[x, y, z], useFixedBase=True)
            boxes.append(box)
        return boxes

    def remove_boxes(self):
        for box in self.boxes:
            pybullet.removeBody(box)

    def populate_with_boxes(self):
        # Populate simulation with randomly positioned boxes -- they must not cause collision with the
        # starting configuration
        is_box_placement_valid = False
        while not is_box_placement_valid:
            self.boxes = self.create_boxes()
            self.set_joint_configuration(self.start_joint_configuration)
            self.update_simulation()
            # If the box placement causes collisions with the robot's starting configuration, sample a new set of boxes
            if self.check_collisions():
                # TODO: possibly add number of rejections counter for plotting?
                self.remove_boxes()
                continue
            else:
                print("FOUND VALID BOX PLACEMENT!")
                is_box_placement_valid = True

    def sample_goal_location(self):
        x = np.random.uniform(self.x_min, self.x_max)
        y = np.random.uniform(self.y_min, self.y_max)
        z = np.random.uniform(self.z_min, self.z_max)
        return np.array([x, y, z])

    def check_if_goal_is_reached_in_task_space(self, configuration, goal_location):
        eps = 1e-1  # This eps will stay as a magic number, too much hassle to move elsewhere
        end_effector_in_task_space = self.get_end_effector_location_in_task_space(configuration)
        if np.linalg.norm(end_effector_in_task_space - goal_location) > eps:
            return False
        else:
            return True

    def create_goal_marker(self, position):
        visual = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 1])
        marker = pybullet.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1,
                                          baseVisualShapeIndex=visual)
        return marker

    def populate_with_goal(self):
        is_goal_location_valid = False
        while not is_goal_location_valid:
            self.goal_location = self.sample_goal_location()
            self.goal_joint_configuration = np.array(
                pybullet.calculateInverseKinematics(self.robot, self.number_of_joints-1, self.goal_location))
            # If we can't actually reach the goal, we want to sample again
            if not self.check_if_goal_is_reached_in_task_space(self.goal_joint_configuration, self.goal_location):
                continue
            # Check for collisions
            self.update_simulation()
            if self.check_collisions():
                # TODO: possibly add number of rejections counter for plotting?
                continue
            print("FOUND VALID GOAL LOCATION!")
            print(f"Goal location: {self.goal_location}")
            print(f"Goal joint configuration: {self.goal_joint_configuration}")
            goal = self.create_goal_marker(self.goal_location)
            is_goal_location_valid = True

    def populate(self):
        # Important to populate boxes first so they can be used to inform the placement of the goal marker
        self.populate_with_boxes()
        self.populate_with_goal()

    def set_joint_configuration(self, joint_configuration):
        for joint in range(len(joint_configuration)):
            pybullet.resetJointState(self.robot, joint, joint_configuration[joint])

    def sample_random_joint_configuration(self):
        random_joint_configuration = []
        for joint in range(self.number_of_joints):
            # Find upper and lower limits of the joint under consideration
            # [8] returns the lower limit from the list, [9] returns the upper limit from the list
            lower_limit = pybullet.getJointInfo(self.robot, joint)[8]
            upper_limit = pybullet.getJointInfo(self.robot, joint)[9]
            # Now, randomly sample a joint position within those limits
            position = np.random.uniform(lower_limit, upper_limit)
            random_joint_configuration.append(position)
        return np.array(random_joint_configuration)

    def get_end_effector_location_in_task_space(self, joint_configuration):
        self.set_joint_configuration(joint_configuration)
        self.update_simulation()
        return np.array(pybullet.getLinkState(self.robot, self.number_of_joints-1)[0])

    def draw_task_space_line_with_joint_space_inputs(self, start_joint_configuration, end_joint_configuration, colour,
                                                     thickness):
        end_effector_location_start = self.get_end_effector_location_in_task_space(start_joint_configuration)
        end_effector_location_end = self.get_end_effector_location_in_task_space(end_joint_configuration)
        pybullet.addUserDebugLine(end_effector_location_start, end_effector_location_end, colour, thickness)

    def check_for_intermediate_collisions_in_task_space(self, start_location, end_location):
        ray_data = pybullet.rayTest(start_location, end_location)
        # [0][0] gives the pybullet id of the body in collision. If none, then it returns -1 automatically
        if ray_data[0][0] <= self.robot:
            # If ray test id is smaller than or equal to robot's, we know it's either "hit" itself
            # or is collision-free
            return False
        else:
            return True

    def play_rrt_results(self, vertices):
        self.set_joint_configuration(self.start_joint_configuration)
        for vertex in vertices:
            pybullet.setJointMotorControlArray(self.robot, range(self.number_of_joints), pybullet.POSITION_CONTROL, vertex)
            for timestep in range(self.playback_number_of_timesteps):
                self.update_simulation()
                time.sleep(self.playback_dt)
