import numpy as np
import time

from arena import Arena
from utils import angular_difference


class JointSpaceRRT:
    def __init__(self, arena: Arena, max_iterations: int, goal_sample_probability: float,
                 number_of_joint_space_collision_nodes,
                 norm_for_distance_checking=2, distance_threshold=4, enable_smoothing=True, use_angular_difference=False):
        self.arena = arena

        self.start_joint_configuration = arena.start_joint_configuration
        self.goal_joint_configuration = arena.goal_joint_configuration
        self.goal_location = arena.goal_location

        self.vertices = [self.start_joint_configuration]
        self.parent_vertex_indices = []

        self.max_iterations = max_iterations
        self.goal_sample_probability = goal_sample_probability

        self.number_of_joint_space_collision_nodes = number_of_joint_space_collision_nodes

        if not any([norm_for_distance_checking == norm for norm in [2, 1, np.inf]]):
            raise Exception("Invalid norm type entered, use either 2, 1, or np.inf")
        self.norm_type = norm_for_distance_checking
        self.distance_threshold = distance_threshold

        self.enable_smoothing = enable_smoothing

        self.use_angular_difference = use_angular_difference

    def sample_candidate_joint_configuration(self):
        if np.random.uniform() < self.goal_sample_probability:
            sampled_joint_configuration = self.goal_joint_configuration
            print(f"SAMPLED GOAL STATE: {sampled_joint_configuration}")
        else:
            sampled_joint_configuration = self.arena.sample_random_joint_configuration()
            print(f"SAMPLED RANDOM STATE: {sampled_joint_configuration}")
        return sampled_joint_configuration

    def find_nearest_vertex_index(self, candidate_joint_configuration):
        distances = np.zeros(len(self.vertices))
        for i, vertex in enumerate(self.vertices):
            if self.use_angular_difference:
                distances[i] = np.linalg.norm(angular_difference(candidate_joint_configuration, vertex), self.norm_type)
            else:
                distances[i] = np.linalg.norm(vertex - candidate_joint_configuration, self.norm_type)
        return np.argmin(distances)

    def apply_distance_threshold(self, nearest_vertex_index, candidate_joint_configuration):
        distance_to_nearest_vertex = np.linalg.norm(self.vertices[nearest_vertex_index] - candidate_joint_configuration,
                                                    self.norm_type)
        if distance_to_nearest_vertex > self.distance_threshold:
            if self.use_angular_difference:
                difference_to_nearest_vertex = angular_difference(candidate_joint_configuration,
                                                                  self.vertices[nearest_vertex_index])
            else:
                difference_to_nearest_vertex = self.vertices[nearest_vertex_index] - candidate_joint_configuration
            unit_vector_to_nearest_vertex = difference_to_nearest_vertex / distance_to_nearest_vertex
            print("THRESHOLD APPLIED TO SAMPLED STATE!")
            return unit_vector_to_nearest_vertex * self.distance_threshold
        else:
            return candidate_joint_configuration

    def trace_solution_backward_from_goal(self):
        vertex_index = len(self.vertices) - 1
        vertex = self.vertices[vertex_index]
        vertices_backward_from_goal = [vertex]
        end_effector_locations_backward_from_goal = [self.arena.get_end_effector_location_in_task_space(vertex)]
        is_first_vertex_reached = False
        while not is_first_vertex_reached:
            # -1 because the list of parent indices is always 1 shorter than the list of vertices
            parent_index = self.parent_vertex_indices[vertex_index - 1]
            parent_vertex = self.vertices[parent_index]
            end_effector_location_parent_vertex = self.arena.get_end_effector_location_in_task_space(parent_vertex)
            vertices_backward_from_goal.append(parent_vertex)
            end_effector_locations_backward_from_goal.append(end_effector_location_parent_vertex)
            # self.arena.draw_task_space_line_with_joint_space_inputs(parent_vertex, vertex, [1, 0, 0], 5.0)
            vertex_index = parent_index
            vertex = parent_vertex
            if parent_index == 0:
                print("FOUND PATH BACKWARD FROM GOAL VERTEX TO START VERTEX!")
                is_first_vertex_reached = True
        return vertices_backward_from_goal, end_effector_locations_backward_from_goal

    def draw_solution_lines(self, vertices_backward_from_goal):
        for i in range(len(vertices_backward_from_goal)-1):
            self.arena.draw_task_space_line_with_joint_space_inputs(vertices_backward_from_goal[i], vertices_backward_from_goal[i + 1], [1, 0, 0], 5.0)


    def apply_solution_smoothing(self, vertices_backward_from_goal, end_effector_locations_backward_from_goal):
        # Iterate through end effector locations and clip ones that have no collisions between them
        is_smoothing_finished = False
        i = 0
        while not is_smoothing_finished:
            # Ray cast for possible collisions between every other node. If a collision free path exists between
            # (say) the first and third node, via the triangle inequality this distance will be shorter than
            # including the (now redundant) middle node
            if not self.arena.check_for_intermediate_collisions_in_task_space(
                    end_effector_locations_backward_from_goal[i],
                    end_effector_locations_backward_from_goal[i + 2]):
                # Remove both redundant task space location and the joint space vertex it corresponds to
                end_effector_locations_backward_from_goal.pop(i + 1)
                vertices_backward_from_goal.pop(i + 1)
            else:
                i += 1
            # End if only two nodes left or at penultimate (due to indexing) element, increment otherwise
            if len(end_effector_locations_backward_from_goal) < 2 or \
                    i == len(end_effector_locations_backward_from_goal) - 2:
                is_smoothing_finished = True
        return vertices_backward_from_goal, end_effector_locations_backward_from_goal

    def draw_smoothed_solution_lines(self, vertices_backward_from_goal):
        for i in range(len(vertices_backward_from_goal) - 1):
            self.arena.draw_task_space_line_with_joint_space_inputs(vertices_backward_from_goal[i],
                                                                    vertices_backward_from_goal[i + 1],
                                                                    [0, 0, 1], 10)

    def run(self):
        is_finished = False
        rrt_time_start = time.time()
        total_find_valid_joint_configuration_time = 0
        for iter in range(self.max_iterations):
            print(f"ITERATION: {iter}")

            is_valid_sample_joint_configuration_found = False
            find_valid_joint_configuration_time_start = time.time()
            while not is_valid_sample_joint_configuration_found:
                sampled_joint_configuration = self.sample_candidate_joint_configuration()

                # Collision checking
                self.arena.set_joint_configuration(sampled_joint_configuration)
                self.arena.update_simulation()
                if self.arena.check_collisions():
                    # TODO: possibly add number of rejections counter for plotting?
                    continue
                else:
                    print("FOUND VALID JOINT CONFIGURATION!")

                # Find nearest vertex in RRT graph according to a chosen norm
                nearest_vertex_index = self.find_nearest_vertex_index(sampled_joint_configuration)
                # Distance thresholding
                sampled_joint_configuration = self.apply_distance_threshold(nearest_vertex_index, sampled_joint_configuration)

                # We know by assumption that nearest vertex already in RRT graph has no collisions, and we've checked
                # that there are no collisions on the sampled joint configuration, but what about the path in joint space
                # between them? Solution: discretise distance between them into nodes and do forward kinematics
                # collision checking on each one
                nearest_vertex = self.vertices[nearest_vertex_index]
                if not self.arena.check_for_intermediate_collisions_in_joint_space(sampled_joint_configuration, nearest_vertex,
                                                                                   self.number_of_joint_space_collision_nodes):
                    print("COLLISION FREE PATH FOUND!")
                    is_valid_sample_joint_configuration_found = True
            find_valid_joint_configuration_time_end = time.time()
            total_find_valid_joint_configuration_time += find_valid_joint_configuration_time_end - find_valid_joint_configuration_time_start

            # Now it's verified to be a valid joint configuration, add parent vertex information, and add to RRT graph
            self.parent_vertex_indices.append(nearest_vertex_index)
            self.vertices.append(sampled_joint_configuration)

            # Add lines in task space showing RRT evolution
            self.arena.draw_task_space_line_with_joint_space_inputs(self.vertices[nearest_vertex_index],
                                                                    sampled_joint_configuration,
                                                                    [0, 1, 0], 2)

            # Check if within tolerance of goal
            if self.arena.check_if_goal_is_reached_in_task_space(sampled_joint_configuration, self.goal_location):
                print("DONE!")
                is_finished = True
                break
            else:
                iter += 1
        rrt_time_end = time.time()
        total_rrt_time = rrt_time_end - rrt_time_start

        if is_finished:
            # Find the solution by working backwards from the goal
            vertices_backward_from_goal, end_effector_locations_backward_from_goal = self.trace_solution_backward_from_goal()
            self.draw_solution_lines(vertices_backward_from_goal)

            # Must have 3 or more vertices (2 or more path segments) in order to smooth anything
            if self.enable_smoothing and len(end_effector_locations_backward_from_goal) >= 3:
                vertices_backward_from_goal, end_effector_locations_backward_from_goal = self.apply_solution_smoothing(
                    vertices_backward_from_goal, end_effector_locations_backward_from_goal)
                self.draw_smoothed_solution_lines(vertices_backward_from_goal)

            # To get the vertices from start to goal, reverse the list
            vertices_to_goal = list(reversed(vertices_backward_from_goal))

            # Send vertex targets to sim and use position control to navigate from start to goal
            self.arena.play_rrt_results(vertices_to_goal)

            return vertices_to_goal, total_rrt_time, total_find_valid_joint_configuration_time
        else:
            print("RRT FAILED!")
            return None, total_rrt_time, total_find_valid_joint_configuration_time
