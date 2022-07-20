import numpy as np

from arena import Arena

class JointSpaceRRT:
    def __init__(self, arena: Arena, max_rrt_iterations: int, max_sample_iterations: int, goal_sample_probability: float,
                 goal_eps: float, number_of_joint_space_collision_nodes,
                 norm_for_distance_checking=3, distance_threshold=4, enable_smoothing=True):
        self.arena = arena

        self.start_joint_configuration = arena.start_joint_configuration
        self.goal_joint_configuration = arena.goal_joint_configuration
        self.goal_location = arena.goal_location

        self.vertices = [self.start_joint_configuration]
        self.parent_vertex_indices = []

        self.max_rrt_iterations = max_rrt_iterations
        self.max_sample_iterations = max_sample_iterations
        self.goal_sample_probability = goal_sample_probability
        self.goal_eps = goal_eps

        self.number_of_joint_space_collision_nodes = number_of_joint_space_collision_nodes

        if not any([norm_for_distance_checking == norm for norm in [2, 1, np.inf]]):
            raise Exception("Invalid norm type entered, use either 2, 1, or np.inf")
        self.norm_type = norm_for_distance_checking
        self.distance_threshold = distance_threshold

        self.enable_smoothing = enable_smoothing

    def run(self):
        is_finished = False
        for rrt_iter in range(self.max_rrt_iterations):
            if is_finished:
                break
            print(f"RRT ITERATION: {rrt_iter}")

            sampling_iter = 0
            for sampling_iter in range(self.max_sample_iterations):
                print(f"SAMPLING ITERATION: {sampling_iter}")

                if np.random.uniform() < self.goal_sample_probability:
                    sampled_joint_state = self.goal_joint_configuration
                    print(f"GOAL STATE: {sampled_joint_state}")
                else:
                    sampled_joint_state = self.arena.sample_random_joint_configuration()
                    print(f"RANDOM STATE: {sampled_joint_state}")

                # Collision checking
                self.arena.set_joint_configuration(sampled_joint_state)
                self.arena.update_simulation()
                if self.arena.check_collisions():
                    # TODO: possibly add number of rejections counter for plotting?
                    sampling_iter += 1
                    continue
                else:
                    print("FOUND VALID JOINT CONFIGURATION!")

                # Find nearest vertex in RRT graph according to a chosen norm
                distances = np.zeros(len(self.vertices))
                for i, vertex in enumerate(self.vertices):
                    # TODO: USING ANGULAR DIFFERENCE:
                    # distances[i] = np.linalg.norm(angular_difference(sampled_joint_state, vertex), norm_type)
                    # NOT:
                    distances[i] = np.linalg.norm(vertex - sampled_joint_state, self.norm_type)
                nearest_vertex_index = np.argmin(distances)
                # Distance thresholding
                distance_to_nearest_vertex = distances[nearest_vertex_index]
                if distance_to_nearest_vertex > self.distance_threshold:
                    # TODO: USING ANGULAR DIFFERENCE:
                    # difference_to_nearest_vertex = angular_difference(sampled_joint_state, vertices[nearest_vertex_index])
                    # NOT:
                    difference_to_nearest_vertex = self.vertices[nearest_vertex_index] - sampled_joint_state
                    unit_vector_to_nearest_vertex = difference_to_nearest_vertex / distance_to_nearest_vertex
                    sampled_joint_state = unit_vector_to_nearest_vertex * self.distance_threshold
                    print("THRESHOLD APPLIED TO SAMPLED STATE!")

                # We know by assumption that nearest vertex already in RRT graph has no collisions, and we've checked in this
                # iteration that there are no collisions on the sampled joint state, but what about the path in joint space
                # between them?
                # Solution: discretise distance between them into nodes and do forward kinematics collision checking on each one
                nearest_vertex = self.vertices[nearest_vertex_index]
                if self.arena.check_for_intermediate_collisions_in_joint_space(sampled_joint_state, nearest_vertex,
                                                                               self.number_of_joint_space_collision_nodes):
                    sampling_iter += 1
                    continue
                else:
                    print("COLLISION FREE PATH FOUND!")

                # Now it's verified to be collision free, add parent vertex information, and add to RRT graph
                self.parent_vertex_indices.append(nearest_vertex_index)
                self.vertices.append(sampled_joint_state)

                # Add lines in task space showing RRT evolution
                self.arena.draw_task_space_line_with_joint_space_inputs(self.vertices[nearest_vertex_index],
                                                                        sampled_joint_state,
                                                                        [0, 1, 0], 2)

                # Check if within tolerance of goal
                if self.arena.check_if_goal_is_reached_in_task_space(sampled_joint_state, self.goal_location):
                    print("DONE!")
                    is_finished = True
                    break
                else:
                    sampling_iter += 1

        if is_finished:
            # Now RRT is finished, highlight the line in red. Start with final vertex in vertices and work backwards to start
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
                self.arena.draw_task_space_line_with_joint_space_inputs(parent_vertex, vertex, [1, 0, 0], 5.0)
                vertex_index = parent_index
                vertex = parent_vertex
                if parent_index == 0:
                    print("FOUND PATH BACKWARD FROM GOAL VERTEX TO START VERTEX!")
                    is_first_vertex_reached = True

            # Must have 3 or more vertices (2 or more path segments) in order to smooth anything
            if self.enable_smoothing and len(end_effector_locations_backward_from_goal) >= 3:
                # Iterate through end effector locations and clip ones that have no collisions between them
                is_smoothing_finished = False
                i = 0
                while not is_smoothing_finished:
                    # Ray cast for possible collisions between every other node. If a collision free path exists between
                    # (say) the first and third node, via the triangle inequality this distance will be shorter than
                    # including the (now redundant) middle node
                    if not self.arena.check_for_intermediate_collisions_in_task_space(end_effector_locations_backward_from_goal[i],
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

                # Now vertices are smoothed, draw lines
                for i in range(len(vertices_backward_from_goal) - 1):
                    self.arena.draw_task_space_line_with_joint_space_inputs(vertices_backward_from_goal[i],
                                                                            vertices_backward_from_goal[i + 1],
                                                                            [0, 0, 1], 10)

            # To get the vertices from start to goal, reverse the list
            vertices_to_goal = list(reversed(vertices_backward_from_goal))

            # Finally, send vertex targets to sim and use position control to navigate from start to goal
            self.arena.play_rrt_results(vertices_to_goal)

        else:
            print("RRT FAILED!")