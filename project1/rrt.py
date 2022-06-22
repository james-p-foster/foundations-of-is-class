import numpy as np
import matplotlib.pyplot as plt
import math

from arena import Arena2D


class RRTPlanner:
    def __init__(self, initial_state: np.array, goal_state: np.array, num_obstacles: int, max_obstacle_radius: float,
                 arena: Arena2D = None,
                 maximum_iterations: int = 1000, maximum_distance_between_vertices: float = 0.1,
                 collision_checking_resolution: float = 1e-3,
                 goal_sample_probability: float = 0.01, goal_eps: float = 1e-2, plotting: bool = False):
        """
        Create an RRT planner with a given initial state, a desired goal state, and an arena
        (potentially with obstacles).
        """
        if arena == None:
            self.arena = Arena2D()
            self.arena.populate_with_random_obstacles(num_obstacles, max_obstacle_radius)
        else:
            self.arena = arena

        self.initial_state = self.arena.populate_state(initial_state)
        self.goal_state = self.arena.populate_state(goal_state)

        self.vertices = [self.initial_state]
        self.edges = []

        self.maximum_iterations = maximum_iterations
        self.maximum_distance_between_vertices = maximum_distance_between_vertices
        self.collision_checking_resolution = collision_checking_resolution
        self.goal_sample_probability = goal_sample_probability
        self.goal_eps = goal_eps

        self.plotting = plotting

    def sample_random_state(self):
        """
        Sample a random state to connect to the RRT graph by sampling uniformly over the search space. In order to
        add some heuristic information and bias the search slightly, we introduce a small chance of sampling the goal
        state, encouraging growth in that direction.
        """
        # There's a small probability we sample the goal state, this introduces a helpful bias in our search
        if np.random.uniform(0, 1) <= self.goal_sample_probability:
            random_state = self.goal_state
        # otherwise we sample uniformly over the search space
        else:
            lower_bound = np.array([self.arena.x_min, self.arena.y_min])
            upper_bound = np.array([self.arena.x_max, self.arena.y_max])
            random_state = np.random.uniform(lower_bound, upper_bound)
        return random_state

    def find_nearest_vertex_index(self, random_state: np.array):
        """Find the nearest vertex existing in the RRT graph to the input random_state."""
        distances = np.empty(len(self.vertices))
        for (i, vertex) in enumerate(self.vertices):
            distances[i] = np.linalg.norm(random_state - vertex)
        return np.argmin(distances)

    def apply_maximum_distance_threshold(self, nearest_vertex_index: np.array, random_state: np.array):
        """
        Apply a maximum relative_position threshold to the sampled random_state. If the sampled state is within the threshold,
        no action is taken.
        """
        # Find unit vector for vector between nearest_vertex and random_state
        nearest_vertex = self.vertices[nearest_vertex_index]
        relative_position = random_state - nearest_vertex
        unit_vector = relative_position / np.linalg.norm(relative_position)
        # If the norm of relative_position between the random_state and the nearest_vertex is smaller than the
        # maximum_distance_between_vertices, then do nothing. Otherwise, we multiply the unit vector we found earlier
        # by the maximum_distance_between_vertices
        if np.linalg.norm(relative_position) > self.maximum_distance_between_vertices:
            random_state = nearest_vertex + self.maximum_distance_between_vertices * unit_vector
        return random_state

    def create_list_of_collision_checking_nodes(self, nearest_vertex_index: np.array, random_state: np.array, resolution: float):
        """
        Creates a list of nodes from the vertex at nearest_vertex_index to the sampled random_state, with resolution
        giving the spacing of the nodes. This list is used for casting forward from the nearest vertex to the sampled
        state in order to find the first collision (if any), which will stop the casting and set the node.
        """
        relative_position = random_state - self.vertices[nearest_vertex_index]
        distance = np.linalg.norm(relative_position)
        number_of_nodes = math.floor(distance / resolution) + 1
        increment = relative_position / number_of_nodes

        list_of_collision_checking_nodes = np.zeros((number_of_nodes, 2))
        multiplier = 0
        for i in range(number_of_nodes):
            list_of_collision_checking_nodes[i] = self.vertices[nearest_vertex_index] + multiplier * increment
            multiplier += 1

        return list_of_collision_checking_nodes

    def find_free_node_via_collision_checking(self, list_of_collision_checking_nodes: np.array):
        """
        Given a list of nodes to collision check over (assumed ordered from closest to nearest vertex to closest to
        sampled state), find the first collision, and return the node previous to it in free space.
        """
        for (i, node) in enumerate(list_of_collision_checking_nodes):
            collision_boolean_vector = [obstacle.check_collision(node) for obstacle in self.arena.obstacles]
            if any(collision_boolean_vector):
                return list_of_collision_checking_nodes[i-1]
        # If no collisions are found on any nodes, return the last one in the list (closest to sampled state)
        return list_of_collision_checking_nodes[-1]

    def run(self):
        success = False
        i = 0
        while i < self.maximum_iterations:
            if self.plotting and i % 20 == 0:
                fig, ax = plt.subplots()
                self.plot(fig, ax)
                plt.title(f"Iteration {i}")
                plt.show()
            # First, sample a random state (possibly the goal state thanks to heuristic)
            random_state = self.sample_random_state()
            # Next, find the index of the nearest vertex
            nearest_vertex_index = self.find_nearest_vertex_index(random_state)
            # Next, apply the threshold of maximum distance between vertices to the candidate. Now, anything we apply
            # backtracking to is guaranteed to be within this threshold
            random_state = self.apply_maximum_distance_threshold(nearest_vertex_index, random_state)
            # Next, apply collision checking to the candidate, casting a ray as close as possible to it before colliding
            list_of_collision_checking_nodes = self.create_list_of_collision_checking_nodes(nearest_vertex_index,
                                                                                            random_state,
                                                                                            self.collision_checking_resolution)
            random_state = self.find_free_node_via_collision_checking(list_of_collision_checking_nodes)
            # Append the candidate to the RRT graph
            self.vertices.append(random_state)
            self.edges.append(np.concatenate((self.vertices[nearest_vertex_index], random_state)))
            # To check if the goal has been reached, we only need to check the most recently created vertex, otherwise,
            # if it was an earlier vertex, we would have exited earlier
            if any([np.linalg.norm(self.goal_state - vertex) < self.goal_eps for vertex in self.vertices]):
                print(f"Success! Number of iterations: {i}")
                success = True
                if self.plotting:
                    fig, ax = plt.subplots()
                    self.plot(fig, ax)
                    plt.title(f"Iteration {i}")
                    plt.show()
                break
            i += 1
        return success, i

    def plot(self, fig, ax):
        self.arena.plot(fig, ax)
        for edge in self.edges:
            plt.plot([edge[0], edge[2]], [edge[1], edge[3]], c='b')
        plt.scatter(self.initial_state[0], self.initial_state[1], c='r', s=50)
        plt.scatter(self.goal_state[0], self.goal_state[1], c='g', s=50)
        ax.scatter([vertex[0] for vertex in self.vertices[1:]], [vertex[1] for vertex in self.vertices[1:]], c='b', s=5)
        return fig, ax
