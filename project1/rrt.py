import numpy as np

from arena import Arena2D


class RRTPlanner:
    def __init__(self, initial_state: np.array, goal_state: np.array, arena: Arena2D):
        """
        Create an RRT planner with a given initial state, a desired goal state, and an arena
        (potentially with obstacles).
        """
        assert arena.x_min <= initial_state[0] <= arena.x_max
        assert arena.y_min <= initial_state[1] <= arena.y_max
        assert arena.x_min <= goal_state[0] <= arena.x_max
        assert arena.y_min <= goal_state[1] <= arena.y_max

        self.arena = arena

        self.initial_state = initial_state
        self.goal_state = goal_state

        self.vertices = [initial_state]
        self.edges = []

        self.maximum_iterations = 2500
        self.maximum_distance_between_vertices = 0.1
        self.goal_eps = 1e-3
        self.goal_sample_probability = 0.01

    def sample_random_state(self):
        # There's a small probability we sample the goal state, this introduces a helpful bias in our search
        if np.random.uniform(0, 1) <= self.goal_sample_probability:
            random_state = self.goal_state
        # otherwise we sample uniformly over the search space
        else:
            lower_bound = np.array([self.arena.x_min, self.arena.y_min])
            upper_bound = np.array([self.arena.x_max, self.arena.y_max])
            random_state = np.random.uniform(lower_bound, upper_bound)
        return random_state

    def find_nearest_vertex_index(self, random_state):
        distances = np.empty(len(self.vertices))
        for (i, vertex) in enumerate(self.vertices):
            distances[i] = np.linalg.norm(random_state - vertex)
        return np.argmin(distances)

    def create_new_vertex(self, nearest_vertex_index, random_state, maximum_distance_between_vertices):
        # Find unit vector for vector between nearest_vertex and random_state
        nearest_vertex = self.vertices[nearest_vertex_index]
        distance = random_state - nearest_vertex
        unit_vector = distance / np.linalg.norm(distance)
        # If distance between the random_state and the nearest_vertex is smaller than the
        # maximum_distance_between_vertices, then we can just add that random_state to the graph
        if np.linalg.norm(distance) < maximum_distance_between_vertices:
            new_vertex = random_state
        # otherwise, we multiply the unit vector we found earlier by the maximum_distance_between_vertices
        else:
            new_vertex = nearest_vertex + maximum_distance_between_vertices * unit_vector
        self.vertices.append(new_vertex)
        self.edges.append(np.concatenate((nearest_vertex, new_vertex)))
        return

    def run(self):
        i = 0
        terminate = False
        while i < self.maximum_iterations:
            random_state = self.sample_random_state()
            nearest_vertex_index = self.find_nearest_vertex_index(random_state)
            self.create_new_vertex(nearest_vertex_index, random_state, self.maximum_distance_between_vertices)
            # Check distances to goal
            for vertex in list(self.vertices):
                if np.linalg.norm(self.goal_state - vertex) < self.goal_eps:
                    print(f"Success! Number of iterations: {i}")
                    terminate = True
                    break
            if terminate:
                break
            i += 1
