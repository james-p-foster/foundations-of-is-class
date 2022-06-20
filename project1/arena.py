import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import qmc

from convex_obstacle import ConvexObstacle


class Arena2D:
    def __init__(self):
        """Create a 2D box arena on (0,1) x (0,1)."""
        self.x_min = 0
        self.x_max = 1
        self.y_min = 0
        self.y_max = 1

        self.obstacles = []

    def populate_with_random_obstacles(self, num_obstacles: int, max_obstacle_radius: float):
        """
        Populate the arena with num_obstacles convex obstacles of max radius max_obstacles radius.

        We place the generated obstacles according to a scrambled Halton sampler, which has been shown to provide more
        uniform coverage of the space than uniform random sampling. See https://en.wikipedia.org/wiki/Halton_sequence
        """
        halton_sampler = qmc.Halton(d=2, scramble=True)
        obstacle_locations = halton_sampler.random(num_obstacles)

        for i in range(num_obstacles):
            # We'll randomly draw the number of vertices per obstacle from the discrete uniform distribution over (3,10)
            obstacle = ConvexObstacle(obstacle_locations[i], np.random.randint(3, 10), max_obstacle_radius)
            self.obstacles.append(obstacle)

    def populate_state(self, desired_state: np.array) -> np.array:
        """
        Populate the arena with a desired state of the agent (can be used for both initial and goal states). This method
        checks whether the user-given desired_state is inside an obstacle or not. If it is, it perturbs the initial
        state by samples drawn from a growing normal distribution.
        """
        assert desired_state.size == 2

        initial_state = desired_state
        perturbation_standard_deviation = 0

        i = 0
        max_iterations = 10
        while i < max_iterations:
            # Quicker to check if any is in collision, rather than all not in collision
            if not any([obstacle.check_collision(desired_state) for obstacle in self.obstacles]):
                print(f"State set to: {initial_state} after {i} perturbations")
                return initial_state
            else:
                perturbation_standard_deviation += 0.1
                initial_state = desired_state + np.random.normal(0.0, perturbation_standard_deviation, 2)
                i += 1
        print(f"Failed to populate state after {max_iterations} iterations, raising exception")
        raise Exception("Maximum number of iterations reached for populating state")

    def plot(self, fig, ax):
        """Plot the arena and its obstacles."""
        ax.set_xlim((self.x_min, self.x_max))
        ax.set_ylim((self.y_min, self.y_max))

        for obstacle in self.obstacles:
            fig, ax = obstacle.plot(fig, ax)
        return fig, ax


if __name__ == "__main__":
    # First, create an arena without obstacles (the plot will be empty!)
    arena = Arena2D()
    fig, ax = plt.subplots()
    fig, ax = arena.plot(fig, ax)
    plt.show()

    # Next, fill the arena with obstacles
    arena.populate_with_random_obstacles(20, 0.1)
    fig, ax = plt.subplots()
    fig, ax = arena.plot(fig, ax)
    plt.show()