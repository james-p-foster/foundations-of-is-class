import numpy as np
import matplotlib.pyplot as plt


class ConvexObstacle:
    def __init__(self, location: np.array, number_of_vertices: int, circle_radius: float):
        """
        Create a convex obstacle by sampling number_of_vertices vertices from a circle of radius
        circle_radius, sorting them by angle in counter-clockwise order, and then connecting them.

        Inspiration for this algorithm was taken from https://observablehq.com/@magrawala/random-convex-polygon
        """
        assert location.size == 2
        assert number_of_vertices > 2
        assert circle_radius > 0

        sampled_angles = np.random.uniform(0, 2 * np.pi, number_of_vertices)
        sorted_angles = np.sort(sampled_angles)

        x = np.empty(number_of_vertices)
        y = np.empty(number_of_vertices)
        # Cartesian position of vertices is constructed by converting polar coordinates (angle, circle_radius). Angle is
        # assumed to rotate counter-clockwise from positive x axis
        for (i, angle) in enumerate(sorted_angles):
            x[i] = location[0] + circle_radius * np.cos(angle)
            y[i] = location[1] + circle_radius * np.sin(angle)

        self.x = x
        self.y = y
        self.number_of_vertices = number_of_vertices

        self.location = location

    def check_collision(self, point: np.array):
        """
        Check if point is inside (and thus colliding) this obstacle.

        This collision-checking algorithm was taken from https://wrfranklin.org/Research/Short_Notes/pnpoly.html
        """
        assert point.size == 2

        i = 0
        j = self.number_of_vertices - 1
        collide = False
        while i < self.number_of_vertices:
            if ((self.y[i] > point[1]) != (self.y[j] > point[1])) and \
                    (point[0] < (self.x[j] - self.x[i]) * (point[1] - self.y[i]) / (self.y[j] - self.y[i]) + self.x[i]):
                collide = not collide
            j = i
            i += 1
        return collide

    def plot(self, fig, ax):
        """Plot an obstacle on to a given figure and axis set."""
        ax.plot(self.x, self.y, c='k')
        # Need to link the last vertex to the first one to complete the obstacle
        ax.plot([self.x[-1], self.x[0]], [self.y[-1], self.y[0]], c='k')
        return fig, ax

# FOR TESTING
if __name__ == "__main__":
    np.random.seed(45)

    # Test obstacle creation
    number_of_vertices = 20
    circle_radius = 5
    location = np.array([1.0, 1.0])
    obstacle = ConvexObstacle(location, number_of_vertices, circle_radius)
    fig, ax = plt.subplots()
    fig, ax = obstacle.plot(fig, ax)

    # Test collision checker
    # Candidate points between -6 and 6 in each dimension
    number_of_candidate_points = 100
    candidate_points_x = np.random.uniform(-6, 6, number_of_candidate_points)
    candidate_points_y = np.random.uniform(-6, 6, number_of_candidate_points)
    candidate_points = np.vstack((candidate_points_x, candidate_points_y))
    candidate_points = np.transpose(candidate_points)

    for i in range(number_of_candidate_points):
        collide = obstacle.check_collision(candidate_points[i])
        ax.scatter(candidate_points_x[i], candidate_points_y[i], c='r' if collide else 'g')
    plt.show()
