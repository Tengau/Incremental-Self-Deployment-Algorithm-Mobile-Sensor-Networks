from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt


def generate_circle_points(center, radius, num_points=360, isFirst=False):
    """generate points in a certain radius around a center point

    Args:
        center (np arr): (2,) array for center
        radius (float): range of sensor
        num_points (int, optional): Number of angles to check. Defaults to 360.

    Returns:
        np array: numpy array of points in the circle (360, 2) (x, y)
    """
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = [
        (center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angles
    ]
    circle_points = np.array(points)
    return circle_points  # (360, 2)


def get_boundary_points(points):

    return reshaped_points


# cp = generate_circle_points(np.array([20, 30]), radius=5, num_points=360, isFirst=False)
# plot_test(cp)
