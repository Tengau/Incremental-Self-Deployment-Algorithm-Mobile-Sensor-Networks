from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
import numpy as np


# boundary
def alpha_shape(points, alpha):
    if len(points) < 4:
        return Polygon(points)

    tri = Delaunay(points)
    edges = set()
    edge_points = []

    # Loop through each triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]

        # Compute length of triangle edges
        a = np.linalg.norm(pa - pb)
        b = np.linalg.norm(pb - pc)
        c = np.linalg.norm(pc - pa)

        # Calculate triangle area
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))

        # Circumradius
        circum_r = a * b * c / (4.0 * area)

        if circum_r < 1.0 / alpha:
            edges.add((ia, ib))
            edges.add((ib, ic))
            edges.add((ic, ia))

    for i, j in edges:
        edge_points.append((tuple(points[i]), tuple(points[j])))

    m = polygonize(edge_points)
    return unary_union(list(m))


def generate_circle_points(center, radius, num_points=360):
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


def bresenham_line(x0, y0, x1, y1):
    """Code sourced from LidarMapper
    Returns the grid cells between two points in the occupancy grid map using the Bresenham's line algorithm.

    :param x0, y0: The coordinates of the first point.
    :param x1, y1: The coordinates of the second point.

    :return: A 2D array where each row represents the (x, y) coordinates of a cell in the occupancy
            grid that lies on the line between the start and end point.
            (N, 2)
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))
    return np.array(points)  # (m, 2) m = hit points
