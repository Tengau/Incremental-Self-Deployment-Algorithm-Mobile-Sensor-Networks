import numpy as np
import cv2
import matplotlib.pyplot as plt


# from pathlib import Path
# from config_parser import load_config
# import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d

curr_map = cv2.imread(
    "../data/input/isaac_sample_ros_nav_hospital_map.png", flags=cv2.IMREAD_GRAYSCALE
)


def read_image(file_name):
    """Reads an image and sets white to 0, grey to 0.5, and black to 1.0

    Args:
        file_name (str): map file name

    Returns:
        np array: generated map of shape (H, W)
    """
    image = cv2.imread(file_name, flags=cv2.IMREAD_GRAYSCALE)

    # preprocessing
    # 1.0 occupied, 0 free
    map = np.zeros_like(image)
    map = np.where(image != 127, map, 0.5)
    map = np.where(image != 0, map, 1.0)

    return map


# algorithm functions
def find_outline(circle_arr):
    """
    Find the outline/boundary between free space (white, value 0) and
    occupied/unknown space (black/gray, values 0.5 or 1.0)

    Returns:
        np.array: Array of shape (N, 2) containing [x, y] coordinates of boundary pixels
    """
    # curr_map = read_image("../data/input/isaac_sample_ros_nav_hospital_map.png")
    # print(circle_arr.shape)

    hull = ConvexHull(circle_arr)
    # plt.figure()
    # plt.imshow(curr_map, cmap="gray")
    sampled_points = []

    for simplex in hull.simplices:
        p1 = circle_arr[simplex[0]]
        p2 = circle_arr[simplex[1]]

        # Plot the convex hull edge
        # plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "m-")

        # Sample points along the line between p1 and p2
        num_samples = 50  # Adjust this number as needed
        for t in np.linspace(0, 1, num_samples, endpoint=False):
            sample_point = (1 - t) * p1 + t * p2
            sampled_points.append(sample_point)

    # Convert to numpy array if needed
    sampled_points = np.array(sampled_points)

    #  plot sampled points
    # plt.plot(sampled_points[:, 0], sampled_points[:, 1], "co", markersize=2)

    # plt.show()
    # print(len(hull.simplices))

    # plt.title("Boundary")
    # plt.show()
    # outline_points = circle_arr[hull.simplices].reshape(-1, 2)
    # plt.figure()

    # plt.plot(outline_points[0, :], outline_points[1, :], "b.-")  # Blue dots + lines

    # print("outline points", outline_points.shape)
    # print("type points", type(outline_points))
    # print(np.array(outline_points).shape)
    return sampled_points


def find_random_outline_location(curr_map, pose_mask, circle_arr, num_locations=1):
    """
    Find random locations on the outline/boundary between free space and occupied/unknown space.

    Args:
        num_locations (int): Number of random locations to find

    Returns:
        np.array: Array of shape (num_locations, 2) containing [x, y] coordinates
    """
    outline_points = find_outline(circle_arr)
    outline_points = np.round(outline_points).astype(int)

    # Filter out any points that are out of bounds
    h, w = curr_map.shape
    valid_mask = (
        (outline_points[:, 0] >= 0)
        & (outline_points[:, 0] < w)
        & (outline_points[:, 1] >= 0)
        & (outline_points[:, 1] < h)
    )

    valid_points = outline_points[valid_mask]

    # Check if each valid point is in free space
    is_free = np.isclose(
        curr_map[valid_points[:, 1], valid_points[:, 0]], 0.0, atol=1e-6
    )

    outline_points = valid_points[is_free]

    if outline_points.shape[0] == 0:
        print("No outline points found in the map")
        new_pose = find_random_free_location(curr_map, pose_mask, 1)
        return new_pose, None

    # Select random indices
    random_indices = np.random.choice(
        outline_points.shape[0], size=num_locations, replace=False
    )
    random_outline_locations = outline_points[random_indices]

    # Convert from [x, y] to [row, col] format (swap coordinates)
    # This is because OpenCV uses (x=col, y=row) but our internal format is [row, col]
    random_outline_locations = np.column_stack(
        (random_outline_locations[:, 1], random_outline_locations[:, 0])
    )

    return random_outline_locations, outline_points


def find_random_free_location(map, pose_mask, num_locations=1):
    """
    Find random locations within the free space (white, value 0) of the map.

    Args:
        num_locations (int): Number of random locations to find

    Returns:
        np.array: Array of shape (num_locations, 2) containing [x, y] coordinates
    """
    # Find all free spaces (where value is 0)
    map = np.where(pose_mask, map, -1)
    free_spaces = np.where(map == 0.0)

    indices = np.column_stack((free_spaces[0], free_spaces[1]))

    if indices.shape[0] == 0:
        print("No free spaces found in the map")
        return None

    # Select random indices
    random_indices = np.random.choice(
        indices.shape[0], size=num_locations, replace=False
    )
    random_locations = indices[random_indices]

    return random_locations


def mask_circle(map, cx, cy, r):
    x = np.arange(0, map.shape[1])
    y = np.arange(0, map.shape[0])

    # circle mask around the point given
    mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r**2

    # check original mask and map out unknown space
    masked_circle = map[mask]

    num_occ = np.count_nonzero(masked_circle == 0.5)

    return num_occ


def find_max_coverage_location(map, pose_mask, sensor_range=100, num_locations=1):
    """
    Find locations within the free space (white, value 0) of the map that maximizes coverage of unknown area.

    Args:
        map (_type_): _description_
        num_locations (int, optional): _description_. Defaults to 1.
    """
    # Find all free spaces (where value is 0)
    map = np.where(pose_mask, map, -1)
    free_spaces = np.where(map == 0.0)
    indices = np.column_stack((free_spaces[0], free_spaces[1]))

    if indices.shape[0] == 0:
        print("No free spaces found in the map")
        return None

    max_coverage = 0.0
    max_coverage_location = np.array([0.0, 0.0])
    for free_space in indices:
        # get area of free space around it?
        num_occ = mask_circle(map, free_space[1], free_space[0], sensor_range)
        if num_occ > max_coverage:
            max_coverage = num_occ
            max_coverage_location[1], max_coverage_location[0] = (
                free_space[1],
                free_space[0],
            )

        # subtract from total area
    return np.expand_dims(max_coverage_location, axis=0)


def debug_plot_points(points, color, point_size=5):
    """plot N points

    Args:
        points (np array of shape [N, 2]): points to plot
        color (str): color
        point_size (int, optional): size of plotted point. Defaults to 5.
    """
    # points (N, 2)
    # plot points
    plt.figure()
    plt.scatter(
        x=points[:, 0],
        y=points[:, 1],
        s=point_size,
        c=color,
        label="Robots",
    )
    plt.show()


def find_max_coverage_max_boundary_location(
    map, pose_mask, circle_points, sensor_range=100, num_locations=1
):
    """
    Find locations within the free space (white, value 0) of the map that maximizes coverage of unknown area and is on the boundary.

    Args:
        map (_type_): _description_
        num_locations (int, optional): _description_. Defaults to 1.
    """
    # Find all free spaces at the boundary(where value is 0)
    # boundary is a (N,2) array of points at the boundary
    boundary_coords = find_outline(circle_points)
    boundary_coords = np.round(boundary_coords).astype(int)

    # Filter out any points that are out of bounds
    h, w = map.shape
    valid_mask = (
        (boundary_coords[:, 0] >= 0)
        & (boundary_coords[:, 0] < w)
        & (boundary_coords[:, 1] >= 0)
        & (boundary_coords[:, 1] < h)
    )

    valid_points = boundary_coords[valid_mask]

    # Check if each valid point is in free space
    is_free = np.isclose(map[valid_points[:, 1], valid_points[:, 0]], 0.0, atol=1e-6)

    valid_boundary_points = valid_points[is_free]

    # Convert to array
    valid_boundary_points = np.array(valid_boundary_points)

    max_coverage = 0.0
    max_coverage_location = np.array([0.0, 0.0])
    for free_space in valid_boundary_points:
        # get area of free space around it?
        num_occ = mask_circle(map, free_space[1], free_space[0], sensor_range)
        if num_occ > max_coverage:
            max_coverage = num_occ
            max_coverage_location[1], max_coverage_location[0] = (
                free_space[0],
                free_space[1],
            )

    return np.expand_dims(max_coverage_location, axis=0)
