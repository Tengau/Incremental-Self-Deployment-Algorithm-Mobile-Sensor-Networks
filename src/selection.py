import numpy as np
import cv2
import matplotlib.pyplot as plt


# from pathlib import Path
# from config_parser import load_config
# import matplotlib.pyplot as plt
# from scipy.spatial import ConvexHull, convex_hull_plot_2d

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


def find_random_outline_location(outline_points, pose_mask):
    if outline_points.shape[0] == 0:
        raise ValueError("Input array is empty.")
    idx = np.random.randint(0, outline_points.shape[0])
    new_pose = outline_points[idx]
    # generate a new pose each time the selected pose is in the pose_mask

    while not pose_mask[int(new_pose[0])][int(new_pose[1])]:
        idx = np.random.randint(0, outline_points.shape[0])
        new_pose = outline_points[idx]
    return outline_points[idx].reshape((1, 2))


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
    max_coverage_location = np.array([0, 0])
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
    map, pose_mask, boundary_coords, sensor_range=50, num_locations=1
):
    """
    Find locations within the free space (white, value 0) of the map that maximizes coverage of unknown area and is on the boundary.

    Args:
        map (_type_): _description_
        num_locations (int, optional): _description_. Defaults to 1.
    """
    # boundary is a (N,2) array of points at the boundary
    max_coverage = -1  # Start with negative so any valid point is better
    max_coverage_location = None
    for coord in boundary_coords:  # (x, y)
        # Filter out any points that are out of bounds

        if (
            coord[1] <= 0
            or coord[1] >= map.shape[0]
            or coord[0] <= 0
            or coord[0] >= map.shape[1]
            or not pose_mask[int(coord[0])][int(coord[1])]  # False = occupied
        ):
            # print("------------------------")
            # print("y, x", int(coord[1]), int(coord[0]))
            # print(pose_mask[int(coord[1])][int(coord[0])])
            # print("skip")
            continue

        # get area of free space around it?
        num_occ = mask_circle(map, coord[0], coord[1], sensor_range)
        if num_occ > max_coverage:
            max_coverage = num_occ
            max_coverage_location = np.array([int(coord[0]), int(coord[1])])

    if max_coverage_location is None:
        # Find any valid free space
        free_spaces = np.where((map == 0.0) & pose_mask)
        if len(free_spaces[0]) > 0:
            idx = np.random.randint(0, len(free_spaces[0]))
            max_coverage_location = np.array([free_spaces[1][idx], free_spaces[0][idx]])
        else:
            # Last resort - just pick a random point in bounds
            max_coverage_location = np.array(
                [np.random.randint(0, map.shape[1]), np.random.randint(0, map.shape[0])]
            )

    # print("-", max_coverage_location)
    # print(max_coverage_location.shape)
    # print(np.expand_dims(max_coverage_location, axis=0))
    # print(np.expand_dims(max_coverage_location, axis=0).shape)
    return np.expand_dims(max_coverage_location, axis=0)
