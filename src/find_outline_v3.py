import numpy as np
import cv2
import matplotlib.pyplot as plt


# from pathlib import Path
# from config_parser import load_config
# import matplotlib.pyplot as plt


# algorithm functions
def find_outline(map):
    """
    Find the outline/boundary between free space (white, value 0) and
    occupied/unknown space (black/gray, values 0.5 or 1.0)

    Returns:
        np.array: Array of shape (N, 2) containing [x, y] coordinates of boundary pixels
    """
    # Create a binary mask where free space is 1 and occupied/unknown is 0
    binary_map = (map == 0.0).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(binary_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Extract all points from contours
    outline_points = []
    for contour in contours:
        for point in contour:
            outline_points.append(point[0])  # point is [[[x, y]]], we want [x, y]
    # print("check")
    return np.array(outline_points)


def find_random_outline_location(curr_map, num_locations=1):
    """
    Find random locations on the outline/boundary between free space and occupied/unknown space.

    Args:
        num_locations (int): Number of random locations to find

    Returns:
        np.array: Array of shape (num_locations, 2) containing [x, y] coordinates
    """
    # Get all outline points
    outline_points = find_outline(curr_map)

    if outline_points.shape[0] == 0:
        print("No outline points found in the map")
        return None

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

    return random_outline_locations


def find_random_free_location(map, num_locations=1):
    """
    Find random locations within the free space (white, value 0) of the map.

    Args:
        num_locations (int): Number of random locations to find

    Returns:
        np.array: Array of shape (num_locations, 2) containing [x, y] coordinates
    """
    # Find all free spaces (where value is 0)
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

    # unknown = masked_circle[masked_circle == 0.5]
    # num_occ = np.sum(2 * unknown)

    num_occ = np.count_nonzero(masked_circle == 0.5)

    return num_occ


def find_max_coverage_location(map, sensor_range=50, num_locations=1):
    """
    Find locations within the free space (white, value 0) of the map that maximizes coverage of unknown area.

    Args:
        map (_type_): _description_
        num_locations (int, optional): _description_. Defaults to 1.
    """
    # Find all free spaces (where value is 0)
    free_spaces = np.where(map == 0.0)
    indices = np.column_stack((free_spaces[0], free_spaces[1]))

    if indices.shape[0] == 0:
        print("No free spaces found in the map")
        return None

    # coverage area = area of circle w/ sensor range
    # total_area = np.pi * (sensor_range**2)
    # print("total_area:", total_area)
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


def find_max_coverage_max_boundary_location(map, sensor_range=50, num_locations=1):
    """
    Find locations within the free space (white, value 0) of the map that maximizes coverage of unknown area and is on the boundary.

    Args:
        map (_type_): _description_
        num_locations (int, optional): _description_. Defaults to 1.
    """
    # Find all free spaces at the boundary(where value is 0)
    # boundary is a (N,2) array of points at the boundary

    boundary_coords = find_outline(map)
    print(boundary_coords.shape)
    boundary_coords = np.column_stack((boundary_coords[:, 1], boundary_coords[:, 0]))
    debug_plot_points(boundary_coords, "m", point_size=5)

    valid_boundary_points = []
    for y, x in boundary_coords:
        if 0 <= y < map.shape[0] and 0 <= x < map.shape[1] and map[y, x] == 0.0:
            valid_boundary_points.append((y, x))

    if not valid_boundary_points:
        print("No valid boundary free-space points found.")
        return None

    # Convert to array
    valid_boundary_points = np.array(valid_boundary_points)
    # indices = np.column_stack((valid_boundary_points[0], valid_boundary_points[1]))

    # if indices.shape[0] == 0:
    #     print("No free spaces found in the map")
    #     return None

    # coverage area = area of circle w/ sensor range
    # total_area = np.pi * (sensor_range**2)
    # print("total_area:", total_area)
    max_coverage = 0.0
    max_coverage_location = np.array([0.0, 0.0])
    for free_space in valid_boundary_points:
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


'''
def find_max_coverage_location(map, sensor_range=50, num_locations=1):
    """
    Find location(s) within free space (value 0) that maximize coverage of unknown area (value 0.5).
    """
    # Step 1: Create circular kernel
    diameter = 2 * sensor_range + 1
    y, x = np.ogrid[-sensor_range : sensor_range + 1, -sensor_range : sensor_range + 1]
    circle_mask = (x**2 + y**2) <= sensor_range**2
    kernel = circle_mask.astype(float)

    # Step 2: Extract unknowns from map
    unknown_map = (map == 0.5).astype(float)

    # Step 3: Pad unknown_map so convolution stays inside bounds
    padded = np.pad(unknown_map, sensor_range, mode="constant", constant_values=0)

    # Step 4: Perform convolution manually using sliding window
    h, w = map.shape
    coverage_map = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            window = padded[i : i + diameter, j : j + diameter]
            if window.shape == kernel.shape:  # Just in case edges are off
                coverage_map[i, j] = np.sum(window * kernel)

    # Step 5: Mask out non-free spaces
    coverage_map[map != 0.0] = 0

    # Step 6: Get top-N best locations
    flat_indices = np.argpartition(coverage_map.ravel(), -num_locations)[
        -num_locations:
    ]
    top_coords = np.column_stack(np.unravel_index(flat_indices, coverage_map.shape))

    # Sort by coverage
    sorted_indices = np.argsort(-coverage_map[top_coords[:, 0], top_coords[:, 1]])
    top_coords = top_coords[sorted_indices]

    return top_coords[:num_locations]
'''
