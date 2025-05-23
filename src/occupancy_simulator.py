import numpy as np
import cv2
import os
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm

from config_parser import load_config
from selection import (
    find_random_outline_location,
    find_random_free_location,
    find_max_coverage_location,
    find_max_coverage_max_boundary_location,
)
from simulation import alpha_shape, generate_circle_points, bresenham_line


# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

vis = True  # debug


class OccupancyGridSimulator:
    """Represents an occupancy grid simulator for robotic mapping."""

    def __init__(self, image_file, starting_pose, sensor_range):
        self.gt_map = self.read_image(image_file)  # (y, x)
        self.starting_pose = starting_pose
        self.robot_pos = starting_pose  # collection of all robots (x, y)
        self.current_robot_pos = starting_pose  # (x, y)
        self.robot_pos_mask = np.full(self.gt_map.shape, True, dtype=bool)
        self.curr_map = 0.5 * np.ones_like(self.gt_map)
        self.px_per_meter = 10.0  # add to config later
        self.sensor_range = sensor_range
        self.map_h, self.map_w = self.gt_map.shape
        self.min_x, self.min_y = 0, 0
        self.robot_radius = 3

        # start it off -->
        self.update_robot_mask(starting_pose[0], starting_pose[1])
        # self.update_config_space()  # ensures robots are not too close to obstacle points
        c_pts = self.get_laser_readings(starting_pose)

        plt.figure(figsize=(8, 4))

        # visualize
        if vis:
            self.plot_visualization(c_pts)

            # self.plot_points(c_pts, "Edge of 2D Lidar", "b", 0.1)
            # plt.imshow(self.curr_map, cmap="gray_r")
            # plt.imshow(self.gt_map, cmap="gray_r", alpha=0.2)
            # plt.colorbar(shrink=0.7, pad=0.02)

            # self.plot_points(
            #     self.current_robot_pos.reshape((1, 2)), "Current Robot", "r"
            # )
            # self.plot_points(self.starting_pose.reshape((1, 2)), "Anchor Node", "m")
            # plt.title("Robot Position Map", fontsize=16, fontweight="bold")
            # plt.xlim(0, self.map_w)
            # plt.ylim(self.map_h, 0)
            # plt.xlabel("X Position", fontsize=12)
            # plt.ylabel("Y Position", fontsize=12)
            # plt.tight_layout()

            # # plt.legend(loc="upper right")
            # plt.pause(interval=0.1)

    def read_image(self, file_name):
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

    def plot_points(self, points, label, color, point_size=5):
        """plot N points

        Args:
            points (np array of shape [N, 2]): points to plot (x, y)
            color (str): color
            point_size (int, optional): size of plotted point. Defaults to 5.
        """
        # points (N, 2)
        # plot points
        logger.debug(f"Points shape: {points.shape}")
        logger.debug(f"Points: {points}")
        plt.scatter(
            x=points[:, 0],
            y=points[:, 1],
            s=point_size,
            c=color,
            label=label,
        )
        # plt.show()

    def get_laser_readings(self, pose):
        """get simulated laser readings from new pose of the robot

        Args:
            pose (np array): simulate a lidar and get readings in 360 degrees (x, y)
        """
        # laser_sweep to get occupied
        # get points at a radius of sensor_range from the pose in the map
        circle_points = generate_circle_points(
            pose, self.sensor_range * self.px_per_meter, num_points=270
        )

        # run bresenham_line on the 360 points from the pose
        obstacle_point = []
        for point in circle_points:
            logger.debug(
                f"Run algorithm on (x, y): {pose[0]}, {pose[1]}, {np.floor(point[0])}, {np.floor(point[1])}"
            )

            points_in_between = bresenham_line(
                pose[0], pose[1], np.floor(point[0]), np.floor(point[1])
            )
            logger.debug(f"Length of points in between: {len(points_in_between)}")

            # filter for obstacle hits (there should only be a few)
            for points in points_in_between:
                # make sure the point youre checking is in this range
                if (
                    self.map_w > (int)(points[0]) > self.min_x
                    and self.map_h > (int)(points[1]) > self.min_y
                ):
                    if self.gt_map[(int)(points[1]), (int)(points[0])] == 1.0:
                        logger.debug(f"Obstacle found at {points}")
                        obstacle_point.append((points))
                        logger.debug(f"Number of obstacles: {len(obstacle_point)}")
                        self.curr_map[(int)(points[1]), (int)(points[0])] = 1.0
                        break
                    else:  # free
                        self.curr_map[(int)(points[1]), (int)(points[0])] = 0.0

        # find boundary points:
        white_pixels = np.where(self.curr_map == 0.0)
        # white_pixels = np.where((self.curr_map == 0.0) & (self.robot_pos_mask))
        pts = np.column_stack((white_pixels[1], white_pixels[0]))  # (x, y)

        alpha = 0.5  # Bigger alpha = tighter fit
        self.shape = alpha_shape(pts, alpha)

        if isinstance(self.shape, MultiPolygon):
            self.shape = max(self.shape.geoms, key=lambda p: p.area)

        x, y = self.shape.exterior.xy

        self.boundary_points = np.column_stack((y, x))

        return circle_points

    def update_robot_mask(self, x, y):
        y, x = int(y), int(x)
        self.robot_pos_mask[
            y - self.robot_radius * 4 : y + self.robot_radius * 4 + 1,
            x - self.robot_radius * 4 : x + self.robot_radius * 4 + 1,
        ] = False
        # false is the robot has been there before

    def update_config_space(self):
        obstacles = np.argwhere(self.gt_map == 1.0)  # (y, x)

        for y, x in obstacles:
            y, x = int(y), int(x)
            self.robot_pos_mask[
                y - self.robot_radius : y + self.robot_radius + 1,
                x - self.robot_radius : x + self.robot_radius + 1,
            ] = False

    def update(self, new_pose):
        """updates the simulation

        Args:
            new_pose (np array of shape (2,)): new pose (x, y)
        """
        if new_pose.shape != (1, 2):
            self.current_robot_pos = new_pose.reshape((1, 2))
        else:
            self.current_robot_pos = new_pose
        self.update_robot_mask(new_pose[0], new_pose[1])
        self.robot_pos = np.vstack((self.robot_pos, new_pose))

        # update map
        c_pts = self.get_laser_readings(new_pose)

        if vis:
            self.plot_visualization(c_pts)

    def plot_visualization(self, c_pts):
        plt.clf()
        plt.imshow(self.curr_map, cmap="gray_r")
        plt.imshow(self.gt_map, cmap="gray_r", alpha=0.2)
        plt.colorbar(shrink=0.7, pad=0.02)

        self.plot_points(self.robot_pos, "Previous Robot", "g")
        self.plot_points(self.current_robot_pos, "Current Robot", "r")
        self.plot_points(c_pts, "Edge of 2D Lidar", "b", 0.1)
        self.plot_points(self.starting_pose.reshape((1, 2)), "Anchor Node", "m")

        # if you want to plot occupied robot space:
        # plt.imshow(self.robot_pos_mask, cmap="gray_r", alpha=0.2)

        # if you want to plot boundaries:
        # bp = self.boundary_points.copy()
        # bp[:, [0, 1]] = bp[:, [1, 0]]
        # self.plot_points(bp, "boundary_points", "m", 0.01)

        plt.xlim(0, self.map_w)
        plt.ylim(self.map_h, 0)

        plt.title("Robot Position Map", fontsize=16, fontweight="bold")
        plt.xlabel("X Position", fontsize=12)
        plt.ylabel("Y Position", fontsize=12)
        plt.tight_layout()
        # plt.legend(loc="upper right") # turn this off for animation
        plt.pause(interval=0.1)  # comment for speedup

    def get_curr_map(self):
        return self.curr_map

    def save_img(self, output_name):
        if vis:
            plt.savefig(output_name)  # Save the figure to a file.

    def get_coverage(self):
        # plt.figure()
        map = np.zeros_like(self.curr_map)
        map = np.where(self.curr_map > 0.0, map, 1.0)
        # plt.imshow(map)
        # plt.colorbar()
        # plt.show()
        return np.sum(map)


parent_dir = Path(__file__).resolve().parent.parent
CONFIG_FILENAME = parent_dir / "config" / "config_test.yaml"


def main():
    config_path = Path(CONFIG_FILENAME)
    config = load_config(config_path)
    parent_dir = Path(__file__).resolve().parent.parent
    logger.info(f"Parent directory: {parent_dir}")
    logger.info(f"Input map: {config['input']['input_map']}")
    file_name = parent_dir / config["input"]["input_map"]
    t_total = 100
    sensor_range = 10

    # occ_sim = OccupancyGridSimulator(file_name, starting_pose=np.array([800.0, 700.0]))
    occ_sim = OccupancyGridSimulator(
        file_name, starting_pose=np.array([200.0, 500.0]), sensor_range=sensor_range
    )
    # mode = "boundary_alg"

    # replace with heuristic code:
    # poses = [
    #     np.array([750.0, 700.0]),
    #     np.array([700.0, 700.0]),
    #     np.array([650.0, 700.0]),
    #     np.array([600.0, 650.0]),
    #     np.array([550.0, 660.0]),
    #     np.array([500.0, 650.0]),
    # ]

    # for pose in poses:
    #     occ_sim.update(new_pose=pose)

    # output_name = parent_dir / config["plot"]["plot_output_filename"]

    # Run the simulation
    mode = "boundary_coverage_alg"
    logger.info(f"Running mode={mode}")
    logger.info(f"Saving the figure to path ../data/output/{mode}/")

    if not os.path.exists(f"../data/output/{mode}"):
        os.makedirs(f"../data/output/{mode}")
    occ_sim.save_img(f"../data/output/{mode}/{mode}_0.png")
    for timestep in tqdm(range(t_total), desc="Simulating", unit="step"):
        # start = time.time()
        # coverages[idx][timestep] = occ_sim.get_coverage()
        # outline_locations_debug = None
        new_map = occ_sim.get_curr_map()
        if mode == "boundary_alg":
            new_pose = find_random_outline_location(
                occ_sim.boundary_points, occ_sim.robot_pos_mask
            )
        elif mode == "random_alg":
            new_pose = find_random_free_location(new_map, occ_sim.robot_pos_mask, 1)
        elif mode == "coverage_alg":
            new_pose = find_max_coverage_location(
                new_map,
                occ_sim.robot_pos_mask,
                occ_sim.boundary_points,
                sensor_range * 10,
                1,
            )
        elif mode == "boundary_coverage_alg":
            new_pose = find_max_coverage_max_boundary_location(
                new_map,
                occ_sim.robot_pos_mask,
                occ_sim.boundary_points,
                sensor_range * 10,
                1,
            )

        occ_sim.update(new_pose=np.array([int(new_pose[0][1]), int(new_pose[0][0])]))

        occ_sim.save_img(f"../data/output/{mode}/{mode}_{timestep+1}.png")
        # end = time.time()
        # runtimes[idx][timestep] = end - start


if __name__ == "__main__":
    main()
