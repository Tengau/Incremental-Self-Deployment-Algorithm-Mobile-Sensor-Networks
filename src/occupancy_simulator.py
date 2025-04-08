import numpy as np
import cv2
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from config_parser import load_config

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Create console handler and set level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# Create formatter
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
# Add formatter to ch
ch.setFormatter(formatter)
# Add ch to logger
logger.addHandler(ch)

map_vis = None  # debug


class OccupancyGridSimulator:
    """Represents an occupancy grid simulator for robotic mapping."""

    def __init__(self, image_file, starting_pose):
        global map_vis  # debug

        self.gt_map = self.read_image(image_file)
        map_vis = self.gt_map
        self.robot_pos = starting_pose  # collection of all robots
        self.current_robot_pos = starting_pose
        self.curr_map = 0.5 * np.ones_like(self.gt_map)
        self.px_per_meter = 10.0  # add to config later
        self.sensor_range = 5.0
        self.map_h, self.map_w = self.gt_map.shape
        self.min_x, self.min_y = 0, 0

        self.curr_circle_points = None
        # start it off -->
        self.get_laser_readings(starting_pose)

    def read_image(self, file_name):
        image = cv2.imread(file_name, flags=cv2.IMREAD_GRAYSCALE)

        # preprocessing
        # 1.0 occupied, 0 free
        map = np.zeros_like(image)
        map = np.where(image != 127, map, 0.5)
        map = np.where(image != 0, map, 1.0)
        # plt.imshow(map, cmap="gray_r")
        # plt.colorbar()

        return map

    def plot_test_map(self, map):
        # have to use gray_reverse for plotting
        plt.imshow(map, cmap="gray_r")
        plt.colorbar()

    def plot_points(self, points, color, point_size=5):
        # points (N, 2)
        # plot points
        logger.debug(f"Points shape: {points.shape}")
        logger.debug(f"Points: {points}")
        plt.scatter(
            x=points[:, 0],
            y=points[:, 1],
            s=point_size,
            c=color,
            label="Robots",
        )
        # plt.show()

    def bresenham_line(self, x0, y0, x1, y1):
        """
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

    def generate_circle_points(self, center, radius, num_points=360):
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        points = [
            (center[0] + radius * np.cos(a), center[1] + radius * np.sin(a))
            for a in angles
        ]
        return np.array(points)  # (360, 2)

    def get_laser_readings(self, pose):
        # laser_sweep to get occupied
        # get points at a radius of sensor_range from the pose in the map

        circle_points = self.generate_circle_points(
            pose, self.sensor_range * self.px_per_meter, num_points=180
        )
        self.plot_points(circle_points, "b", 0.1)
        # run bresenham_line on the 360 points from the pose or make a
        # diff algorithm that stops when it hits an obstacle
        obstacle_point = []
        for point in circle_points:
            logger.debug(
                f"Run algorithm on: {pose[0]}, {pose[1]}, {np.floor(point[0])}, {np.floor(point[1])}"
            )

            points_in_between = self.bresenham_line(
                pose[0], pose[1], np.floor(point[0]), np.floor(point[1])
            )  # double check resolution
            logger.debug(f"Length of points in between: {len(points_in_between)}")

            # filter for obstacle hits (there should only be a few)
            for points in points_in_between:
                # obst
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

            # make sure to handle if there are no obstacle hits
            if not obstacle_point:
                logger.debug(f"No obstacles found for {pose[0]}, {pose[1]}")

    def update(self, new_pose):
        # will need to run paper algorithm before this to find new pose
        self.current_robot_pos = new_pose.reshape((1, 2))
        self.robot_pos = np.vstack((self.robot_pos, new_pose))
        # update map
        self.get_laser_readings(new_pose)

        plt.imshow(self.curr_map, cmap="gray_r")
        plt.colorbar(shrink=0.7, pad=0.02)

        self.plot_points(self.robot_pos, "g")
        self.plot_points(self.current_robot_pos, "r")
        plt.title("Robot Position Map", fontsize=16, fontweight="bold")
        plt.xlabel("X Position", fontsize=12)
        plt.ylabel("Y Position", fontsize=12)
        plt.tight_layout()
        plt.show()


parent_dir = Path(__file__).resolve().parent.parent
CONFIG_FILENAME = parent_dir / "config" / "config_test.yaml"


def main():
    config_path = Path(CONFIG_FILENAME)
    config = load_config(config_path)
    parent_dir = Path(__file__).resolve().parent.parent
    logger.info(f"Parent directory: {parent_dir}")
    logger.info(f"Input map: {config['input']['input_map']}")
    file_name = parent_dir / config["input"]["input_map"]

    occ_sim = OccupancyGridSimulator(file_name, starting_pose=np.array([800.0, 700.0]))

    poses = [
        np.array([750.0, 700.0]),
        np.array([700.0, 700.0]),
        np.array([650.0, 700.0]),
        np.array([600.0, 650.0]),
    ]

    # Run the simulation
    for pose in poses:
        occ_sim.update(new_pose=pose)

    # occ_sim.update(new_pose=np.array([750.0, 700.0]))
    # occ_sim.update(new_pose=np.array([700.0, 700.0]))
    # occ_sim.update(new_pose=np.array([800.0, 650.0]))
    # occ_sim.update(new_pose=np.array([700.0, 650.0]))


if __name__ == "__main__":
    main()
