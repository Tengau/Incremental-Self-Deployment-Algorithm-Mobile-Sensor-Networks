import numpy as np
import cv2
from pathlib import Path
from config_parser import load_config
import matplotlib.pyplot as plt
from occupancy_tz import OccupancyGrid


# utils move later
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


class OccupancyGridSimulator:
    """Represents an occupancy grid for robotic mapping."""

    def __init__(self, image_file, starting_pose):
        self.gt_map = self.read_image(image_file)
        self.robot_pos = starting_pose  # collection of all robots
        self.current_robot_pos = starting_pose
        self.curr_map = None
        self.px_per_meter = 10.0  # add to config later
        self.sensor_range = 5.0

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

    def plot_points(self, points, color):
        # points (N, 2)
        # plot points
        plt.scatter(
            x=points[:, 0] * self.px_per_meter,
            y=points[:, 1] * self.px_per_meter,
            s=5,
            c=color,
            label="Robots",
        )
        plt.show()

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
        # run bresenham_line on the 360 points from the pose or make a
        # diff algorithm that stops when it hits an obstacle
        obstacle_point = []
        for point in circle_points:
            print(
                "run alg on: ", pose[0], pose[1], np.floor(point[0]), np.floor(point[1])
            )

            points_in_between = self.bresenham_line(
                pose[0], pose[1], np.floor(point[0]), np.floor(point[1])
            )  # double check resolution
            print("run alg on: ", pose[0], pose[1], point[0], point[1])
            print(len(points_in_between))

            # filter for obstacle hits (there should only be a few)
            # found_obstacle = False
            for points in points_in_between:
                if self.gt_map[(int)(points[0]), (int)(points[1])] == 1.0:
                    # dist = distance(points, pose)
                    # obstacle_point.append((points, dist))
                    obstacle_point.append((points))
                    print(len(obstacle_point))
                    break
            # make sure to handle if there are no obstacle hits
            if not obstacle_point:
                print(f"no obstacles found for {pose[0]}, {pose[1]}")
        print(obstacle_point)
        print("length")
        print(len(obstacle_point))
        # keep the full bresenham_line returned values to save time and compute

        # calculate and sort obstacles by distance so you can keep the shortest distance
        # thats your obstacle

        # the area in between the shortest distance obstacle point is your free space and the area behind is unknown
        # run therest like normal
        # print(obstacle_point)
        pass

    def update(self, new_pose):
        # will need to run paper algorithm before this to find new pose
        self.current_robot_pos = new_pose
        self.robot_pos = np.vstack((self.robot_pos, new_pose))

        # update map
        self.get_laser_readings(new_pose)

        self.plot_points(self.robot_pos, "g")
        self.plot_points(self.current_robot_pos, "r")


parent_dir = Path(__file__).resolve().parent.parent
CONFIG_FILENAME = parent_dir / "config" / "config_test.yaml"


def main():
    config_path = Path(CONFIG_FILENAME)
    config = load_config(config_path)
    parent_dir = Path(__file__).resolve().parent.parent
    print(parent_dir)
    print(config["input"]["input_map"])
    file_name = parent_dir / config["input"]["input_map"]

    # gt_map = read_image(file_name)
    # current_map = get_laser_readings((10.0, 10.0), gt_map, 5)

    # plot_test_map(gt_map)
    # plt.show()

    # robot_pos = np.array([[800.0, 700.0], [800.0 + 5 * 10, 700]])
    # plot_points(robot_pos)

    # occ_map = OccupancyGrid(gt_map.shape, config, gt_map)
    occ_sim = OccupancyGridSimulator(file_name, starting_pose=np.array([800.0, 700]))

    occ_sim.update(new_pose=np.array([750.0 / 10, 700.0 / 10]))


if __name__ == "__main__":
    main()
