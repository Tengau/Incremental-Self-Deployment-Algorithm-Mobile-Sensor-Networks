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
        plt.show()

    def plot_points(self, points, color, label="Points"):
        # points (N, 2)
        # plot points
        plt.figure(figsize=(10, 10))
        plt.imshow(self.gt_map, cmap="gray_r")
        plt.scatter(
            x=points[:, 1],  # Note: x is the column index (1)
            y=points[:, 0],  # y is the row index (0)
            s=20,
            c=color,
            label=label,
        )
        plt.colorbar()
        plt.legend()
        plt.show()

    def find_random_free_location(self, num_locations=1):
        """
        Find random locations within the free space (white, value 0) of the map.
        
        Args:
            num_locations (int): Number of random locations to find
            
        Returns:
            np.array: Array of shape (num_locations, 2) containing [x, y] coordinates
        """
        # Find all free spaces (where value is 0)
        free_spaces = np.where(self.gt_map == 0.0)
        indices = np.column_stack((free_spaces[0], free_spaces[1]))
        
        if indices.shape[0] == 0:
            print("No free spaces found in the map")
            return None
        
        # Select random indices
        random_indices = np.random.choice(indices.shape[0], size=num_locations, replace=False)
        random_locations = indices[random_indices]
        
        return random_locations

    def find_outline(self):
        """
        Find the outline/boundary between free space (white, value 0) and 
        occupied/unknown space (black/gray, values 0.5 or 1.0)
        
        Returns:
            np.array: Array of shape (N, 2) containing [x, y] coordinates of boundary pixels
        """
        # Create a binary mask where free space is 1 and occupied/unknown is 0
        binary_map = (self.gt_map == 0.0).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        # Extract all points from contours
        outline_points = []
        for contour in contours:
            for point in contour:
                outline_points.append(point[0])  # point is [[[x, y]]], we want [x, y]
        
        return np.array(outline_points)
    
    def find_random_outline_location(self, num_locations=1):
        """
        Find random locations on the outline/boundary between free space and occupied/unknown space.
        
        Args:
            num_locations (int): Number of random locations to find
            
        Returns:
            np.array: Array of shape (num_locations, 2) containing [x, y] coordinates
        """
        # Get all outline points
        outline_points = self.find_outline()
        
        if outline_points.shape[0] == 0:
            print("No outline points found in the map")
            return None
        
        # Select random indices
        random_indices = np.random.choice(outline_points.shape[0], size=num_locations, replace=False)
        random_outline_locations = outline_points[random_indices]
        
        # Convert from [x, y] to [row, col] format (swap coordinates)
        # This is because OpenCV uses (x=col, y=row) but our internal format is [row, col]
        random_outline_locations = np.column_stack((random_outline_locations[:, 1], random_outline_locations[:, 0]))
        
        return random_outline_locations
    
    def draw_map_with_features(self, output_file=None, include_outline_points=True):
        """
        Draw the map with random points and outline, and save to a file
        
        Args:
            output_file (str): Path to save the image
            include_outline_points (bool): Whether to include random points on the outline
        """
        # Create a color version of the map for drawing
        color_map = cv2.cvtColor(np.uint8(self.gt_map * 255), cv2.COLOR_GRAY2BGR)
        
        # Find random locations (5 points)
        random_locations = self.find_random_free_location(num_locations=5)
        
        # Find the outline
        outline_points = self.find_outline()
        
        # Find random locations on the outline (3 points)
        if include_outline_points:
            random_outline_locations = self.find_random_outline_location(num_locations=3)
        
        # Draw random points in free space (as blue circles)
        for point in random_locations:
            cv2.circle(color_map, (point[1], point[0]), 10, (255, 0, 0), -1)  # BGR format: Blue
        
        # Draw outline (as red pixels)
        for point in outline_points:
            color_map[point[1], point[0]] = (0, 0, 255)  # BGR format: Red
        
        # Draw random points on outline (as green circles)
        if include_outline_points:
            for point in random_outline_locations:
                cv2.circle(color_map, (point[1], point[0]), 10, (0, 255, 0), -1)  # BGR format: Green
        
        # Save or display the result
        if output_file:
            cv2.imwrite(output_file, color_map)
            print(f"Map with features saved to {output_file}")
        
        # Convert to RGB for matplotlib display
        color_map_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 12))
        plt.imshow(color_map_rgb)
        plt.title("Map with Free Locations (Blue), Outline (Red), and Outline Locations (Green)")
        plt.show()
        
        results = {
            "free_locations": random_locations,
            "outline_points": outline_points,
        }
        
        if include_outline_points:
            results["outline_locations"] = random_outline_locations
            
        return results
    
    def plot_map_with_features(self, random_points=None, outline_points=None, outline_locations=None):
        """
        Plot the map with optional random points and outline using matplotlib
        
        Args:
            random_points (np.array, optional): Random points to plot
            outline_points (np.array, optional): Outline points to plot
            outline_locations (np.array, optional): Random locations on the outline to plot
        """
        plt.figure(figsize=(12, 12))
        plt.imshow(self.gt_map, cmap="gray_r")
        
        if random_points is not None:
            plt.scatter(random_points[:, 1], random_points[:, 0], c='blue', s=50, label='Random Free Locations')
        
        if outline_points is not None:
            plt.scatter(outline_points[:, 0], outline_points[:, 1], c='red', s=1, label='Outline')
        
        if outline_locations is not None:
            plt.scatter(outline_locations[:, 1], outline_locations[:, 0], c='green', s=50, label='Random Outline Locations')
        
        plt.colorbar()
        plt.legend()
        plt.title("Map with Features")
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
        # Rest of the method stays the same
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

    # Rest of your existing methods...


parent_dir = Path(__file__).resolve().parent.parent
CONFIG_FILENAME = parent_dir / "config" / "config_test.yaml"


def main():
    config_path = Path(CONFIG_FILENAME)
    config = load_config(config_path)
    parent_dir = Path(__file__).resolve().parent.parent
    print(parent_dir)
    print(config["input"]["input_map"])
    file_name = parent_dir / config["input"]["input_map"]
    
    # Initialize occupancy grid simulator
    occ_sim = OccupancyGridSimulator(file_name, starting_pose=np.array([800.0, 700]))
    
    # Display the original map
    gt_map = occ_sim.read_image(file_name)
    print("Displaying original map...")
    occ_sim.plot_test_map(gt_map)
    
    # Draw and display the map with features
    print("Drawing map with random free locations, outline, and random outline locations...")
    results = occ_sim.draw_map_with_features(
        output_file=str(parent_dir / "output" / "map_with_features.png"),
        include_outline_points=True
    )
    
    random_locations = results["free_locations"]
    outline_points = results["outline_points"]
    random_outline_locations = results["outline_locations"]
    
    print(f"Found {len(random_locations)} random free locations:")
    print(random_locations)
    
    print(f"Found {len(outline_points)} outline points")
    
    print(f"Found {len(random_outline_locations)} random outline locations:")
    print(random_outline_locations)
    
    # Alternative matplotlib visualization
    print("Plotting map with features using matplotlib...")
    occ_sim.plot_map_with_features(
        random_points=random_locations, 
        outline_points=outline_points,
        outline_locations=random_outline_locations
    )


if __name__ == "__main__":
    main()