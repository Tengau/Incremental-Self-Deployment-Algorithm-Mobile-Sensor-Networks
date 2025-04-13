import logging
import numpy as np
from pathlib import Path
import os

from config_parser import load_config
from occupancy_simulator import OccupancyGridSimulator
from find_outline_v3 import (
    find_random_free_location,
    find_random_outline_location,
    find_max_coverage_location,
    find_max_coverage_max_boundary_location,
)  # , find_outline


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
# Get the directory of the current script
parent_dir = Path(__file__).resolve().parent.parent
CONFIG_FILENAME = parent_dir / "config" / "config_test.yaml"


def main():
    config_path = Path(CONFIG_FILENAME)
    config = load_config(config_path)
    parent_dir = Path(__file__).resolve().parent.parent
    logger.info(f"Parent directory: {parent_dir}")
    logger.info(f"Input map: {config['input']['input_map']}")
    file_name = parent_dir / config["input"]["input_map"]

    t_total = 10
    occ_sim = OccupancyGridSimulator(file_name, starting_pose=np.array([200.0, 500.0]))

    mode = "random_alg"
    if not os.path.exists(f"../data/output/{mode}"):
        os.makedirs(f"../data/output/{mode}")
    occ_sim.save_img(f"../data/output/{mode}/{mode}_0.png")

    for timestep in range(t_total):  # poses:
        new_map = occ_sim.get_curr_map()
        if mode == "boundary_alg":
            new_pose = find_random_outline_location(new_map, occ_sim.robot_pos_mask, 1)
        elif mode == "random_alg":
            new_pose = find_random_free_location(new_map, occ_sim.robot_pos_mask, 1)
        elif mode == "coverage_alg":
            new_pose = find_max_coverage_location(
                new_map, occ_sim.robot_pos_mask, 50, 1
            )
        elif mode == "boundary_coverage_alg":
            new_pose = find_max_coverage_max_boundary_location(
                new_map, occ_sim.robot_pos_mask, 50, 1
            )
        # print("np", new_pose.shape)
        # print(new_pose)
        occ_sim.update(new_pose=np.array([new_pose[0][1], new_pose[0][0]]))
        occ_sim.save_img(f"../data/output/{mode}/{mode}_{timestep+1}.png")

    occ_sim.save_img(f"../data/output/{mode}_final.png")


if __name__ == "__main__":
    main()
