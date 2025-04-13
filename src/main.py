import logging
import numpy as np
from pathlib import Path

from config_parser import load_config
from occupancy_simulator import OccupancyGridSimulator
from find_outline_v3 import (
    find_random_free_location,
    find_random_outline_location,
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
    t_total = 50
    occ_sim = OccupancyGridSimulator(file_name, starting_pose=np.array([800.0, 700.0]))

    # mode = "boundary_alg"
    mode = "random"

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
    if mode == "boundary_coverage":
        for timestep in range(t_total):  # poses:
            new_map = occ_sim.get_curr_map()
            new_pose = find_random_outline_location(new_map, 1)
            # print("np", new_pose.shape)
            # print(new_pose)
            occ_sim.update(new_pose=np.array([new_pose[0][1], new_pose[0][0]]))
            occ_sim.save_img(f"../data/output/{mode}/{mode}_{timestep}.png")

        occ_sim.save_img("simulator_test.png")
    elif mode == "random":
        for timestep in range(t_total):  # poses:
            new_map = occ_sim.get_curr_map()
            new_pose = find_random_free_location(new_map, 1)
            # print("np", new_pose.shape)
            # print(new_pose)
            occ_sim.update(new_pose=np.array([new_pose[0][1], new_pose[0][0]]))
            occ_sim.save_img(f"../data/output/{mode}/{mode}_{timestep}.png")

    occ_sim.save_img(f"../data/output/{mode}_final.png")


if __name__ == "__main__":
    main()
