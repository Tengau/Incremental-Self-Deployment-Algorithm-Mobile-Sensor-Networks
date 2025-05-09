import logging
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


from config_parser import load_config
from occupancy_simulator import OccupancyGridSimulator
from selection import (
    find_random_free_location,
    find_random_outline_location,
    find_max_coverage_location,
    find_max_coverage_max_boundary_location,
)


# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
parent_dir = Path(__file__).resolve().parent.parent
CONFIG_FILENAME = parent_dir / "config" / "config_test.yaml"


def plot_coverage_v_time(coverages, t_total):
    """Generate a plot for coverage

    Args:
        coverages (np array of (4, t_total)): data for 4 coverage heuristics
        t_total (int): total timesteps
    """
    plt.figure()
    x = range(t_total)
    plt.plot(x, coverages[0, :] / 100.0, label="P1", marker="o")
    plt.plot(x, coverages[1, :] / 100.0, label="P2", marker="s")
    plt.plot(x, coverages[2, :] / 100.0, label="P3", marker="^")
    plt.plot(x, coverages[3, :] / 100.0, label="P4", marker="D")
    plt.xlabel("Deployed Nodes")
    plt.ylabel("Coverage (m^2)")
    plt.title("Plot of Coverage vs Deployed Node")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("coverage_v_node.png", bbox_inches="tight")


def plot_runtime_v_time(runtimes, t_total):
    """Generate a plot for coverage

    Args:
        coverages (np array of (4, t_total)): data for 4 coverage heuristics
        t_total (int): total timesteps
    """
    plt.figure()
    x = range(t_total)
    plt.plot(x, runtimes[0, :], label="P1", marker="o")
    plt.plot(x, runtimes[1, :], label="P2", marker="s")
    plt.plot(x, runtimes[2, :], label="P3", marker="^")
    plt.plot(x, runtimes[3, :], label="P4", marker="D")
    plt.xlabel("Time (sec)")
    plt.ylabel("Deployed Nodes")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Plot of Runtime vs Deployed Nodes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("runtime_v_time.png", bbox_inches="tight")


def main():
    config_path = Path(CONFIG_FILENAME)
    config = load_config(config_path)
    parent_dir = Path(__file__).resolve().parent.parent
    logger.info(f"Parent directory: {parent_dir}")
    logger.info(f"Input map: {config['input']['input_map']}")
    file_name = parent_dir / config["input"]["input_map"]
    t_total = 50
    # occ_sim = OccupancyGridSimulator(file_name, starting_pose=np.array([800.0, 700.0]))
    algorithms = ["random_alg", "boundary_alg", "coverage_alg", "boundary_coverage_alg"]
    # algorithms = ["boundary_alg"]

    sensor_range = 5
    starting_pose = np.array([800.0, 450.0])

    coverages = np.zeros((4, t_total))
    runtimes = np.zeros((4, t_total))
    for idx, mode in enumerate(algorithms):
        print(f"in mode {mode}")
        occ_sim = OccupancyGridSimulator(
            file_name, starting_pose=starting_pose, sensor_range=sensor_range
        )

        # Run the simulation
        logger.info(f"Running mode={mode}")
        logger.info(f"Saving the figure to path ../data/output/{mode}/")

        if not os.path.exists(f"../data/output/{mode}"):
            os.makedirs(f"../data/output/{mode}")
        occ_sim.save_img(f"../data/output/{mode}/{mode}_0.png")

        for timestep in tqdm(range(t_total), desc="Simulating", unit="step"):
            start = time.time()
            coverages[idx][timestep] = occ_sim.get_coverage()
            new_map = occ_sim.get_curr_map()
            if mode == "boundary_alg":
                new_pose = find_random_outline_location(
                    occ_sim.boundary_points, occ_sim.robot_pos_mask
                )
            elif mode == "random_alg":
                new_pose = find_random_free_location(new_map, occ_sim.robot_pos_mask, 1)
            elif mode == "coverage_alg":
                new_pose = find_max_coverage_location(
                    new_map, occ_sim.robot_pos_mask, sensor_range * 10, 1
                )
            elif mode == "boundary_coverage_alg":
                new_pose = find_max_coverage_max_boundary_location(
                    new_map,
                    occ_sim.robot_pos_mask,
                    occ_sim.boundary_points,
                    sensor_range * 10,
                    1,
                )
            occ_sim.update(new_pose=np.array([new_pose[0][1], new_pose[0][0]]))
            occ_sim.save_img(f"../data/output/{mode}/{mode}_{timestep+1}.png")
            end = time.time()
            runtimes[idx][timestep] = end - start

    occ_sim.save_img(f"../data/output/{mode}_final.png")

    plot_coverage_v_time(coverages, t_total)
    np.savetxt(
        f"../data/output/coverages_{starting_pose[0]}_{starting_pose[1]}_{t_total}_nodes_{sensor_range}m_sensor_range_hospital.csv",
        coverages,
        delimiter=",",
        fmt="%.3f",
    )  # You can adjust `fmt` as needed

    plot_runtime_v_time(runtimes, t_total)

    # TODO: Record coverage and time
    # update a flag for visualization

    np.savetxt(
        f"../data/output/runtimes_{starting_pose[0]}_{starting_pose[1]}_{t_total}_nodes_{sensor_range}m_sensor_range_hospital.csv",
        runtimes,
        delimiter=",",
        fmt="%.3f",
    )  # You can adjust `fmt` as needed


if __name__ == "__main__":
    main()
