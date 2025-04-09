# Incremental Self Deployment Algorithm for Mobile Sensor Networs

This repository contains an unofficial python paper implementation for "An Incremental Self-Deployment Algorithm for Mobile Sensor Networks" (Howard et. al)

![Output](data/output/boundary_alg.gif)

The application reads the input occupancy map from the `config.yaml` file, processes the data, and plots a simulated lidar acting with various heuristics.

## Repository Structure

```bash
.
├── LICENSE                      # Contains the license agreement 
├── README.md                    # The top-level description and general information
├── config                       # Configuration files directory
│   └── config.yaml              # Main configuration file
├── data                         # Data files directory
│   ├── input                    # Input data files
│   │   ├── map.png              # Data input file
│   └── output                   # Output data/files generated
│       ├── occupancy_plot.pdf   # Generated occupancy plot
│       └── output.gif           # Generated Gif
├── docs                         # Documentation files
├── setup.py                     # Build script for setuptools. Helps in packaging and distribution.
└── src                          # Source code directory
    ├── __init__.py              # Makes the directory a package
    ├── main.py                  # Main application source file
    ├── occupancy_simulator.py   # Source file for occupancy grid functionalities
    └── selection.py       # Handles plot operations

```

## Setup and Running
1. First, clone the repository and navigate into it:
```bash
git clone https://github.com/Tengau/Incremental-Self-Deployment-Algorithm-Mobile-Sensor-Networks.git
cd Incremental-Self-Deployment-Algorithm-Mobile-Sensor-Networks
```
2. Then, create a virtual environment (optional but recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install the necessary dependencies:
```bash
pip install .
```
4. To run the main script:
```bash
cd src
python3 occupancy_simulator.py
```

This will process the input ground truth occupancy grid specified in the config.yaml file and generate the occupancy grid map plot in data/output.

## Configuring the Script
You can modify the config.yaml file to change the behavior of the script. 

## Dataset source

The ground truth occupancy grid for this test is from the Isaac Sim Occupancy Grid Simulator. This simulated environment allows us to demonstrate the functionality and potential applications of this project effectively.


## License
This project is licensed under the terms of the MIT license. See the [LICENSE](LICENCE) file for the full text of the license.

## Citation
Howard, A., Matarić, M.J. & Sukhatme, G.S. An Incremental Self-Deployment Algorithm for Mobile Sensor Networks. Autonomous Robots 13, 113–126 (2002). https://doi.org/10.1023/A:1019625207705
