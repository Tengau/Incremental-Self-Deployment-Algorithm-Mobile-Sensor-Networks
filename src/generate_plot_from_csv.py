# Load and process the CSV data
import matplotlib.pyplot as plt
import csv
import numpy as np

# Load and process the CSV data
file_path = "..\data\output\coverages_30_nodes_10m_sensor_range_hospital.csv"  # Update this if your file has a different name or path
data = []

with open(file_path, "r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        processed_row = [float(item) * 0.01 for item in row]  # Scale by 0.01
        data.append(processed_row)
data = np.array(data)
x = np.arange(1, data.shape[1] + 1)

# Define marker, edge, and fill behavior for each policy
styles = [
    {"marker": "s", "facecolor": "#1f77b4", "edgecolor": "#1f77b4"},  # Filled square
    {"marker": "x", "facecolor": "none", "edgecolor": "#ff7f0e"},  # X
    {"marker": "s", "facecolor": "none", "edgecolor": "#2ca02c"},  # Empty square
    {"marker": "o", "facecolor": "none", "edgecolor": "#d62728"},  # Empty circle
]
labels = [
    "Selection policy P1",
    "Selection policy P2",
    "Selection policy P3",
    "Selection policy P4",
]

plt.figure(figsize=(10, 6))

for i in range(len(data)):
    style = styles[i]
    plt.plot(
        x,
        data[i],
        linestyle="-",
        linewidth=1,
        marker=style["marker"],
        markerfacecolor=style["facecolor"],
        markeredgecolor=style["edgecolor"],
        color=style["edgecolor"],
        markersize=7,
        label=labels[i],
    )

plt.xlabel("Deployed nodes")
plt.ylabel("Coverage (m$^2$)")
plt.title("Coverage vs Deployed Nodes")
plt.grid(True, linestyle="-", linewidth=0.5)
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
