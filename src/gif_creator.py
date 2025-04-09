import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import re


def natural_key(s):
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def update(i):
    im.set_array(image_array[i])
    return (im,)


# Read and sort all the .png files in the directory by filename
# files = sorted(
#     glob.glob("../data/output/boundary_alg/*.png"), key=lambda x: os.path.basename(x)
# )
files = sorted(glob.glob("../data/output/boundary_alg/*.png"), key=natural_key)


print(files)
print(f"Found {len(files)} PNG files.")

# Load images into array
image_array = [np.array(Image.open(f)) for f in files]

print("Loaded image_array with shape:", np.array(image_array).shape)

# Create figure and axes
fig, ax = plt.subplots()
im = ax.imshow(image_array[0], animated=True)

# Create animation
animation_fig = animation.FuncAnimation(
    fig,
    update,
    frames=len(image_array),
    interval=500,
    blit=True,
    repeat_delay=10,
)

# Show or save the animation
# plt.show()
animation_fig.save("../data/boundary_alg.gif")
