import argparse
import glob
import os
import re
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def natural_key(s):
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def update(i):
    im.set_array(image_array[i])
    return (im,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an animation from PNG files.")
    parser.add_argument("input_dir", help="Directory containing PNG files.")
    parser.add_argument("output_gif", help="Path to save the output GIF.")
    parser.add_argument(
        "--interval",
        type=int,
        default=500,
        help="Frame interval in ms (default: 500ms)",
    )

    args = parser.parse_args()

    # Collect and sort PNG files naturally
    pattern = os.path.join(args.input_dir, "*.png")
    files = sorted(glob.glob(pattern), key=natural_key)

    if not files:
        raise FileNotFoundError(f"No PNG files found in directory: {args.input_dir}")

    print(f"Found {len(files)} PNG files.")

    # Load images
    image_array = [np.array(Image.open(f)) for f in files]
    print("Loaded image_array with shape:", np.array(image_array).shape)

    # Create plot
    fig, ax = plt.subplots()
    im = ax.imshow(image_array[0], animated=True)

    animation_fig = animation.FuncAnimation(
        fig,
        update,
        frames=len(image_array),
        interval=args.interval,
        blit=True,
        repeat_delay=10,
    )

    # Save animation
    animation_fig.save(args.output_gif)
    print(f"Saved animation to {args.output_gif}")
