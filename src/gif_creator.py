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
    # Use: python3 gif_creator.py ../data/output/random_alg/ ../data/random_alg.gif
    parser = argparse.ArgumentParser(description="Create an animation from PNG files.")
    parser.add_argument("input_dir", help="Directory containing PNG files.")
    parser.add_argument("output_gif", help="Path to save the output GIF.")
    parser.add_argument(
        "--interval",
        type=int,
        default=1000,
        help="Frame interval in ms (default: 500ms)",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.input_dir, "*.png")
    files = sorted(glob.glob(pattern), key=natural_key)

    if not files:
        raise FileNotFoundError(f"No PNG files found in directory: {args.input_dir}")

    print(f"Found {len(files)} PNG files.")

    image_array = [np.array(Image.open(f)) for f in files]
    # print("Loaded image_array with shape:", np.array(image_array).shape)

    # Create figure and remove axes
    fig, ax = plt.subplots()
    im = ax.imshow(image_array[0], animated=True)

    ax.axis("off")  # Hide axis
    plt.tight_layout(pad=0)  # Remove padding
    fig.patch.set_alpha(0.0)  # Optional: transparent background

    animation_fig = animation.FuncAnimation(
        fig,
        update,
        frames=len(image_array),
        interval=args.interval,
        blit=True,
        repeat_delay=10,
    )

    animation_fig.save(args.output_gif, dpi=100, writer="pillow")
    print(f"Saved animation to {args.output_gif}")
