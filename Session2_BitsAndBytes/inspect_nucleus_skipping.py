"""Load an image, print every 3rd pixel (skipping 2 each time), and show it in napari.

Usage:  uv run python Session2_BitsAndBytes/inspect_nucleus_skipping.py nucleus_cropped_2.tif

What changed in the code: argparse reads whatever you type after the script name and uses it 
as the file name inside testImages. If you type nothing, it uses nucleus_cropped_1.tif. 
I also added the rows and columns printout.
"""

import argparse
from pathlib import Path

import napari
from skimage import io

# Read the file name typed after the script name on the command line.
parser = argparse.ArgumentParser(description="Inspect an image from the testImages folder.")
parser.add_argument(
    "filename",
    nargs="?",
    default="nucleus_cropped_1.tif",
    help="name of a file in testImages (default: nucleus_cropped_1.tif)",
)
args = parser.parse_args()

# Build the path to the image relative to this script, so it works
# no matter which folder you run the script from.
image_path = Path(__file__).resolve().parent.parent / "testImages" / args.filename

# Read the image file into a NumPy array (a grid of numbers).
image = io.imread(image_path)

# Slicing is [start:stop:step]. A step of 3 means: take one value, skip the
# next 2, take one, and so on. We take 10 rows and 10 columns this way,
# so we reach rows/columns 0, 3, 6, ... 27.
step = 3
n = 10
subsampled = image[: n * step : step, : n * step : step]

print(f"File: {image_path.name}")
print(f"{n} rows and {n} columns, skipping 2 each time:")
print(subsampled)

# image.shape is (rows, columns); image.dtype is the number type.
print(f"\nRows:    {image.shape[0]}")
print(f"Columns: {image.shape[1]}")
print(f"Number type: {image.dtype}")

# Open the full image in napari and keep the window open until you close it.
viewer = napari.Viewer()
viewer.add_image(image, name=image_path.name)
napari.run()
