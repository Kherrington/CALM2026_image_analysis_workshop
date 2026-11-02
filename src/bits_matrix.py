"""Look at an image as a matrix of numbers.

Run it from the top folder of the workshop repository with:

    uv run src/bits_matrix.py

The first 10 rows and columns are printed in the terminal, and the image opens
in Napari.  Close the Napari window to end the script.
"""
import argparse
from pathlib import Path

import napari
import numpy as np
import tifffile

REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "nucleus_cropped_1.tif"

# argparse lets you type the file name after the script name.
parser = argparse.ArgumentParser(description="Summarize and display a TIFF image.")
REPO = Path(__file__).resolve().parent.parent
parser.add_argument(
    "image",
    type=Path,
    nargs="?",  # "?" makes the file name optional
    default=REPO / "testImages" / "nucleus_cropped_1.tif",  # used if you give none
    help="path to the .tif file (default: testImages/nucleus_cropped_1.tif)",
)
args = parser.parse_args()

# An image is just a grid of numbers.  tifffile reads the file into a NumPy array.
pixels = tifffile.imread(IMAGE)

rows, columns = pixels.shape
print(f"File       : {IMAGE.name}")
print(f"Rows       : {rows}")
print(f"Columns    : {columns}")
print(f"Number type: {pixels.dtype}")

# "pixels[:10, :10]" means: rows 0 to 9, columns 0 to 9.
# (Python starts counting at 0, and the end of a range is not included.)
print("\nFirst 10 rows and 10 columns:")
np.set_printoptions(linewidth=200)
print(pixels[:10, :10])

viewer = napari.Viewer()
viewer.add_image(pixels, name=IMAGE.name)
napari.run()
