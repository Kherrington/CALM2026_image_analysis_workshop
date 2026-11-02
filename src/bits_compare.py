"""Open any TIFF image given on the command line, print a summary, show it.

Run it from the top folder of the workshop repository with, for example:

    uv run src/bits_compare.py

(without a file name it opens nucleus_cropped_1.tif, which is also what happens
when you press the "Run" button in VS Code), or name the file yourself:

    uv run src/bits_compare.py testImages/nucleus_cropped_1.tif
    uv run src/bits_compare.py testImages/nucleus_cropped_2.tif
    uv run src/bits_compare.py testImages/nucleus_cropped_3.tif

Close the Napari window to end the script.
"""

import argparse
from pathlib import Path

import napari
import numpy as np
import tifffile

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

pixels = tifffile.imread(args.image)

rows, columns = pixels.shape
print(f"File       : {args.image.name}")
print(f"Rows       : {rows}")
print(f"Columns    : {columns}")
print(f"Number type: {pixels.dtype}")
print(f"Min / max  : {pixels.min()} / {pixels.max()}")
print(f"Bytes/pixel: {pixels.itemsize}  (file data: {pixels.nbytes} bytes)")

np.set_printoptions(linewidth=200)
print("\nFirst 10 rows and 10 columns:")
print(pixels[:10, :10])

viewer = napari.Viewer()
viewer.add_image(pixels, name=args.image.name)
napari.run()
