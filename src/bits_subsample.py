"""Look at every third pixel of an image.

Run it from the top folder of the workshop repository with:

    uv run src/bits_subsample.py

Close the Napari window to end the script.
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

pixels = tifffile.imread(IMAGE)

# "pixels[start:stop:step]" picks every "step"-th entry.  Skipping 2 rows and
# 2 columns each time means a step of 3: we keep one pixel, skip two, and so on.
# "[:30:3, :30:3]" takes 10 rows and 10 columns spread over the first 30.
subsampled = pixels[:30:3, :30:3]

print(f"File       : {IMAGE.name}")
print(f"Number type: {pixels.dtype}")
print(f"\n10 x 10 pixels, skipping 2 rows and 2 columns each time:")
np.set_printoptions(linewidth=200)
print(subsampled)

# Show the original and the subsampled version side by side in Napari.
viewer = napari.Viewer()
viewer.add_image(pixels, name="original")
viewer.add_image(subsampled, name="every 3rd pixel", translate=(0, pixels.shape[1] + 5))
napari.run()
