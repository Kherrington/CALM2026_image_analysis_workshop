"""Compare three median filters on the noisy nucleus image.

Run it from the top folder of the workshop repository with:

    uv run src/median_filters.py

or, to use a different image:

    uv run src/median_filters.py testImages/some_other_image.tif

Napari opens with four layers: the noisy original and the three filtered
versions.  Close the Napari window to end the script.
"""

import argparse
from pathlib import Path

import napari
import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(description="Compare three median filters.")
parser.add_argument(
    "image",
    type=Path,
    nargs="?",
    default=REPO / "testImages" / "Nucleus_cropped_noise.tif",
    help="path to the .tif file (default: testImages/Nucleus_cropped_noise.tif)",
)
args = parser.parse_args()

pixels = tifffile.imread(args.image)

# A median filter slides a small window ("kernel") over the image.  At every
# position it collects the pixel values under the window, sorts them, and
# writes the middle one to the output.  Only the pixels that are True in the
# footprint take part.  A single bright or dark noise pixel ends up at the end
# of the sorted list, so it is never picked: noise is removed, while edges stay
# fairly sharp.

# 3 x 3 square: 9 pixels take part.
square_3x3 = np.ones((3, 3), dtype=bool)

# 3 x 3 without corners (a plus sign): 5 pixels take part.
plus_3x3 = square_3x3.copy()
plus_3x3[[0, 0, 2, 2], [0, 2, 0, 2]] = False

# 5 x 5 square: 25 pixels take part.
square_5x5 = np.ones((5, 5), dtype=bool)

filters = {
    "median 3x3 no corners": plus_3x3,
    "median 3x3": square_3x3,
    "median 5x5": square_5x5,
}

results = {}
for name, footprint in filters.items():
    # The output has the same shape and number type as the input.
    results[name] = ndimage.median_filter(pixels, footprint=footprint)

# Print numbers to compare: the standard deviation of the pixel values is a
# rough measure of how much "roughness" (noise, but also fine detail) remains.
print(f"Image: {args.image.name}  shape {pixels.shape}  type {pixels.dtype}")
print(f"{'original':24s} std = {pixels.std():8.1f}")
for name, filtered in results.items():
    print(f"{name:24s} std = {filtered.std():8.1f}")

# All four images get the same display range, so differences in brightness
# are real differences and not an effect of automatic contrast.
limits = (float(pixels.min()), float(np.percentile(pixels, 99.9)))

viewer = napari.Viewer()
viewer.add_image(pixels, name="original", contrast_limits=limits)
for name, filtered in results.items():
    viewer.add_image(filtered, name=name, contrast_limits=limits)

# Grid mode puts the four images side by side.  Click the eye icons or turn
# grid mode off (the square button at the bottom left) to flip between them.
viewer.grid.enabled = True
viewer.grid.shape = (2, 2)
napari.run()
