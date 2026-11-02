"""Compare three averaging (convolution) filters with the median filters.

Run it from the top folder of the workshop repository with:

    uv run src/mean_filters.py

or, to use a different image:

    uv run src/mean_filters.py testImages/some_other_image.tif

Napari opens with the original, three averaging-filter results and three
median-filter results in a grid.  Close the Napari window to end the script.
"""

import argparse
from pathlib import Path

import napari
import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(description="Compare averaging and median filters.")
parser.add_argument(
    "image",
    type=Path,
    nargs="?",
    default=REPO / "testImages" / "Nucleus_cropped_noise.tif",
    help="path to the .tif file (default: testImages/Nucleus_cropped_noise.tif)",
)
args = parser.parse_args()

pixels = tifffile.imread(args.image)

# --- Averaging filters (convolution) ---------------------------------------
# A kernel is a small grid of weights.  At every pixel, each neighbor is
# multiplied by its weight and the products are added up.

kernel_3x3 = np.ones((3, 3))

# Same, but the center pixel counts 4 times: the pixel itself keeps more
# influence than each neighbor.
kernel_center = np.ones((3, 3))
kernel_center[1, 1] = 4

kernel_7x7 = np.ones((7, 7))

kernels = {
    "mean 3x3": kernel_3x3,
    "mean 3x3 center 4": kernel_center,
    "mean 7x7": kernel_7x7,
}

results = {}
for name, kernel in kernels.items():
    # Dividing by the sum of the weights keeps the overall brightness the same.
    normalized = kernel / kernel.sum()
    smoothed = ndimage.convolve(pixels.astype(float), normalized, mode="reflect")
    # Go back to the original number type (rounded to whole numbers).
    results[name] = np.rint(smoothed).astype(pixels.dtype)

# --- Median filters (from median_filters.py) -------------------------------
plus_3x3 = np.ones((3, 3), dtype=bool)
plus_3x3[[0, 0, 2, 2], [0, 2, 0, 2]] = False

medians = {
    "median 3x3 no corners": plus_3x3,
    "median 3x3": np.ones((3, 3), dtype=bool),
    "median 5x5": np.ones((5, 5), dtype=bool),
}
for name, footprint in medians.items():
    results[name] = ndimage.median_filter(pixels, footprint=footprint)

# --- Numbers for comparison ------------------------------------------------
all_images = {"original": pixels, **results}
names = list(all_images)

print(f"Image: {args.image.name}  shape {pixels.shape}  type {pixels.dtype}\n")
print(f"{'':24s} {'mean':>9s} {'std':>9s} {'max':>7s}")
for name, img in all_images.items():
    print(f"{name:24s} {img.mean():9.1f} {img.std():9.1f} {img.max():7d}")

# Mean absolute difference between every pair of images: 0 means identical,
# larger means the two filters changed the image in different ways.
print("\nMean absolute difference between images (pixel units):")
short = [f"{i}" for i in range(len(names))]
print(" " * 24 + "".join(f"{s:>7s}" for s in short))
for i, a in enumerate(names):
    row = [
        np.abs(all_images[a].astype(float) - all_images[b].astype(float)).mean()
        for b in names
    ]
    print(f"{i} {a:22s}" + "".join(f"{v:7.1f}" for v in row))

# --- Napari ----------------------------------------------------------------
# One shared display range, so brightness differences are real.
limits = (float(pixels.min()), float(np.percentile(pixels, 99.9)))

viewer = napari.Viewer()
for name, img in all_images.items():
    viewer.add_image(img, name=name, contrast_limits=limits)
viewer.grid.enabled = True
viewer.grid.shape = (2, 4)
napari.run()
