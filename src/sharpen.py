"""Apply the ImageJ "Sharpen" filter to Actin_cropped.tif and compare with the input.

Run it from the top folder of the workshop repository with:

    uv run src/sharpen.py

or, to use a different image:

    uv run src/sharpen.py testImages/some_other_image.tif

Napari opens with the original, the sharpened image and their difference.
Close the Napari window to end the script.
"""

import argparse
from pathlib import Path

import napari
import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(description="Sharpen an image like ImageJ does.")
parser.add_argument(
    "image",
    type=Path,
    nargs="?",
    default=REPO / "testImages" / "Actin_cropped.tif",
    help="path to the .tif file (default: testImages/Actin_cropped.tif)",
)
args = parser.parse_args()

pixels = tifffile.imread(args.image)

# The Sharpen kernel of ImageJ: the center pixel is boosted (12) and all
# 8 neighbors are subtracted (-1 each).
kernel = np.array(
    [
        [-1, -1, -1],
        [-1, 12, -1],
        [-1, -1, -1],
    ],
    dtype=float,
)

# Normalize by the sum of the weights (12 - 8 = 4), like ImageJ does, so the
# overall brightness stays the same.
kernel = kernel / kernel.sum()

sharpened_float = ndimage.convolve(pixels.astype(float), kernel, mode="reflect")

# The result can fall below 0 or above the largest number the pixel type can
# hold.  Clip those values to the allowed range, then round and convert back.
info = np.iinfo(pixels.dtype)
n_clipped = int(((sharpened_float < info.min) | (sharpened_float > info.max)).sum())
sharpened = np.rint(np.clip(sharpened_float, info.min, info.max)).astype(pixels.dtype)

# A signed difference (needs a number type that can be negative).
difference = sharpened.astype(float) - pixels.astype(float)


def roughness(img):
    """Pixel-to-pixel roughness: the spread of what is left when each pixel is
    compared with the average of its 3 x 3 neighborhood.  Noise raises it."""
    img = img.astype(float)
    return (img - ndimage.uniform_filter(img, size=3)).std()


print(f"Image: {args.image.name}  shape {pixels.shape}  type {pixels.dtype}\n")
print(f"{'':12s} {'mean':>8s} {'std':>8s} {'min':>7s} {'max':>7s} {'roughness':>10s}")
for name, img in (("original", pixels), ("sharpened", sharpened)):
    print(
        f"{name:12s} {img.mean():8.1f} {img.std():8.1f} {img.min():7d} {img.max():7d}"
        f" {roughness(img):10.1f}"
    )
print(f"\nRoughness increased {roughness(sharpened) / roughness(pixels):.1f}x")
print(f"Pixels clipped to the allowed range: {n_clipped}")

limits = (float(pixels.min()), float(np.percentile(pixels, 99.9)))

viewer = napari.Viewer()
viewer.add_image(pixels, name="original", contrast_limits=limits)
viewer.add_image(sharpened, name="sharpened", contrast_limits=limits)
spread = float(np.abs(difference).max())
viewer.add_image(
    difference,
    name="difference (sharpened - original)",
    colormap="PiYG",
    contrast_limits=(-spread / 2, spread / 2),
)
viewer.grid.enabled = True
viewer.grid.shape = (1, 3)
napari.run()
