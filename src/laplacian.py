"""The Laplacian edge filter: compare with Sharpen and the unsharp mask.

Run it from the top folder of the workshop repository with:

    uv run src/laplacian.py

The Laplacian is saved as testImages/actin_edges/Actin_laplacian.tif and shown
in Napari next to the original and the two unsharp masks.  Close the Napari
window to end the script.
"""

from pathlib import Path

import napari
import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Actin_cropped.tif"
OUT_DIR = REPO / "testImages" / "actin_edges"
OUT_DIR.mkdir(exist_ok=True)

LAPLACIAN = np.array(
    [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ],
    dtype=float,
)

SHARPEN = np.array(
    [
        [-1, -1, -1],
        [-1, 12, -1],
        [-1, -1, -1],
    ],
    dtype=float,
)

# What must be added to the Laplacian to get Sharpen?  Subtract and look:
print("Sharpen - Laplacian =")
print(SHARPEN - LAPLACIAN)
print("Only the center changes: 4 times the original image is added.\n")


def gaussian_kernel(size):
    """size x size Gaussian kernel (sigma = (size - 1) / 4), sum = 1."""
    sigma = (size - 1) / 4
    x = np.arange(size) - size // 2
    line = np.exp(-(x**2) / (2 * sigma**2))
    kernel = np.outer(line, line)
    return kernel / kernel.sum()


with tifffile.TiffFile(IMAGE) as tif:
    pixels = tif.asarray()
    xres = tif.pages[0].tags["XResolution"].value
    yres = tif.pages[0].tags["YResolution"].value
    unit = tif.imagej_metadata.get("unit", "micron")

original = pixels.astype(float)
laplacian = ndimage.correlate(original, LAPLACIAN, mode="reflect")

tifffile.imwrite(
    OUT_DIR / "Actin_laplacian.tif",
    laplacian.astype(np.float32),
    imagej=True,
    resolution=(xres, yres),
    metadata={"unit": unit},
)

# The unsharp masks (original - Gaussian blur) of the previous section.
masks = {}
for size in (5, 21):
    blurred = ndimage.convolve(original, gaussian_kernel(size), mode="reflect")
    masks[size] = original - blurred

# Original - 3 x 3 average is *exactly* the Laplacian divided by 9.
mean3 = ndimage.uniform_filter(original, size=3, mode="reflect")
print(
    "Laplacian == 9 * (original - 3x3 average)?",
    bool(np.allclose(laplacian, 9 * (original - mean3))),
)

# Correlation coefficient: 1 = same pattern, 0 = unrelated.
print("\nCorrelation of the Laplacian with the unsharp masks:")
for size, mask in masks.items():
    r = np.corrcoef(laplacian.ravel(), mask.ravel())[0, 1]
    print(f"  {size:2d} x {size:<2d} Gaussian mask: {r:.3f}")

# --- Is an edge filter reversible? ----------------------------------------
# Add the same number to every pixel and filter again.
brighter = original + 1000
laplacian_bright = ndimage.correlate(brighter, LAPLACIAN, mode="reflect")
print("\nAdd 1000 to every pixel, then apply the Laplacian:")
print(f"  largest change in the output: {np.abs(laplacian_bright - laplacian).max():.6f}")
print(f"  sum of the kernel weights: {LAPLACIAN.sum():.0f}")

# --- Napari ----------------------------------------------------------------
viewer = napari.Viewer()
limits = (float(pixels.min()), float(np.percentile(pixels, 99.9)))
viewer.add_image(pixels, name="original", contrast_limits=limits)

spread = float(np.percentile(np.abs(laplacian), 99.5))
viewer.add_image(
    laplacian, name="Laplacian", colormap="PiYG", contrast_limits=(-spread, spread)
)
for size, mask in masks.items():
    s = float(np.percentile(np.abs(mask), 99.5))
    viewer.add_image(
        mask, name=f"unsharp mask {size}x{size}", colormap="PiYG", contrast_limits=(-s, s)
    )

viewer.grid.enabled = True
viewer.grid.shape = (2, 2)
napari.run()
