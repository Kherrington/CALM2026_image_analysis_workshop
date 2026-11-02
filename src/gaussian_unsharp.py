"""Gaussian blur and "unsharp masked" images of Actin_cropped.tif.

Run it from the top folder of the workshop repository with:

    uv run src/gaussian_unsharp.py

Results are saved in testImages/actin_unsharp/ and opened in Napari:

    Actin_gauss_5x5.tif / Actin_gauss_21x21.tif   blurred images
    Actin_mask_5x5.tif  / Actin_mask_21x21.tif    original minus blurred

Close the Napari window to end the script.
"""

from pathlib import Path

import napari
import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Actin_cropped.tif"
OUT_DIR = REPO / "testImages" / "actin_unsharp"
OUT_DIR.mkdir(exist_ok=True)


def gaussian_kernel(size):
    """A size x size kernel with a 2D Gaussian shape, normalized to sum 1.

    The width (sigma) is chosen so that the kernel edge lies two sigma from the
    center: a 5 x 5 kernel gets sigma = 1, a 21 x 21 kernel sigma = 5.
    """
    sigma = (size - 1) / 4
    x = np.arange(size) - size // 2          # -2 ... 2 for size 5
    line = np.exp(-(x**2) / (2 * sigma**2))  # the bell curve along one direction
    kernel = np.outer(line, line)            # bell curve in 2D
    return kernel / kernel.sum()             # weights add up to 1: same brightness


# --- Read the image and its pixel size ------------------------------------
with tifffile.TiffFile(IMAGE) as tif:
    pixels = tif.asarray()
    xres = tif.pages[0].tags["XResolution"].value  # pixels per micrometer
    yres = tif.pages[0].tags["YResolution"].value
    unit = tif.imagej_metadata.get("unit", "micron")
print(f"Pixel size: {1 / (xres[0] / xres[1]):.4f} {unit}")


def save(name, data):
    """Save as float32 TIFF and copy the pixel size from the input image."""
    tifffile.imwrite(
        OUT_DIR / name,
        data.astype(np.float32),
        imagej=True,
        resolution=(xres, yres),
        metadata={"unit": unit},
    )


viewer = napari.Viewer()
limits = (float(pixels.min()), float(np.percentile(pixels, 99.9)))
viewer.add_image(pixels, name="original", contrast_limits=limits)

original = pixels.astype(float)
for size in (5, 21):
    # Blur: every pixel becomes a weighted average of its neighborhood.
    blurred = ndimage.convolve(original, gaussian_kernel(size), mode="reflect")
    # Unsharp mask: what the blur removed (fine detail and noise).
    mask = original - blurred

    save(f"Actin_gauss_{size}x{size}.tif", blurred)
    save(f"Actin_mask_{size}x{size}.tif", mask)

    print(f"\n{size} x {size} Gaussian kernel")
    print(f"  blurred : min {blurred.min():9.2f}  max {blurred.max():9.2f}")
    print(f"  mask    : min {mask.min():9.2f}  max {mask.max():9.2f}  mean {mask.mean():.3f}")
    print(f"  first 3 blurred pixels: {blurred[0, :3]}")
    print(f"  first 3 mask pixels   : {mask[0, :3]}")
    print(f"  all blurred values whole numbers? {bool(np.all(blurred == np.rint(blurred)))}")

    viewer.add_image(blurred, name=f"gauss {size}x{size}", contrast_limits=limits)
    spread = float(np.percentile(np.abs(mask), 99.9))
    viewer.add_image(
        mask,
        name=f"mask {size}x{size}",
        colormap="PiYG",
        contrast_limits=(-spread, spread),
    )

print(f"\nSaved in {OUT_DIR}")
viewer.grid.enabled = True
viewer.grid.shape = (2, 3)
napari.run()
