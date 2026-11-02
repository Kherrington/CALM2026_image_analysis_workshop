"""Find edges in Actin_cropped.tif with the Sobel filters.

Run it from the top folder of the workshop repository with:

    uv run src/sobel_edges.py

Results are saved in testImages/actin_edges/ and opened in Napari:

    Actin_sobel_x.tif          Sobel x  (changes from left to right)
    Actin_sobel_y.tif          Sobel y  (changes from top to bottom)
    Actin_gradient.tif         gradient magnitude, sqrt(x*x + y*y)
    Actin_gauss5_gradient.tif  gradient magnitude after a 5 x 5 Gaussian blur

Close the Napari window to end the script.
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

# Sobel x: right column minus left column (the middle row counts double).
# It responds where the brightness changes from left to right.
SOBEL_X = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ],
    dtype=float,
)

# Sobel y: bottom row minus top row.  Same kernel turned by 90 degrees.
SOBEL_Y = SOBEL_X.T


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


def save(name, data):
    """Save as float32 TIFF (edge images contain decimals and negative values)
    and copy the pixel size from the input image."""
    tifffile.imwrite(
        OUT_DIR / name,
        data.astype(np.float32),
        imagej=True,
        resolution=(xres, yres),
        metadata={"unit": unit},
    )


def sobel(img):
    """Return Sobel x, Sobel y and the gradient magnitude of an image.

    "correlate" multiplies the neighborhood with the kernel exactly as printed
    above (no flipping), like the Convolve command of ImageJ.
    """
    gx = ndimage.correlate(img, SOBEL_X, mode="reflect")
    gy = ndimage.correlate(img, SOBEL_Y, mode="reflect")
    return gx, gy, np.sqrt(gx**2 + gy**2)


original = pixels.astype(float)
gx, gy, magnitude = sobel(original)

# Same thing, but blur first.
blurred = ndimage.convolve(original, gaussian_kernel(5), mode="reflect")
_, _, magnitude_blur = sobel(blurred)

save("Actin_sobel_x.tif", gx)
save("Actin_sobel_y.tif", gy)
save("Actin_gradient.tif", magnitude)
save("Actin_gauss5_gradient.tif", magnitude_blur)

# --- Numbers ---------------------------------------------------------------
print(f"Mean |Sobel x| = {np.abs(gx).mean():.1f}    Mean |Sobel y| = {np.abs(gy).mean():.1f}")
print(f"Sobel x range: {gx.min():.0f} to {gx.max():.0f}  (negative values exist)")


def describe(name, m):
    """Strong edges = the brightest 1 % of the edge image.  Background = the
    median.  A large ratio means edges stand out clearly from the noise."""
    strong = np.percentile(m, 99)
    background = np.median(m)
    print(
        f"{name:26s} median {background:7.1f}   "
        f"99th percentile {strong:7.1f}   ratio {strong / background:5.1f}"
    )


print("\nGradient magnitude:")
describe("without blur", magnitude)
describe("after 5x5 Gaussian blur", magnitude_blur)

# --- Napari ----------------------------------------------------------------
viewer = napari.Viewer()
limits = (float(pixels.min()), float(np.percentile(pixels, 99.9)))
viewer.add_image(pixels, name="original", contrast_limits=limits)

for name, g in (("Sobel x", gx), ("Sobel y", gy)):
    spread = float(np.percentile(np.abs(g), 99.5))
    viewer.add_image(g, name=name, colormap="PiYG", contrast_limits=(-spread, spread))

top = float(np.percentile(magnitude, 99.5))
viewer.add_image(magnitude, name="gradient magnitude", contrast_limits=(0, top))
viewer.add_image(
    magnitude_blur, name="gradient magnitude (blurred first)", contrast_limits=(0, top)
)

print(f"\nSaved in {OUT_DIR}")
viewer.grid.enabled = True
viewer.grid.shape = (2, 3)
napari.run()
