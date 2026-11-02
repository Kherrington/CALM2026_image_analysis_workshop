"""Plain cross-correlation of Mito_Cropped.tif with Mito_template.tif.

Run it from the top folder of the workshop repository with:

    uv run src/cross_correlation.py

The result is saved as testImages/Mito_crosscorr.tif and opened in Napari
next to the image and the template.  Close the Napari window to end the script.
"""

from pathlib import Path

import napari
import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Mito_Cropped.tif"
TEMPLATE = REPO / "testImages" / "Mito_template.tif"
OUTPUT = REPO / "testImages" / "Mito_crosscorr.tif"

# Where the template was cut from (see make_mito_template.py), for comparison.
CUT_ROW, CUT_COL = 225, 172

with tifffile.TiffFile(IMAGE) as tif:
    pixels = tif.asarray()
    xres = tif.pages[0].tags["XResolution"].value
    yres = tif.pages[0].tags["YResolution"].value
    unit = tif.imagej_metadata.get("unit", "micron")
template = tifffile.imread(TEMPLATE)

# Cross-correlation: put the template on top of the image with its center on a
# pixel, multiply every template pixel with the image pixel below it, and add
# everything up.  The sum is the output value of that pixel.  Repeat for every
# pixel.  (This is the same as convolution, except that the kernel is not
# flipped.)  Outside the image, pixels count as 0.
result = ndimage.correlate(pixels.astype(float), template.astype(float), mode="constant")

# Save as float32: the values are far larger than 16 bits can hold.
tifffile.imwrite(
    OUTPUT,
    result.astype(np.float32),
    imagej=True,
    resolution=(xres, yres),
    metadata={"unit": unit},
)

row, col = np.unravel_index(np.argmax(result), result.shape)
print(f"Image {pixels.shape}, template {template.shape}")
print(f"Highest value {result.max():.3e} at row {row}, column {col}")
print(f"The template was cut from row {CUT_ROW}, column {CUT_COL}")
print(f"Value at the cut position: {result[CUT_ROW, CUT_COL]:.3e}"
      f"  ({result[CUT_ROW, CUT_COL] / result.max():.0%} of the maximum)")
print(f"Saved {OUTPUT}")

viewer = napari.Viewer()
viewer.add_image(pixels, name="Mito_Cropped")
viewer.add_image(template, name="template")
viewer.add_image(result, name="cross-correlation", colormap="magma")
viewer.add_points(
    [[CUT_ROW, CUT_COL]],
    name="template origin",
    size=15,
    face_color="transparent",
    border_color="cyan",
)
viewer.grid.enabled = True
viewer.grid.shape = (1, 3)
napari.run()
