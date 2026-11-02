"""Measure how far an image has moved with cross-correlation, and move it back.

Run it from the top folder of the workshop repository with:

    uv run src/measure_drift.py

To use a different shift (7 pixels down and 4 pixels to the right by default):

    uv run src/measure_drift.py --down 3 --right -9

Saved in testImages/:

    Mito_shifted.tif    the shifted copy
    Mito_corrected.tif  the shifted copy, moved back using the measured shift

Close the Napari window to end the script.
"""

import argparse
from pathlib import Path

import napari
import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Mito_Cropped.tif"

parser = argparse.ArgumentParser(description="Measure image drift by cross-correlation.")
parser.add_argument("--down", type=int, default=7, help="pixels to shift down")
parser.add_argument("--right", type=int, default=4, help="pixels to shift right")
args = parser.parse_args()

with tifffile.TiffFile(IMAGE) as tif:
    pixels = tif.asarray()
    xres = tif.pages[0].tags["XResolution"].value
    yres = tif.pages[0].tags["YResolution"].value
    unit = tif.imagej_metadata.get("unit", "micron")


def save(name, data):
    tifffile.imwrite(
        REPO / "testImages" / name,
        data,
        imagej=True,
        resolution=(xres, yres),
        metadata={"unit": unit},
    )


# Make the shifted copy.  Pixels moved in from outside the image become 0.
shifted = ndimage.shift(pixels, (args.down, args.right), order=0, mode="constant", cval=0)
save("Mito_shifted.tif", shifted)

# Cross-correlation of the whole image with the shifted copy.  Sliding one
# image over the other pixel by pixel is slow, so this uses the Fourier
# transform: correlation in the normal image = multiplication in Fourier space.
# The average is subtracted first, otherwise the overall brightness dominates.
a = pixels.astype(float) - pixels.mean()
b = shifted.astype(float) - shifted.mean()
correlation = np.fft.ifft2(np.fft.fft2(b) * np.conj(np.fft.fft2(a))).real

# The highest value sits at the shift (row, column).  Because the Fourier
# approach wraps around the edges, positions past half the image size mean
# negative shifts (up or left).
peak = np.array(np.unravel_index(np.argmax(correlation), correlation.shape))
for axis in (0, 1):
    if peak[axis] > pixels.shape[axis] // 2:
        peak[axis] -= pixels.shape[axis]
down, right = int(peak[0]), int(peak[1])

print(f"Shift applied : {args.down} down, {args.right} right")
print(f"Shift measured: {down} down, {right} right")

# Correct the drift: move the shifted image back by the measured amount.
corrected = ndimage.shift(shifted, (-down, -right), order=0, mode="constant", cval=0)
save("Mito_corrected.tif", corrected)

# How well do original and corrected agree where both have image content?
inside = np.zeros(pixels.shape, dtype=bool)
rows = slice(max(-down, 0), pixels.shape[0] - max(down, 0))
cols = slice(max(-right, 0), pixels.shape[1] - max(right, 0))
inside[rows, cols] = True
print(f"Pixels identical after correction (where image content remains): "
      f"{np.mean(pixels[inside] == corrected[inside]):.1%}")

viewer = napari.Viewer()
viewer.add_image(pixels, name="original", colormap="green", blending="additive")
viewer.add_image(shifted, name="shifted", colormap="magenta", blending="additive")
viewer.add_image(corrected, name="corrected", colormap="magenta", blending="additive", visible=False)
viewer.add_image(correlation, name="cross-correlation", colormap="magma", visible=False)
print("In Napari: 'shifted' (magenta) and 'original' (green) do not overlap;")
print("hide 'shifted' and show 'corrected' to see them line up (white).")
napari.run()
