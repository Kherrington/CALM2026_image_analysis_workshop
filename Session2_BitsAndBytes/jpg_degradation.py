"""Save an image as JPG, compare it to the original, then re-save the JPG 100 times and compare again."""

import tempfile
from pathlib import Path

import napari
import numpy as np
from skimage import io

image_path = Path(__file__).resolve().parent.parent / "testImages" / "Cell_cropped_2.tif"
original = io.imread(image_path)  # uint8 colour image, shape (rows, columns, 3)

# Keep the JPG in a temporary folder so we don't clutter the project.
jpg_path = Path(tempfile.mkdtemp()) / "cell_copy.jpg"


def difference(a, b):
    """Absolute difference per pixel and channel.

    Pixels are uint8 (0-255), so subtracting directly would wrap around
    (e.g. 3 - 5 = 254). We convert to int16 first, then take the absolute value.
    """
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)


def report(label, diff):
    print(f"{label}: mean difference = {diff.mean():.2f}, "
          f"max difference = {diff.max()} (on a 0-255 scale)")


# --- Step 1: save as JPG once, read it back, compare ---
io.imsave(jpg_path, original, check_contrast=False)
jpg_once = io.imread(jpg_path)
diff_once = difference(original, jpg_once)
report("After 1 save   ", diff_once)

# --- Step 2: re-read and re-save the same JPG 100 times ---
for _ in range(100):
    current = io.imread(jpg_path)
    io.imsave(jpg_path, current, check_contrast=False)

jpg_many = io.imread(jpg_path)
diff_many = difference(original, jpg_many)
report("After 101 saves", diff_many)

# --- Show everything in napari ---
viewer = napari.Viewer()
viewer.add_image(original, name="original", rgb=True)
viewer.add_image(jpg_once, name="jpg (1 save)", rgb=True)
viewer.add_image(diff_once, name="difference (1 save)", rgb=True)
viewer.add_image(jpg_many, name="jpg (101 saves)", rgb=True)
viewer.add_image(diff_many, name="difference (101 saves)", rgb=True)
napari.run()
