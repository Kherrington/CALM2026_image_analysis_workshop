"""Load an image, print some facts about it, and show it in napari."""

from pathlib import Path

import napari
from skimage import io

# Build the path to the image relative to this script, so it works
# no matter which folder you run the script from.
image_path = Path(__file__).resolve().parent.parent / "testImages" / "nucleus_cropped_1.tif"

# Read the image file into a NumPy array (a grid of numbers).
image = io.imread(image_path)

# The top-left 10 x 10 block of pixel values.
print("First 10 rows and columns:")
print(image[:10, :10])

# image.shape is (rows, columns); image.dtype is the number type.
print(f"\nRows:    {image.shape[0]}")
print(f"Columns: {image.shape[1]}")
print(f"Number type: {image.dtype}")

# Open the image in napari and keep the window open until you close it.
viewer = napari.Viewer()
viewer.add_image(image, name=image_path.name)
napari.run()
