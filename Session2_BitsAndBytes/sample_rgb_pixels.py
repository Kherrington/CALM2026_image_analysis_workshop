"""Print R, G, B values of ~3% of the pixels (regularly spaced) in a table, then show the image."""

from pathlib import Path

import napari
from skimage import io

image_path = Path(__file__).resolve().parent.parent / "testImages" / "Cell_cropped_2.tif"

# A colour image is a 3D array: (rows, columns, 3), where the last
# dimension holds the Red, Green and Blue values of each pixel.
image = io.imread(image_path)
n_rows, n_cols, _ = image.shape

# Flatten the grid into one long list of pixels (row after row),
# each with its R, G, B values: shape (n_rows * n_cols, 3).
pixels = image.reshape(-1, 3)

# 3% of the pixels = every ~33rd pixel (100 / 3 = 33.3).
step = round(100 / 3)
sampled_indices = range(0, len(pixels), step)

print(f"Image: {image_path.name}  ({n_rows} rows x {n_cols} columns)")
print(f"Step between sampled pixels: {step}. Showing{len(sampled_indices)} of {len(pixels)} "
      f"({100 * len(sampled_indices) / len(pixels):.1f}%)\n")

# Print the table. divmod turns the flat index back into (row, column).
print(f"{'row':>5} {'col':>5} | {'R':>4} {'G':>4} {'B':>4}")
print("-" * 30)
for i in sampled_indices:
    row, col = divmod(i, n_cols)
    r, g, b = pixels[i]
    print(f"{row:>5} {col:>5} | {r:>4} {g:>4} {b:>4}")

# Show the full image in napari (it recognises the 3 colour channels as RGB).
viewer = napari.Viewer()
viewer.add_image(image, name=image_path.name, rgb=True)
napari.run()
