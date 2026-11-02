"""Normalized cross-correlation: find all places that look like the template.

Run it from the top folder of the workshop repository with:

    uv run src/template_matching.py

To try another threshold (the default is 0.8):

    uv run src/template_matching.py --threshold 0.6

The result is saved as testImages/Mito_ncc.tif.  Napari shows the image with a
circle on every match, plus the correlation map.  Close the Napari window to
end the script.
"""

import argparse
from pathlib import Path

import napari
import numpy as np
import tifffile
from skimage.feature import match_template, peak_local_max

REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Mito_Cropped.tif"
TEMPLATE = REPO / "testImages" / "Mito_template.tif"
OUTPUT = REPO / "testImages" / "Mito_ncc.tif"

parser = argparse.ArgumentParser(description="Template matching with normalized cross-correlation.")
parser.add_argument("--threshold", type=float, default=0.8, help="lowest value that counts as a match")
parser.add_argument("--distance", type=int, default=4, help="minimum distance between matches in pixels")
args = parser.parse_args()

with tifffile.TiffFile(IMAGE) as tif:
    pixels = tif.asarray()
    xres = tif.pages[0].tags["XResolution"].value
    yres = tif.pages[0].tags["YResolution"].value
    unit = tif.imagej_metadata.get("unit", "micron")
template = tifffile.imread(TEMPLATE)

# Normalized cross-correlation.  For every position, the piece of the image
# under the template is compared with the template after subtracting the
# average and dividing by the standard deviation of both.  The result runs from
# -1 (exact opposite) over 0 (no similarity) to 1 (perfect match), and does not
# depend on how bright the region is.  "pad_input=True" gives an output as big
# as the input image.
ncc = match_template(pixels, template, pad_input=True)

tifffile.imwrite(
    OUTPUT,
    ncc.astype(np.float32),
    imagej=True,
    resolution=(xres, yres),
    metadata={"unit": unit},
)

# A peak is a pixel that is the highest within "distance" pixels around it.
# Only peaks above the threshold are kept.
peaks = peak_local_max(ncc, min_distance=args.distance, threshold_abs=args.threshold)
values = ncc[peaks[:, 0], peaks[:, 1]]
order = np.argsort(-values)  # best match first
peaks, values = peaks[order], values[order]

print(f"NCC range: {ncc.min():.2f} to {ncc.max():.2f}")
print(f"\n{len(peaks)} peaks with value >= {args.threshold} and at least {args.distance} pixels apart")
for (r, c), v in zip(peaks[:15], values[:15]):
    print(f"  row {r:3d}, column {c:3d}   value {v:.3f}")
if len(peaks) > 15:
    print(f"  ... and {len(peaks) - 15} more")

print("\nNumber of peaks for other thresholds:")
for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    n = len(peak_local_max(ncc, min_distance=args.distance, threshold_abs=t))
    print(f"  threshold {t:4.2f}: {n}")

viewer = napari.Viewer()
viewer.add_image(pixels, name="Mito_Cropped")
viewer.add_points(
    peaks,
    name=f"matches >= {args.threshold}",
    size=template.shape[0] + 4,  # a bit larger than the template
    face_color="transparent",
    border_color="cyan",
    border_width=0.1,
)
viewer.add_image(template, name="template")
viewer.add_image(ncc, name="normalized cross-correlation", colormap="magma", contrast_limits=(0, 1))
viewer.grid.enabled = True
viewer.grid.shape = (1, 3)
napari.run()
