"""Classical unsharp mask with an adjustable amount, for two blur sizes.

Run it from the top folder of the workshop repository with:

    uv run src/unsharp_mask.py

The sharpened image is   original + amount * (original - blurred).
Napari shows the original and two results (5 x 5 and 21 x 21 Gaussian blur).
Use the "amount" slider at the right to change the amount live.  Amount 0
gives back the original image.  Close the Napari window to end the script.
"""

from pathlib import Path

import napari
import numpy as np
import tifffile
from magicgui import magicgui
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Actin_cropped.tif"
OUT_DIR = REPO / "testImages" / "actin_unsharp"
OUT_DIR.mkdir(exist_ok=True)


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
masks = {}
for size in (5, 21):
    blurred = ndimage.convolve(original, gaussian_kernel(size), mode="reflect")
    masks[size] = original - blurred

limits = (float(pixels.min()), float(np.percentile(pixels, 99.9)))
viewer = napari.Viewer()
viewer.add_image(pixels, name="original", contrast_limits=limits)
layers = {}
for size, mask in masks.items():
    layers[size] = viewer.add_image(
        original + mask, name=f"unsharp {size}x{size}", contrast_limits=limits
    )


@magicgui(
    amount={"widget_type": "FloatSlider", "min": 0.0, "max": 10.0, "step": 0.1},
    save_now={"widget_type": "PushButton", "text": "Save current images"},
    auto_call=True,
)
def controls(amount: float = 1.0, save_now=False):
    """Recalculate:  original + amount * mask."""
    for size, layer in layers.items():
        layer.data = original + amount * masks[size]


@controls.save_now.changed.connect
def _save():
    amount = controls.amount.value
    for size in layers:
        result = original + amount * masks[size]
        name = f"Actin_unsharp_{size}x{size}_amount{amount:.1f}.tif"
        tifffile.imwrite(
            OUT_DIR / name,
            result.astype(np.float32),
            imagej=True,
            resolution=(xres, yres),
            metadata={"unit": unit},
        )
        print("Saved", OUT_DIR / name)


viewer.window.add_dock_widget(controls, area="right")
viewer.grid.enabled = True
viewer.grid.shape = (1, 3)
napari.run()
