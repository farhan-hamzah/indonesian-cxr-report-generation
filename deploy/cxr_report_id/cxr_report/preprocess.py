"""
Image input. Both models were trained on 224x224 grayscale PNGs that show the whole radiograph
(no padding or cropping). load_image brings any input to that same form; a 224x224 grayscale
PNG passes through unchanged, exactly as in the evaluation pipeline. Then, per model:
MedGemma's processor upsamples to 896x896 itself; CheXOne gets the image resized to 448x448
first (as in training), which its processor turns into 256 image tokens.

DICOM is not read here: export it to PNG/JPEG first, with the stored window applied and
MONOCHROME1 images inverted (bones white, as in a normal chest X-ray view).
"""

import io
from pathlib import Path

import numpy as np
from PIL import Image

TRAIN_SIZE = 224


def _to_8bit(im: Image.Image) -> Image.Image:
    """16-bit / float images (some PNG exports): min-max to 0-255 instead of clipping."""
    a = np.asarray(im, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    a = (a - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(a)
    return Image.fromarray(a.round().astype(np.uint8))


def load_image(image) -> Image.Image:
    """Path, bytes, file object, numpy array or PIL image -> 224x224 RGB PIL image (shared by all models)."""
    if isinstance(image, Image.Image):
        im = image
    elif isinstance(image, (str, Path)):
        im = Image.open(image)
    elif isinstance(image, (bytes, bytearray)):
        im = Image.open(io.BytesIO(image))
    elif isinstance(image, np.ndarray):
        im = Image.fromarray(image)
    elif hasattr(image, "read"):
        im = Image.open(image)
    else:
        raise TypeError(f"unsupported image input: {type(image).__name__}")

    im.load()
    if im.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        im = _to_8bit(im)
    im = im.convert("L")  # training images are grayscale; a color export is reduced the same way
    if im.size != (TRAIN_SIZE, TRAIN_SIZE):
        im = im.resize((TRAIN_SIZE, TRAIN_SIZE), Image.BICUBIC)
    return im.convert("RGB")


def model_input(im: Image.Image, size: int | None) -> Image.Image:
    """The 224x224 image -> the size a model was trained with (rrg_indo/rrg/data.py load_image)."""
    if size and im.size != (size, size):
        im = im.resize((size, size), Image.BICUBIC)
    return im
