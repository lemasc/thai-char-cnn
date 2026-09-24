"""Image -> fixed-size model input.

Two policies. `stretch`, the training default, resizes the tight ink crop straight to a square.
`letterbox` is the same rule as `pixel_grid_vec` in the audit (§4): scale to fit, keep the aspect
ratio, LANCZOS, pad with white, centred (images are tight ink crops on white paper, so white is the
background by construction).

Letterbox keeps aspect ratio, which was expected to matter for pairs like า/ๅ. In the 04_train
ablation (3 seeds, before and after the data audit) stretch scored higher macro-F1 on every seed,
and ๅ itself improved; letterbox stays as the control.
"""

import numpy as np
from PIL import Image


def letterbox(gray: Image.Image, size: int = 32) -> np.ndarray:
    """Grayscale PIL image -> (size, size) uint8, ink dark on white."""
    w, h = gray.size
    s = min(size / w, size / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(gray.resize((nw, nh), Image.Resampling.LANCZOS), ((size - nw) // 2, (size - nh) // 2))
    return np.asarray(canvas, np.uint8)


def stretch(gray: Image.Image, size: int = 32) -> np.ndarray:
    """Grayscale PIL image -> square uint8 input, without preserving aspect ratio."""
    return np.asarray(gray.resize((size, size), Image.Resampling.LANCZOS), np.uint8)


def preprocess(gray: Image.Image, size: int, mode: str) -> np.ndarray:
    """Apply one of the explicitly supported model-input resize policies."""
    if mode == "letterbox":
        return letterbox(gray, size)
    if mode == "stretch":
        return stretch(gray, size)
    raise ValueError(f"unknown preprocessing mode {mode!r}")
