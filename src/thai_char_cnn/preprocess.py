"""Image -> fixed-size model input.

Same rule as `pixel_grid_vec` in the audit (§4): scale to fit, keep the aspect ratio, LANCZOS,
pad with white, centred. Images are tight ink crops on white paper, so white is the background by
construction and a stretch would erase the aspect cues that separate า from ๅ.
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
