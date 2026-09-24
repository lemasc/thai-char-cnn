"""Conservative, review-first quality measurements for dark glyphs on light paper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from scipy import ndimage as ndi
from tqdm.auto import tqdm

from .preprocess import letterbox

QUALITY_VERSION = 1
PREPROCESS_MODES = ("RAW", "BBOX_NORMALIZATION", "LIGHT_DENOISE", "MEDIAN_FILTER", "COMPONENT_CLEANING")


def _foreground(gray: np.ndarray) -> np.ndarray:
    # Otsu can classify a blank page as foreground. Clamp its threshold to protect pale ink.
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    between = (mu[-1] * omega - mu) ** 2 / np.maximum(omega * (1 - omega), 1e-12)
    threshold = min(int(np.argmax(between)), 225)
    return gray <= threshold


def measure_image(path: Path) -> dict:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    h, w = gray.shape
    fg = _foreground(gray)
    labels, count = ndi.label(fg, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel())[1:]
    tiny = areas <= max(1, int(fg.sum() * 0.001))
    border = bool(fg[0].any() or fg[-1].any() or fg[:, 0].any() or fg[:, -1].any())
    ys, xs = np.where(fg)
    box_ratio = float(((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)) / gray.size) if len(xs) else 0.0
    # Count enclosed bright pinholes relative to the full image.
    holes = ndi.binary_fill_holes(fg) & ~fg & (gray >= 245)
    hole_labels, hole_count = ndi.label(holes)
    hole_areas = np.bincount(hole_labels.ravel())[1:]
    white_noise = float(hole_areas[hole_areas <= max(1, int(fg.sum() * .001))].sum() / gray.size) if hole_count else 0.
    lap = ndi.laplace(gray.astype(np.float32))
    return dict(width=w, height=h, aspect_ratio=w / h,
                mean_intensity=float(gray.mean()), std_intensity=float(gray.std()),
                foreground_ratio=float(fg.mean()), white_ratio=float((gray >= 245).mean()),
                black_ratio=float((gray <= 10).mean()), num_components=count,
                tiny_components=int(tiny.sum()), black_speckle_ratio=float(areas[tiny].sum() / gray.size) if count else 0.0,
                white_speckle_ratio=white_noise, border_touch=border,
                blur_score=float(lap.var()), contrast_score=float(np.percentile(gray, 95) - np.percentile(gray, 5)),
                bbox_ratio=box_ratio, possible_empty=bool(fg.sum() == 0),
                possible_mostly_background=bool(fg.mean() < 0.005))


def audit_images(images: pd.DataFrame, raw: Path) -> pd.DataFrame:
    """Scan every readable row. Unreadable files stay in the report as corruption flags."""
    rows = []
    for row in tqdm(images.itertuples(index=False), total=len(images), desc="Audit image quality"):
        entry = dict(file_path=row.path, class_idx=row.class_idx, writer_id=row.writer_id,
                     split=row.split, included=row.included)
        try:
            entry.update(measure_image(raw / row.path))
            entry["read_error"] = ""
        except (OSError, ValueError) as exc:
            entry["read_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(entry)
    return pd.DataFrame(rows)


def flag_quality(audit: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Robust relative flags. Thresholds describe review queues, not deletion rules."""
    out = audit.copy()
    cols = ["aspect_ratio", "mean_intensity", "std_intensity", "foreground_ratio",
            "blur_score", "num_components", "black_speckle_ratio", "white_speckle_ratio", "bbox_ratio"]
    limits = {}
    for col in cols:
        valid = reference[col].dropna()
        limits[col] = (float(valid.quantile(.01)), float(valid.quantile(.99))) if len(valid) else (0., 0.)
    class_reference = {}
    for class_idx, group in reference.groupby("class_idx"):
        if len(group) < 20:
            continue
        center = group[["aspect_ratio", "foreground_ratio", "mean_intensity"]].median()
        spread = (group[["aspect_ratio", "foreground_ratio", "mean_intensity"]] - center).abs().median().clip(lower=.01)
        class_reference[class_idx] = (center, spread)
    flags = []
    scores = []
    for r in out.itertuples(index=False):
        if r.read_error:
            flags.append("POSSIBLE_CORRUPTION")
            scores.append(100.)
            continue
        found = []
        if r.possible_empty or r.possible_mostly_background: found.append("POSSIBLE_EMPTY")
        if r.contrast_score < 35 or r.std_intensity < max(12, limits["std_intensity"][0]): found.append("LOW_CONTRAST")
        if r.blur_score < limits["blur_score"][0]: found.append("VERY_BLURRY")
        if r.mean_intensity < limits["mean_intensity"][0]: found.append("TOO_DARK")
        if r.mean_intensity > limits["mean_intensity"][1]: found.append("TOO_BRIGHT")
        if r.white_speckle_ratio > limits["white_speckle_ratio"][1] and r.white_speckle_ratio > 0: found.append("EXCESSIVE_WHITE_NOISE")
        if r.black_speckle_ratio > limits["black_speckle_ratio"][1] and r.black_speckle_ratio > 0: found.append("EXCESSIVE_BLACK_NOISE")
        if r.num_components > limits["num_components"][1]: found.append("TOO_MANY_COMPONENTS")
        if r.bbox_ratio < limits["bbox_ratio"][0]: found.append("CHARACTER_TOO_SMALL")
        if r.border_touch: found.append("CHARACTER_TOUCHES_BORDER")
        if r.aspect_ratio < limits["aspect_ratio"][0] or r.aspect_ratio > limits["aspect_ratio"][1]: found.append("ABNORMAL_ASPECT_RATIO")
        if r.class_idx in class_reference:
            center, spread = class_reference[r.class_idx]
            deviations = [abs(getattr(r, col) - center[col]) / spread[col]
                          for col in ("aspect_ratio", "foreground_ratio", "mean_intensity")]
            if sum(value > 5 for value in deviations) >= 2:
                found.append("POSSIBLE_OUTLIER")
        if len(found) >= 3: found.append("POSSIBLE_CORRUPTION")
        flags.append(" | ".join(found))
        scores.append(float(len(found) + 2 * ("POSSIBLE_CORRUPTION" in found)))
    out["quality_flags"] = flags
    out["quality_score"] = scores
    return out


def _bbox_and_pad(gray: Image.Image, padding: float = 0.12) -> Image.Image:
    a = np.asarray(gray.convert("L"))
    ys, xs = np.where(_foreground(a))
    if len(xs) == 0:
        return gray
    left, right, top, bottom = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    # Proportional padding protects Thai marks near the main stroke.
    pad = max(2, round(max(right - left, bottom - top) * padding))
    crop = gray.crop((left, top, right, bottom))
    canvas = Image.new("L", (crop.width + 2 * pad, crop.height + 2 * pad), 255)
    canvas.paste(crop, (pad, pad))
    return canvas


def _light_denoise(gray: Image.Image) -> Image.Image:
    """Replace isolated extreme pixels only; leave strokes and gray antialiasing intact."""
    a = np.asarray(gray, dtype=np.uint8)
    median = ndi.median_filter(a, size=3)
    isolated = (np.abs(a.astype(np.int16) - median.astype(np.int16)) >= 180)
    support = ndi.convolve((a < 220).astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant")
    isolated &= (support <= 2) | ((a > 245) & (support >= 7))
    clean = a.copy()
    clean[isolated] = median[isolated]
    return Image.fromarray(clean)


def _component_clean(gray: Image.Image) -> Image.Image:
    a = np.asarray(gray, dtype=np.uint8)
    mask = _foreground(a)
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count < 2:
        return gray
    areas = np.bincount(labels.ravel())[1:]
    largest = int(areas.max())
    threshold = max(1, int(largest * 0.002))
    if threshold <= 1:
        return gray
    main = labels == (int(np.argmax(areas)) + 1)
    near = ndi.binary_dilation(main, iterations=max(2, round(max(a.shape) * .15)))
    remove = np.zeros(count + 1, dtype=bool)
    for i, area in enumerate(areas, start=1):
        component = labels == i
        if area <= threshold and not np.any(component & near):
            remove[i] = True
    result = a.copy()
    result[remove[labels]] = 255
    return Image.fromarray(result)


def preprocess_candidate(gray: Image.Image, size: int, mode: str) -> np.ndarray:
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"unknown preprocessing mode {mode!r}")
    gray = gray.convert("L")
    if mode == "RAW":
        return letterbox(gray, size)
    if mode == "LIGHT_DENOISE":
        gray = _light_denoise(gray)
    elif mode == "MEDIAN_FILTER":
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
    elif mode == "COMPONENT_CLEANING":
        gray = _component_clean(gray)
    return letterbox(_bbox_and_pad(gray), size)


def count_cleaned_images(paths: list[str], raw: Path, mode: str) -> int:
    """Count images whose source-resolution pixels changed before crop or resize."""
    cleaning = {"LIGHT_DENOISE": _light_denoise, "MEDIAN_FILTER":
                lambda image: image.filter(ImageFilter.MedianFilter(size=3)),
                "COMPONENT_CLEANING": _component_clean}
    if mode not in cleaning:
        return 0
    changed = 0
    for path in tqdm(paths, desc="Count actual pixel cleaning"):
        with Image.open(raw / path) as source:
            gray = source.convert("L")
            before = np.asarray(gray)
            after = np.asarray(cleaning[mode](gray))
        changed += bool(np.any(before != after))
    return changed
