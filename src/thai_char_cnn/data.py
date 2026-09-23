"""The split's images as in-RAM tensors.

All included, split-canonical images are decoded once into a uint8 tensor (62k x 32x32 is about
64 MB) and cached on disk keyed by the path list, so a notebook rerun doesn't touch `raw/` again.
Normalisation statistics come from **train only**.
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from .paths import CACHE, RAW, SPLIT_DIR
from .preprocess import preprocess
from .split import stable_hash

PREPROCESS_VERSION = 2       # bump when input preprocessing changes; invalidates tensor caches
N_GEO = 3


def load_split_images(split_dir=SPLIT_DIR) -> pd.DataFrame:
    """Rows a model sees: included and canonical within their split."""
    im = pd.read_csv(split_dir / "images.csv", keep_default_na=False)
    return im[im.included & im.is_split_canonical].sort_values("path").reset_index(drop=True)


def load_classes(split_dir=SPLIT_DIR) -> pd.DataFrame:
    return pd.read_csv(split_dir / "classes.csv", keep_default_na=False).sort_values("class_idx")


def decode(paths: list[str], size: int, mode: str = "letterbox", raw=RAW, cache=CACHE) -> tuple[np.ndarray, np.ndarray]:
    """-> pixels (N, size, size) uint8, geometry (N, 3) float32 = log w, log h, w/h."""
    key = stable_hash(dict(paths=paths, size=size, mode=mode, v=PREPROCESS_VERSION))
    f = cache / f"tensors_{mode}_{size}_{key}.npz"
    if f.exists():
        z = np.load(f)
        return z["pixels"], z["geo"]
    pixels = np.empty((len(paths), size, size), np.uint8)
    geo = np.empty((len(paths), N_GEO), np.float32)
    for i, p in enumerate(paths):
        with Image.open(raw / p) as im:
            gray = im.convert("L")
        w, h = gray.size
        pixels[i] = preprocess(gray, size, mode)
        geo[i] = (np.log(w), np.log(h), w / h)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(f, pixels=pixels, geo=geo)
    return pixels, geo


class SplitData:
    """Train and val tensors plus the train-only normalisation used by both."""

    def __init__(self, size: int = 32, mode: str = "letterbox", split_dir=SPLIT_DIR):
        self.images = load_split_images(split_dir)
        self.classes = load_classes(split_dir)
        pixels, geo = decode(self.images.path.tolist(), size, mode)
        self.pixels = torch.from_numpy(pixels).unsqueeze(1)             # (N, 1, s, s) uint8
        self.geo = torch.from_numpy(geo)
        self.y = torch.tensor(self.images.class_idx.to_numpy(np.int64))
        self.idx = {s: np.flatnonzero(self.images.split.to_numpy() == s) for s in ("train", "val")}
        tr = self.idx["train"]
        px = self.pixels[tr].float() / 255
        self.pixel_mean, self.pixel_std = float(px.mean()), float(px.std())
        self.geo_mean, self.geo_std = self.geo[tr].mean(0), self.geo[tr].std(0)

    def normalise(self, pixels_u8: torch.Tensor, geo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (pixels_u8.float() / 255 - self.pixel_mean) / self.pixel_std
        return x, (geo - self.geo_mean) / self.geo_std

    def dataset(self, split: str, transform=None) -> "ImageSet":
        return ImageSet(self, self.idx[split], transform)


class ImageSet(Dataset):
    def __init__(self, data: SplitData, idx: np.ndarray, transform=None):
        self.data, self.idx, self.transform = data, idx, transform

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = self.idx[i]
        px = self.data.pixels[j]
        if self.transform is not None:
            px = self.transform(px)
        x, g = self.data.normalise(px, self.data.geo[j])
        return x, g, self.data.y[j]


def make_sampler(labels: np.ndarray, kind: str, seed: int) -> WeightedRandomSampler | None:
    """`none` (natural frequencies), `sqrt` (weight 1/sqrt(n_c)) or `balanced` (1/n_c)."""
    if kind == "none":
        return None
    counts = np.bincount(labels)
    power = {"sqrt": 0.5, "balanced": 1.0}[kind]
    w = 1.0 / counts[labels].astype(np.float64) ** power
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.from_numpy(w), num_samples=len(labels), replacement=True, generator=g)
