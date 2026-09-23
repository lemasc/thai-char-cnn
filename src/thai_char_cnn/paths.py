"""Repository locations, found the same way the notebooks find them: cwd, or its parent when
running from `notebooks/`."""

import os
from pathlib import Path

DATASET = os.environ.get("DATASET", "baseline")

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA = ROOT / "data" / DATASET
RAW = DATA / "raw"
MANIFEST = DATA / "manifest"
CACHE = DATA / "cache"
SPLIT_DIR = DATA / "split"
CONFIGS = ROOT / "configs"
RUNS = ROOT / "runs"
FONT = ROOT / "assets" / "fonts" / "sarabun" / "THSarabun.ttf"
