"""Repository locations, found from the current directory or its ancestors."""

import os
from pathlib import Path

DATASET = os.environ.get("DATASET", "baseline")

ROOT = next((path for path in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
             if (path / "pyproject.toml").is_file() and (path / "src" / "thai_char_cnn").is_dir()), None)
if ROOT is None:
    raise RuntimeError("Run from the thai-char-cnn project directory or one of its subdirectories")
DATA = ROOT / "data" / DATASET
RAW = DATA / "raw"
MANIFEST = DATA / "manifest"
CACHE = DATA / "cache"
SPLIT_DIR = DATA / "split"
CONFIGS = ROOT / "configs"
RUNS = ROOT / "runs"
FONT = ROOT / "assets" / "fonts" / "sarabun" / "THSarabun.ttf"
