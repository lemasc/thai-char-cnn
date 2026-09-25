"""Test-set submission: a trained run -> one `class_id` per test image.

A candidate is either **frozen** or a **run**. A frozen candidate is a copy of a run in
`models/<experiment>/`, checked into git. A run candidate is any finished run in `runs/<run_id>/`.
Freezing decouples the submission from the run cache: `runs/` is gitignored and lives on one
machine, and a run id stops resolving from its config once `TRAIN_VERSION` or the split changes. A
frozen copy holds three files:

- `model.pt`: a byte-for-byte copy of the run's weights, config and normalisation statistics.
- `classes.csv`: the class order the model was trained with, so a later split can't remap its outputs.
- `meta.json`: the run's provenance and val metrics, plus the sha256 of `model.pt`.

`04_train` freezes with `FREEZE=1`; `05_submit` predicts.

Test images go through the same preprocessing as training (grey, `stretch` or `letterbox` to the
run's `img_size`, geometry from the original size), then the run's train-only normalisation. One step
is added: an image on a dark background (white ink on black, which training never shows) is inverted
first. The rule needs at least 90% of the border near-black, so tight dark-on-white crops, whose ink
often touches the border, are left alone: on val it inverts no image and changes no prediction.
"""

import hashlib
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from .data import load_classes
from .model import build_model
from .paths import ROOT, RUNS, SPLIT_DIR
from .preprocess import preprocess
from .train import DEFAULT_CONFIG, current_split_id, load_runs

MODELS = ROOT / "models"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
DARK, DARK_BORDER = 64, 0.9            # grey level counted as dark; share of the border that must be


def dark_background(gray: np.ndarray) -> bool:
    """True for white ink on a black background: nearly all of the border is dark."""
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    return min(gray.shape) >= 8 and (border < DARK).mean() >= DARK_BORDER


def load_test_image(path: Path, size: int, mode: str) -> tuple[np.ndarray, tuple[float, float, float], bool]:
    """`data.load_image`, plus inverting a dark-background image. -> pixels, geometry, inverted."""
    with Image.open(path) as im:
        gray = im.convert("L")
    inverted = dark_background(np.asarray(gray))
    if inverted:
        gray = ImageOps.invert(gray)
    w, h = gray.size
    return preprocess(gray, size, mode), (np.log(w), np.log(h), w / h), inverted


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def freeze(run_dir: Path, experiment: str, models_dir: Path = MODELS, split_dir: Path = SPLIT_DIR) -> Path:
    """Copy a finished run to `models/<experiment>/`, replacing any earlier copy."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["split_id"] == current_split_id(split_dir), \
        f"run {run_dir.name} is on split {cfg['split_id']}, not the current one: its class order is unknown"
    dest = models_dir / experiment
    tmp = models_dir / f".{experiment}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copyfile(run_dir / "model.pt", tmp / "model.pt")
    load_classes(split_dir)[["class_idx", "folder", "character"]].to_csv(tmp / "classes.csv", index=False)
    meta = dict(experiment=experiment, run_id=cfg["run_id"], seed=cfg["seed"], split_id=cfg["split_id"],
                train_version=cfg["train_version"], code_hash=cfg["code_hash"], model=cfg["model"],
                img_size=cfg["img_size"], preprocess=cfg["preprocess"], epochs=cfg["epochs"],
                val_macro_f1=metrics["val_macro_f1"], val_macro_f1_all=metrics["val_macro_f1_all"],
                val_acc=metrics["val_acc"], best_epoch=metrics["best_epoch"], n_params=metrics["n_params"],
                sha256=sha256(tmp / "model.pt"), git_commit=_git_commit(),
                frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    return dest


def list_candidates(n_recent: int = 10, models_dir: Path = MODELS, runs_dir: Path = RUNS,
                    split_dir: Path = SPLIT_DIR) -> pd.DataFrame:
    """Frozen models first, then the `n_recent` most recently finished runs on this machine.

    `key` is what `load_candidate` takes. `usable` is False for a run on another split, whose class
    order this checkout doesn't know.
    """
    frozen = [json.loads(m.read_text()) | dict(source="frozen", key=m.parent.name, name=m.parent.name,
                                               updated=m.stat().st_mtime, usable=True)
              for m in sorted(models_dir.glob("[!.]*/meta.json"))]
    rows = []
    if runs_dir.exists():
        runs = load_runs(runs_dir)
        if len(runs):
            split_id = current_split_id(split_dir)
            runs["updated"] = [(runs_dir / r / "metrics.json").stat().st_mtime for r in runs.run_id]
            runs = runs.sort_values("updated", ascending=False).head(n_recent)
            rows = [dict(source="run", key=r.run_id, name=r.name, run_id=r.run_id, seed=r.seed,
                         model=r["cfg.model"], img_size=r["cfg.img_size"], epochs=r["cfg.epochs"], val_macro_f1=r.val_macro_f1,
                         val_acc=r.val_acc, train_version=r.train_version, split_id=r.split_id,
                         updated=r.updated, usable=r.split_id == split_id)
                    for _, r in runs.iterrows()]
    cols = ["source", "key", "name", "run_id", "seed", "model", "img_size", "epochs", "val_macro_f1", "val_acc",
            "train_version", "split_id", "updated", "usable"]
    out = pd.DataFrame(frozen + rows).reindex(columns=cols)
    out["updated"] = pd.to_datetime(out.updated, unit="s").dt.strftime("%Y-%m-%d %H:%M")
    return out


@dataclass
class Predictor:
    """A loaded model plus everything it needs to turn image files into class indices."""

    model: torch.nn.Module
    cfg: dict
    classes: pd.DataFrame           # class_idx, folder, character in the model's output order
    pixel_mean: float
    pixel_std: float
    geo_mean: torch.Tensor
    geo_std: torch.Tensor
    device: str
    source: str
    meta: dict

    @torch.no_grad()
    def predict(self, paths: list[Path], batch: int = 512) -> torch.Tensor:
        """Softmax probabilities, (N, n_classes), computed the same way as `train.predict`.

        `self.inverted` is then a boolean array: which images were inverted first.
        """
        size, mode = self.cfg["img_size"], self.cfg["preprocess"]
        loaded = [load_test_image(p, size, mode) for p in paths]
        pixels = torch.from_numpy(np.stack([px for px, _, _ in loaded])).unsqueeze(1)
        geo = torch.tensor([g for _, g, _ in loaded], dtype=torch.float32)
        self.inverted = np.array([inv for _, _, inv in loaded])
        self.model.eval()
        out = []
        for s in range(0, len(paths), batch):
            x = (pixels[s:s + batch].float() / 255 - self.pixel_mean) / self.pixel_std
            g = (geo[s:s + batch] - self.geo_mean) / self.geo_std
            with torch.autocast(self.device, enabled=self.device == "cuda"):
                out.append(self.model(x.to(self.device), g.to(self.device)).float().softmax(1).cpu())
        return torch.cat(out)


def load_candidate(key: str, models_dir: Path = MODELS, runs_dir: Path = RUNS,
                   split_dir: Path = SPLIT_DIR, device: str | None = None) -> Predictor:
    """A frozen model by experiment name (`models/<key>/`) or a run by id (`runs/<key>/`)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if (models_dir / key / "meta.json").exists():
        d = models_dir / key
        meta = json.loads((d / "meta.json").read_text())
        assert sha256(d / "model.pt") == meta["sha256"], f"{d / 'model.pt'} does not match its meta.json sha256"
        classes = pd.read_csv(d / "classes.csv", dtype={"folder": str}, keep_default_na=False)
        source = f"frozen models/{key}"
    elif (runs_dir / key / "metrics.json").exists():
        d = runs_dir / key
        meta = json.loads((d / "config.json").read_text()) | json.loads((d / "metrics.json").read_text())
        assert meta["split_id"] == current_split_id(split_dir), \
            f"run {key} is on split {meta['split_id']}, not the current one: its class order is unknown"
        classes = load_classes(split_dir)[["class_idx", "folder", "character"]].astype({"folder": str})
        source = f"run runs/{key}"
    else:
        raise FileNotFoundError(f"no frozen model models/{key}/ and no finished run runs/{key}/")
    ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
    # runs older than a config key lack it; its default is the behaviour they were trained with
    cfg = DEFAULT_CONFIG | ck["config"]
    # rebuild the architecture only: the checkpoint overwrites every weight, so skip the ImageNet download
    model = build_model(cfg | dict(pretrained=None, freeze_through=None), ck["n_classes"])
    model.load_state_dict(ck["state_dict"])
    assert len(classes) == ck["n_classes"] and (classes.class_idx.to_numpy() == np.arange(len(classes))).all()
    return Predictor(model.to(device).eval(), cfg, classes.reset_index(drop=True),
                     float(ck["pixel_mean"]), float(ck["pixel_std"]), ck["geo_mean"], ck["geo_std"],
                     device, source, meta)


def load_class_ids(data_dict: Path) -> dict[str, str]:
    """The assignment's data dictionary: training folder -> 3-digit `class_id`."""
    dd = pd.read_csv(data_dict, dtype=str, keep_default_na=False)
    dd = dd[dd.folder != ""]
    assert dd.folder.is_unique and dd.class_id.is_unique, "data dictionary repeats a folder or class_id"
    assert dd.class_id.str.fullmatch(r"[123]\d\d").all(), "class_id must be three digits starting 1, 2 or 3"
    return dict(zip(dd.folder, dd.class_id, strict=True))


def resolve_image(test_root: Path, gt_path: str) -> Path | None:
    """`./test_dataset/ts_img_001` -> the file under `test_root`, with or without an image suffix.

    The trailing number may be zero-padded differently on disk (the sheet says `ts_img_001`, the
    released files are `ts_img_00001.jpg`), so it is matched by value when the exact name is missing.
    """
    p = test_root / gt_path
    if p.is_file():
        return p
    hit = next((p.with_name(p.name + s) for s in IMAGE_SUFFIXES if p.with_name(p.name + s).is_file()), None)
    if hit is not None or not p.parent.is_dir():
        return hit
    m = re.fullmatch(r"(.*?)(\d+)", p.stem if p.suffix in IMAGE_SUFFIXES else p.name)
    if m is None:
        return None
    prefix, num = m[1], int(m[2])
    return next((f for f in sorted(p.parent.iterdir()) if f.suffix in IMAGE_SUFFIXES
                 and (n := re.fullmatch(re.escape(prefix) + r"(\d+)", f.stem)) and int(n[1]) == num), None)
