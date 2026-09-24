"""Feasibility of the external synthesized dataset (`data/synthetic/raw/`) as extra training data.

Standalone: the pipeline (01..04) is untouched, synthetic images are decoded here and appended to
the baseline `SplitData` as extra *train* rows. Val stays the real baseline val, so every number is
measured on real glyphs. Runs are written to `runs_synth/`, never to the shared `runs/` cache (the
configs would hash to the same run ids as the real-only runs there).

    uv run python experiments/synth_feasibility.py            # all experiments
    SEEDS=42 uv run python experiments/synth_feasibility.py   # one seed, quick

Synthetic images are 64x64 renders with padding; baseline images are tight ink crops. Each synthetic
image is therefore cropped to its ink bounding box first, so pixels *and* geometry features follow
the same rule as the real data. Images whose ink touches the canvas edge (a clipped render) can be
dropped with `drop_clipped`.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from thai_char_cnn.data import SplitData
from thai_char_cnn.metrics import confusion, macro_f1, per_class
from thai_char_cnn.model import build_model
from thai_char_cnn.paths import ROOT, RUNS
from thai_char_cnn.preprocess import preprocess
from thai_char_cnn.train import fit, predict

SYN = ROOT / "data" / "synthetic" / "raw"
OUT = ROOT / "runs_synth"
INK = 200            # a pixel below this is ink when finding the crop box (renders are anti-aliased)
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,137,271").split(",")]
RARE = 100           # "rare" class: fewer real train images than this
BASE = dict(model="small_cnn", preprocess="letterbox", use_geometry=False, img_size=32)


def load_synthetic(classes: pd.DataFrame, size: int, mode: str) -> pd.DataFrame:
    """labels.csv joined to the baseline class index, with cropped pixels and geometry."""
    lab = pd.read_csv(SYN / "labels.csv", encoding="utf-8-sig", dtype={"folder": str})
    lab["path"] = lab.path.str.replace("\\", "/").str.split("/", n=1).str[1]
    idx = dict(zip(classes.folder.astype(str), classes.class_idx))
    lab["class_idx"] = lab.folder.map(idx).astype("Int64")
    pix, geo, clipped = [], [], []
    for p in lab.path:
        with Image.open(SYN / p) as im:
            g = im.convert("L")
        a = np.asarray(g)
        ys, xs = np.where(a < INK)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        clipped.append(x0 == 0 or y0 == 0 or x1 == a.shape[1] or y1 == a.shape[0])
        c = g.crop((x0, y0, x1, y1))
        w, h = c.size
        pix.append(preprocess(c, size, mode))
        geo.append((np.log(w), np.log(h), w / h))
    lab["clipped"] = clipped
    lab.attrs["pixels"] = np.stack(pix)
    lab.attrs["geo"] = np.asarray(geo, np.float32)
    return lab


def with_synthetic(data: SplitData, syn: pd.DataFrame, keep_real: bool, drop_clipped: bool,
                   rare_only: bool = False) -> SplitData:
    """Append synthetic rows (72 baseline classes only) to `data` as train; recompute train stats.
    `rare_only` keeps only classes with fewer than RARE real train images."""
    m = syn.class_idx.notna().to_numpy() & ~(syn.clipped.to_numpy() & drop_clipped)
    if rare_only:
        rare = data.classes.class_idx[data.classes.train_n < RARE]
        m &= syn.class_idx.isin(rare).to_numpy()
    n0 = len(data.pixels)
    data.pixels = torch.cat([data.pixels, torch.from_numpy(syn.attrs["pixels"][m]).unsqueeze(1)])
    data.geo = torch.cat([data.geo, torch.from_numpy(syn.attrs["geo"][m])])
    data.y = torch.cat([data.y, torch.tensor(syn.class_idx[m].astype(int).to_numpy(), dtype=torch.int64)])
    new = np.arange(n0, len(data.pixels))
    data.idx["train"] = np.concatenate([data.idx["train"], new]) if keep_real else new
    tr = data.idx["train"]
    px = data.pixels[tr].float() / 255
    data.pixel_mean, data.pixel_std = float(px.mean()), float(px.std())
    data.geo_mean, data.geo_std = data.geo[tr].mean(0), data.geo[tr].std(0)
    return data


def label_agreement(syn: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    """E0: the best existing real-only run labels every synthetic image of a shared class."""
    best = max(RUNS.glob("*/metrics.json"), key=lambda f: json.loads(f.read_text())["val_macro_f1"])
    ck = torch.load(best.parent / "model.pt", weights_only=False)
    cfg = ck["config"]
    s = syn if (cfg["img_size"], cfg["preprocess"]) == (BASE["img_size"], BASE["preprocess"]) else \
        load_synthetic(classes, cfg["img_size"], cfg["preprocess"])
    model = build_model(cfg, ck["n_classes"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    m = s.class_idx.notna().to_numpy()
    x = (torch.from_numpy(s.attrs["pixels"][m]).unsqueeze(1).float() / 255 - ck["pixel_mean"]) / ck["pixel_std"]
    g = (torch.from_numpy(s.attrs["geo"][m]) - ck["geo_mean"]) / ck["geo_std"]
    with torch.no_grad():
        pred = torch.cat([model(x[i:i + 4096], g[i:i + 4096]).argmax(1) for i in range(0, len(x), 4096)]).numpy()
    sub = s[m].assign(pred=pred)
    name = dict(zip(classes.class_idx, classes.folder.astype(str)))
    out = sub.groupby("folder").apply(lambda d: pd.Series(dict(
        n=len(d), agree=(d.pred == d.class_idx).mean(),
        top_pred=name[d.pred.mode().iloc[0]], top_pred_share=(d.pred == d.pred.mode().iloc[0]).mean())),
        include_groups=False)
    print(f"E0 model {best.parent.name} ({json.loads(best.read_text())['name']}): "
          f"synthetic agreement {(sub.pred == sub.class_idx).mean():.3f}")
    return out.sort_values("agree")


def rare_real_accuracy(run_dir: Path, data: SplitData, syn_only: bool) -> dict:
    """For a synthetic-only run, every real image of a not-validated class is unseen: score them."""
    if not syn_only:
        return {}
    ck = torch.load(run_dir / "model.pt", weights_only=False)
    model = build_model(ck["config"], ck["n_classes"]).cuda() if torch.cuda.is_available() else \
        build_model(ck["config"], ck["n_classes"])
    model.load_state_dict(ck["state_dict"])
    nv = set(data.classes.class_idx[data.classes.status != "validated"])
    real = np.flatnonzero(np.isin(data.y[:len(data.images)].numpy(), list(nv)))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pred = predict(model, data, real, dev).argmax(1).numpy()
    return dict(rare_real_n=len(real), rare_real_acc=float((pred == data.y[real].numpy()).mean()))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    real = SplitData(BASE["img_size"], BASE["preprocess"])
    syn = load_synthetic(real.classes, BASE["img_size"], BASE["preprocess"])
    print(f"synthetic: {len(syn)} images, {syn.class_idx.isna().sum()} in classes outside the 72, "
          f"{syn.clipped.sum()} touch the canvas edge")
    agree = label_agreement(syn, real.classes)
    agree.to_csv(OUT / "e0_label_agreement.csv")
    print(agree.head(15).to_string())

    arms = {  # name -> (keep_real, add_synthetic, drop_clipped, rare_only)
        "real": (True, False, False, False),
        "synthetic_only": (False, True, True, False),
        "real+synthetic": (True, True, False, False),
        "real+synthetic-clean": (True, True, True, False),
        "real+synthetic-rare": (True, True, True, True),
    }
    rows = []
    for arm, (keep_real, add, drop, rare) in arms.items():
        for seed in SEEDS:
            data = SplitData(BASE["img_size"], BASE["preprocess"])
            if add:
                data = with_synthetic(data, syn, keep_real, drop, rare)
            cfg = BASE | dict(name=arm, seed=seed)
            run_dir = fit(cfg, runs_dir=OUT / arm, data=data, verbose=False)
            met = json.loads((run_dir / "metrics.json").read_text())
            pc = pd.read_csv(run_dir / "per_class.csv")
            small = pc[(pc.status == "validated") & (pc.train_n < RARE)]
            rows.append(dict(arm=arm, seed=seed, n_train=met["n_train"], val_macro_f1=met["val_macro_f1"],
                             val_acc=met["val_acc"], small_class_f1=small.f1.mean(),
                             best_epoch=met["best_epoch"]) | rare_real_accuracy(run_dir, data, not keep_real))
            print(rows[-1])
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results.csv", index=False)
    print(res.groupby("arm", sort=False).agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    sys.exit(main())
