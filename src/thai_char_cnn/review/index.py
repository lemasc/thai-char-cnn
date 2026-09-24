"""Similar-image lookup and model suggestions, one row per manifest image.

Two signals, deliberately different:

- **pixel** -- RMS distance on the audit's 16x16 letterboxed grid (`cache/pixels.npy`), the same
  signal the near-duplicate groups were cut on. Literal: it finds the same strokes.
- **cnn** -- cosine similarity of the best run's 128-d global-average-pooled features. Learned: it
  finds the same *character* rendered differently (other fonts, sizes), which is what agreement on an ambiguous glyph
  needs. It was trained on the current labels, so it inherits their mistakes; predictions on
  images the model trained on are partly memorised and flagged as such in the UI.

Build once (about a minute; the decode is cached by `data.decode`):

    uv run python -m thai_char_cnn.review.index
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data import decode
from ..model import build_model
from ..paths import CACHE, RUNS, SPLIT_DIR
from ..split import stable_hash
from ..train import load_runs
from .catalog import Catalog

INDEX_VERSION = 1
TOP_K = 5


def best_run(runs_dir: Path = RUNS) -> str:
    runs = load_runs(runs_dir)
    assert len(runs), f"no finished runs under {runs_dir} -- train one with notebooks/04_train.ipynb"
    return str(runs.sort_values("val_macro_f1", ascending=False).run_id.iloc[0])


def index_path(run_id: str, paths: list[str], cache: Path = CACHE) -> Path:
    key = stable_hash(dict(run=run_id, paths=paths, v=INDEX_VERSION))
    return cache / f"review_index_{run_id}_{key}.npz"


@torch.no_grad()
def build(run_id: str | None = None, catalog: Catalog | None = None, runs_dir: Path = RUNS,
          device: str | None = None, batch: int = 4096) -> Path:
    cat = catalog or Catalog()
    run_id = run_id or best_run(runs_dir)
    paths = cat.images.path.tolist()
    out = index_path(run_id, paths, cat.cache)
    if out.exists():
        return out

    ck = torch.load(runs_dir / run_id / "model.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg, ck["n_classes"])
    model.load_state_dict(ck["state_dict"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    pixels, geo = decode(paths, cfg["img_size"], cfg["preprocess"], raw=cat.raw, cache=cat.cache)
    emb, top_i, top_p, probs_folder = [], [], [], []
    classes = pd.read_csv(SPLIT_DIR / "classes.csv").sort_values("class_idx")
    idx_folder = classes.folder.to_numpy(np.int64)                       # class_idx -> folder
    folder_idx = {int(f): i for i, f in enumerate(idx_folder)}
    folder_of_row = cat.images.folder.map(folder_idx).to_numpy(np.int64, copy=True)
    for s in range(0, len(paths), batch):
        x = torch.from_numpy(pixels[s:s + batch]).unsqueeze(1).float().div(255)
        x = ((x - ck["pixel_mean"]) / ck["pixel_std"]).to(device)
        g = ((torch.from_numpy(geo[s:s + batch]) - ck["geo_mean"]) / ck["geo_std"]).to(device)
        h = model.features(x)
        z = torch.cat([h, model.geo(g)], 1) if model.use_geometry else h
        p = model.head(z).float().softmax(1)
        emb.append(torch.nn.functional.normalize(h.float(), dim=1).cpu())
        tp, ti = p.topk(TOP_K, 1)
        top_i.append(ti.cpu())
        top_p.append(tp.cpu())
        probs_folder.append(p[torch.arange(len(p)), torch.from_numpy(folder_of_row[s:s + batch]).to(device)].cpu())

    top_i = torch.cat(top_i).numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, emb=torch.cat(emb).numpy().astype(np.float16),
             top_folder=idx_folder[top_i].astype(np.int16),
             top_p=torch.cat(top_p).numpy().astype(np.float16),
             p_folder=torch.cat(probs_folder).numpy().astype(np.float32),
             meta=json.dumps(dict(run_id=run_id, name=cfg.get("name", ""), n=len(paths))))
    return out


@dataclass
class Neighbours:
    rows: np.ndarray        # manifest rows, best first
    score: np.ndarray       # cnn: cosine; pixel: RMS gray levels; both: fused rank score
    vote: pd.Series         # folder -> count among the neighbours


class Index:
    def __init__(self, catalog: Catalog, run_id: str | None = None, runs_dir: Path = RUNS):
        self.cat = catalog
        f = build(run_id, catalog, runs_dir)
        z = np.load(f)
        self.meta = json.loads(str(z["meta"]))
        self.emb = z["emb"].astype(np.float32)
        self.top_folder = z["top_folder"].astype(np.int64)
        self.top_p = z["top_p"].astype(np.float32)
        self.p_folder = z["p_folder"]
        self.pixels = np.load(catalog.cache / "pixels.npy").astype(np.float32)
        self._sq = (self.pixels ** 2).sum(1)
        im = catalog.images
        self.folder = im.folder.to_numpy()
        self.sha = pd.factorize(im.sha256)[0]

    def _rms(self, row: int) -> np.ndarray:
        q = self.pixels[row]
        d2 = self._sq - 2 * self.pixels @ q + q @ q
        return np.sqrt(np.maximum(d2, 0) / self.pixels.shape[1])

    def neighbours(self, row: int, signal: str = "cnn", scope: str = "all", k: int = 24) -> Neighbours:
        """k nearest images, one per distinct sha256 and never the query's own copies."""
        if signal == "cnn":
            score = self.emb @ self.emb[row]
            order_key = -score
        elif signal == "pixel":
            score = self._rms(row)
            order_key = score
        elif signal == "both":
            # reciprocal-rank fusion: the two scores live on different scales, their ranks don't
            r_c = np.empty(len(self.emb), np.int64)
            r_c[np.argsort(-(self.emb @ self.emb[row]))] = np.arange(len(self.emb))
            r_p = np.empty_like(r_c)
            r_p[np.argsort(self._rms(row))] = np.arange(len(self.emb))
            score = 1 / (60 + r_c) + 1 / (60 + r_p)
            order_key = -score
        else:
            raise ValueError(f"unknown signal {signal!r}")

        mask = self.sha != self.sha[row]
        if scope == "same class":
            mask &= self.folder == self.folder[row]
        elif scope == "other classes":
            mask &= self.folder != self.folder[row]
        cand = np.flatnonzero(mask)
        m = min(len(cand), k * 4)
        top = cand[np.argpartition(order_key[cand], m - 1)[:m]] if m else cand
        top = top[np.argsort(order_key[top], kind="stable")]
        _, first = np.unique(self.sha[top], return_index=True)
        top = top[np.sort(first)][:k]
        vote = pd.Series(self.folder[top]).value_counts()
        return Neighbours(top, score[top], vote)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run id (default: best val macro-F1)")
    a = ap.parse_args()
    cat = Catalog()
    ix = Index(cat, a.run)
    im = cat.images
    split = im.path.map(cat.items.set_index("path").split).fillna("")  # representative rows only
    val = (split == "val").to_numpy()
    agree = ix.top_folder[:, 0] == ix.folder
    print(f"index for run {ix.meta['run_id']} ({ix.meta['name']}): emb {ix.emb.shape}")
    print(f"model top-1 == folder label: all {agree.mean():.4f} · val representatives {agree[val].mean():.4f}")


if __name__ == "__main__":
    main()
