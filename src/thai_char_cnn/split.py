"""Writer-level train/val split.

A writer's whole hand goes to one side, so validation measures handwriting the model has not seen.
The only images allowed to break that are those of *tiny* classes (too few to validate at all),
which always go to train and are counted.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SPLIT_CONFIG = dict(unit="writer", val_fraction=0.2, min_val_class_images=20, seed=42,
                            search_iters=5000, unruled_cross_class="drop")
EMPTY_VAL_PENALTY = 10.0      # per eligible class left without a single val image


def stable_hash(obj, n: int = 12) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def load_split_config(path: Path) -> dict:
    cfg = DEFAULT_SPLIT_CONFIG | (json.loads(path.read_text()) if path.exists() else {})
    assert cfg["unit"] == "writer", f"unsupported split unit {cfg['unit']!r}"
    return cfg


def config_hash(cfg: dict) -> str:
    """Hash of the knobs that decide the writer assignment (not the ruling policy)."""
    keys = ["unit", "val_fraction", "min_val_class_images", "seed", "search_iters"]
    return stable_hash({k: cfg[k] for k in keys})


def class_index(classes_csv: Path) -> dict[int, int]:
    """Folder -> 0..C-1, fixed from the audit's folder list so rulings can never renumber classes."""
    folders = sorted(pd.read_csv(classes_csv).class_folder.astype(int))
    return {f: i for i, f in enumerate(folders)}


def dedup_counts(df: pd.DataFrame) -> pd.Series:
    """Distinct images per label among included rows (one per sha group and label)."""
    inc = df[df.included]
    return inc.drop_duplicates(["sha_group_id", "label_class"]).groupby("label_class").size()


def tiny_classes(df: pd.DataFrame, cfg: dict) -> set[int]:
    n = dedup_counts(df)
    return set(n[n < cfg["min_val_class_images"]].index.astype(int))


def _score(val: np.ndarray, tot: np.ndarray, frac: float) -> float:
    share = val / tot
    return float(np.abs(share - frac).mean() + EMPTY_VAL_PENALTY * (val == 0).sum())


def assign_writers(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Seeded random search over writer orders; each order is greedily filled to `val_fraction`.

    `df` is the ruled table (one row per image with writer_id, label_class, sha_group_id,
    included). Counts are over included rows of *eligible* (non-tiny) classes, one per
    (sha, label, writer). Returns (writers table, search summary).
    """
    tiny = tiny_classes(df, cfg)
    inc = df[df.included & ~df.label_class.isin(tiny)].drop_duplicates(
        ["sha_group_id", "label_class", "writer_id"])
    M = pd.crosstab(inc.writer_id, inc.label_class)          # writers x eligible classes
    writers, W = M.index.to_numpy(), M.to_numpy(np.int64)
    tot, n_w = W.sum(0), W.sum(1)
    frac = cfg["val_fraction"]
    target = frac * n_w.sum()

    rng = np.random.default_rng(cfg["seed"])
    best, best_score = None, np.inf
    for _ in range(cfg["search_iters"]):
        order = rng.permutation(len(writers))
        in_val = np.zeros(len(writers), bool)
        filled = 0
        for w in order:                       # add writers while they fit, skip ones that overshoot
            if filled + n_w[w] <= target * 1.05:
                in_val[w] = True
                filled += n_w[w]
            if filled >= target * 0.95:
                break
        s = _score(W[in_val].sum(0), tot, frac) + abs(filled / n_w.sum() - frac)
        if s < best_score:
            best, best_score = in_val, s

    all_writers = sorted(df.writer_id.unique())
    val_set = set(writers[best])
    out = pd.DataFrame({"writer_id": all_writers})
    out["split"] = np.where(out.writer_id.isin(val_set), "val", "train")
    per = df[df.included].groupby("writer_id").agg(n_images=("path", "size"),
                                                   n_classes=("label_class", "nunique"))
    out = out.merge(per, on="writer_id", how="left").fillna({"n_images": 0, "n_classes": 0})
    out[["n_images", "n_classes"]] = out[["n_images", "n_classes"]].astype(int)
    out["config_hash"] = config_hash(cfg)
    val = W[best].sum(0)
    summary = dict(score=round(float(best_score), 5), val_writers=int(best.sum()),
                   val_share_of_eligible=round(float(val.sum() / tot.sum()), 4),
                   per_class_share_min=round(float((val / tot).min()), 4),
                   per_class_share_max=round(float((val / tot).max()), 4),
                   eligible_classes_without_val=int((val == 0).sum()),
                   tiny_classes=sorted(tiny))
    return out, summary


def load_or_assign_writers(df: pd.DataFrame, cfg: dict, writers_csv: Path) -> tuple[pd.DataFrame, dict]:
    """Reuse the saved assignment while `split.json` is unchanged, so new rulings only move the
    images they touch -- never whole writers. Rebuilt only when the split config changes."""
    if writers_csv.exists():
        saved = pd.read_csv(writers_csv)
        if (saved.config_hash == config_hash(cfg)).all():
            new = set(df.writer_id) - set(saved.writer_id)
            assert not new, f"writers not in {writers_csv.name}: {sorted(new)}; delete it to rebuild"
            return saved, dict(reused=True)
    out, summary = assign_writers(df, cfg)
    return out, summary | dict(reused=False)


def assign_images(df: pd.DataFrame, writers: pd.DataFrame, tiny: set[int]) -> pd.DataFrame:
    """Per image split: its writer's side, except tiny classes which always train."""
    df = df.copy()
    wsplit = df.writer_id.map(writers.set_index("writer_id").split)
    df["forced_train"] = df.included & df.label_class.isin(tiny) & (wsplit == "val")
    df["split"] = np.where(df.included, np.where(df.label_class.isin(tiny), "train", wsplit), "")
    return df


def dedup_within_split(df: pd.DataFrame) -> pd.DataFrame:
    """Mark one canonical row per (sha, label, split): a copied image counts once on each side."""
    df = df.copy()
    df["is_split_canonical"] = False
    inc = df[df.included].sort_values("path")
    first = inc.drop_duplicates(["sha_group_id", "label_class", "split"]).index
    df.loc[first, "is_split_canonical"] = True
    return df
