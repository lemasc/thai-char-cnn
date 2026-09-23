"""Turn human rulings under `configs/` into one label (or an exclusion) per image.

Precedence, highest first:

1. `image_rulings.csv` -- per path, `ruling in {keep, reassign, drop}`. Optional file.
2. `cross_class_rulings.csv` -- per exact-duplicate SHA group: `reassign` sets the label to
   `correct_class`, `hold` excludes the whole group.
3. Policy for what is still unresolved (`unruled_cross_class` in `configs/split.json`): an image in
   a group that still carries two labels *after* 1-2, with no image ruling of its own.

A missing file or a missing row means "keep the folder label".
"""

from pathlib import Path

import pandas as pd

OUT_COLS = ["path", "orig_class", "label_class", "included", "exclude_reason", "ruling_source"]
IMAGE_RULING_COLS = ["path", "ruling", "correct_class", "source", "note"]
IMAGE_RULINGS = {"keep", "reassign", "drop"}
GROUP_RULINGS = {"reassign", "hold"}
POLICIES = {"drop", "keep"}


def _read(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def load_image_rulings(configs: Path) -> pd.DataFrame:
    r = _read(configs / "image_rulings.csv", IMAGE_RULING_COLS)
    bad = set(r.ruling.dropna()) - IMAGE_RULINGS
    assert not bad, f"image_rulings.csv: unknown ruling(s) {bad}"
    assert r.path.is_unique, "image_rulings.csv: a path is ruled twice"
    need = r.ruling == "reassign"
    assert r.loc[need, "correct_class"].notna().all(), "image_rulings.csv: reassign without correct_class"
    return r


def load_group_rulings(configs: Path) -> pd.DataFrame:
    r = _read(configs / "cross_class_rulings.csv", ["sha_group_id", "correct_class", "ruling"])
    bad = set(r.ruling.dropna()) - GROUP_RULINGS
    assert not bad, f"cross_class_rulings.csv: unknown ruling(s) {bad}"
    assert r.sha_group_id.is_unique, "cross_class_rulings.csv: a group is ruled twice"
    return r


def apply_rulings(images: pd.DataFrame, configs: Path, unruled_cross_class: str = "drop") -> pd.DataFrame:
    """One row per manifest row, in manifest order, with the label the split should use."""
    assert unruled_cross_class in POLICIES, f"unruled_cross_class must be one of {POLICIES}"
    out = pd.DataFrame({"path": images.path, "orig_class": images.class_folder.astype(int)})
    out["label_class"] = out.orig_class
    out["included"] = images.readable.astype(bool).to_numpy()
    out["exclude_reason"] = ""
    out.loc[~out.included, "exclude_reason"] = "unreadable"
    out["ruling_source"] = ""

    # 2 · group rulings on exact cross-class duplicates
    groups = load_group_rulings(configs)
    g = images.sha_group_id.map(groups.set_index("sha_group_id").ruling)
    gc = images.sha_group_id.map(groups.set_index("sha_group_id").correct_class)
    reassign, hold = (g == "reassign").to_numpy(), (g == "hold").to_numpy()
    out.loc[reassign, "label_class"] = gc[reassign].astype(int).to_numpy()
    out.loc[hold, ["included", "exclude_reason"]] = [False, "cross_class_hold"]
    out.loc[reassign | hold, "ruling_source"] = "cross_class_rulings"

    # 1 · image rulings override everything above
    ir = load_image_rulings(configs).set_index("path")
    unknown = set(ir.index) - set(out.path)
    assert not unknown, f"image_rulings.csv names paths not in the manifest: {sorted(unknown)[:5]}"
    ruled = out.path.isin(ir.index).to_numpy()
    r = out.path.map(ir.ruling)
    rc = out.path.map(ir.correct_class)
    # an image ruling starts from the folder label, whatever its group ruling said
    out.loc[ruled, "label_class"] = out.loc[ruled, "orig_class"]
    out.loc[ruled, ["included", "exclude_reason"]] = [True, ""]
    rr = ruled & (r == "reassign").to_numpy()
    out.loc[rr, "label_class"] = rc[rr].astype(int).to_numpy()
    rd = ruled & (r == "drop").to_numpy()
    out.loc[rd, ["included", "exclude_reason"]] = [False, "image_ruling_drop"]
    out.loc[ruled, "ruling_source"] = "image_rulings"
    out.loc[ruled & ~images.readable.astype(bool).to_numpy(), ["included", "exclude_reason"]] = [False, "unreadable"]

    # 3 · policy for groups that still carry two labels among the included images
    inc = out.included.to_numpy()
    nd = images.near_dup_group_id
    n_labels = out[inc].groupby(nd[inc]).label_class.nunique()
    still_spans = nd.map(n_labels).fillna(0).to_numpy() > 1
    unruled = inc & still_spans & ~ruled
    if unruled_cross_class == "drop":
        out.loc[unruled, ["included", "exclude_reason"]] = [False, "unruled_cross_class"]
        out.loc[unruled, "ruling_source"] = "policy:unruled_cross_class"

    out["label_class"] = out.label_class.astype(int)
    out["included"] = out.included.astype(bool)
    return out[OUT_COLS]
