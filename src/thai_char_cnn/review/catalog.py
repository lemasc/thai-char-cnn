"""What gets reviewed, and how it is drawn.

The unit of review is one image under its **folder** label, with byte-identical copies in the same
folder collapsed into one item: judging the pixels once judges every copy. A cross-class exact
duplicate therefore appears once in each class it sits in -- each copy's label is right or wrong on
its own. The representative is the first path of the copies, sorted; export expands a decision on
it back to all of them.
"""

import hashlib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from ..paths import CACHE, CONFIGS, MANIFEST, RAW, SPLIT_DIR

# glyph size that fills a thumbnail in "true size" mode -- p99.9 of max(w, h) on baseline; bigger
# images are shrunk to fit, so only the rare giant loses its relative scale
TRUE_SIZE_REF = 50

STATE_COLORS = {
    "": (225, 225, 225),
    "ok": (46, 160, 67),
    "flag": (218, 54, 51),
    "wrong_class": (137, 87, 229),
    "drop": (60, 60, 60),
    "unsure": (230, 140, 20),
}


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, keep_default_na=False, na_values=[""]) if path.exists() else None


@dataclass
class Catalog:
    manifest: Path = MANIFEST
    configs: Path = CONFIGS
    split_dir: Path = SPLIT_DIR
    raw: Path = RAW
    cache: Path = CACHE

    @cached_property
    def images(self) -> pd.DataFrame:
        """The manifest, in manifest order (the row index is the index `pixels.npy` uses)."""
        im = pd.read_csv(self.manifest / "images.csv", keep_default_na=False, na_values=[""])
        im["folder"] = im.raw_class_folder.astype(int)
        return im

    @cached_property
    def classes(self) -> pd.DataFrame:
        """class_folder, character, label -- one row per folder, sorted."""
        folders = sorted(self.images.folder.unique())
        lab = _read(self.configs / "class_labels.csv")
        char = {} if lab is None else dict(zip(lab.class_folder, lab.character.fillna(""), strict=True))
        df = pd.DataFrame({"class_folder": folders})
        df["character"] = df.class_folder.map(char).fillna("")
        df["label"] = [f"{f} {c}".strip() for f, c in zip(df.class_folder, df.character, strict=True)]
        return df

    @cached_property
    def label_of(self) -> dict[int, str]:
        return dict(zip(self.classes.class_folder, self.classes.label, strict=True))

    @cached_property
    def items(self) -> pd.DataFrame:
        """One row per (folder, sha256): the reviewable units."""
        im = self.images.assign(row=np.arange(len(self.images)))
        im = im.sort_values("path")
        copies = im.groupby(["folder", "sha256"]).path.agg(list)
        items = im.drop_duplicates(["folder", "sha256"]).reset_index(drop=True)
        items["copies"] = [copies[k] for k in zip(items.folder, items.sha256, strict=True)]
        items["n_copies"] = items.copies.map(len)

        out = _read(self.manifest / "findings_outliers.csv")
        if out is not None:
            items["outlier"] = items.path.map(dict(zip(out.path, out.reasons, strict=True))).fillna("")
        else:
            items["outlier"] = ""

        split = _read(self.split_dir / "images.csv")
        if split is not None:
            s = split.set_index("path")
            items["writer"] = items.path.map(s.writer_id).fillna("")
            items["split"] = items.path.map(s.split).fillna("excluded")
            items.loc[~items.path.map(s.included).fillna(False).astype(bool), "split"] = "excluded"
        else:
            items["writer"], items["split"] = "", ""

        items["prior_ruling"] = items.path.map(self._prior_rulings()).fillna("")
        items = items.sort_values(["folder", "path"]).reset_index(drop=True)
        return items[["row", "path", "folder", "class_folder", "sha256", "sha_group_id", "copies", "n_copies",
                      "width", "height", "file_name", "writer", "split", "outlier",
                      "sha_group_spans_classes", "near_dup_group_id", "near_dup_group_spans_classes",
                      "prior_ruling"]]

    def _prior_rulings(self) -> dict[str, str]:
        """Decisions already in `configs/` from other tools, as short text -- context, never edited here."""
        notes: dict[str, str] = {}
        gr = _read(self.configs / "cross_class_rulings.csv")
        if gr is not None:
            by_sha = gr.set_index("sha_group_id")
            for p, sid in zip(self.images.path, self.images.sha_group_id, strict=True):
                if sid in by_sha.index:
                    r = by_sha.loc[sid]
                    cc = "" if pd.isna(r.correct_class) else f" → {int(r.correct_class)}"
                    notes[p] = f"cross_class_rulings: {r.ruling}{cc}"
        ir = _read(self.configs / "image_rulings.csv")
        if ir is not None:
            for r in ir[ir.ruling.notna()].itertuples():
                cc = "" if pd.isna(r.correct_class) else f" → {int(r.correct_class)}"
                notes[r.path] = f"image_rulings ({r.source}): {r.ruling}{cc}"
        return notes

    @cached_property
    def item_of_row(self) -> np.ndarray:
        """manifest row -> item index (every copy maps to its representative)."""
        out = np.full(len(self.images), -1, np.int64)
        path_row = dict(zip(self.images.path, range(len(self.images)), strict=True))
        for k, copies in enumerate(self.items.copies):
            for p in copies:
                out[path_row[p]] = k
        return out

    @cached_property
    def prototypes(self) -> dict[int, np.ndarray]:
        f = self.cache / "prototypes_n150_s48_fitwhite.npz"
        if not f.exists():
            return {}
        z = np.load(f)
        return {int(k): z[k] for k in z.files}

    # ---- drawing ---------------------------------------------------------------------------

    def gray(self, path: str) -> Image.Image:
        with Image.open(self.raw / path) as im:
            return im.convert("L")

    def render(self, path: str, cell: int = 96, true_size: bool = False, state: str = "",
               border: int = 4, pad: int = 10) -> Image.Image:
        """The glyph on white, NEAREST-upscaled so strokes stay crisp, framed in its state colour.

        `true_size` scales every glyph by the same factor, so a tone mark looks small next to a
        consonant -- the absolute-size cue that separates ่ from ๅ, which letterboxing erases.
        `pad` keeps the glyph's ink clear of the frame so it reads at a glance in the gallery grid,
        without having to hover or open the image full-size.
        """
        g = self.gray(path)
        inner = cell - 2 * border - 2 * pad
        w, h = g.size
        s = inner / max(TRUE_SIZE_REF, w, h) if true_size else inner / max(w, h)
        g = g.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (cell - 2 * border, cell - 2 * border), (255, 255, 255))
        canvas.paste(g, ((canvas.width - g.width) // 2, (canvas.height - g.height) // 2))
        return ImageOps.expand(canvas, border=border, fill=STATE_COLORS.get(state, STATE_COLORS[""]))

    def thumb(self, path: str, cell: int = 96, true_size: bool = False, state: str = "") -> str:
        """`render` cached as a PNG under cache/review_thumbs/, returned as a file path for gr.Gallery."""
        key = hashlib.sha1(f"{path}|{cell}|{true_size}|{state}|v2".encode()).hexdigest()[:20]
        f = self.cache / "review_thumbs" / f"{key}.png"
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(f".{np.random.randint(1 << 30)}.tmp")
            self.render(path, cell, true_size, state).save(tmp, "PNG")
            tmp.replace(f)
        return str(f)

    def big(self, path: str, size: int = 320) -> Image.Image:
        g = self.gray(path)
        s = max(1, size // max(g.size))
        return g.resize((g.width * s, g.height * s), Image.Resampling.NEAREST)

    def prototype_image(self, folder: int, size: int = 144) -> Image.Image | None:
        p = self.prototypes.get(int(folder))
        return None if p is None else Image.fromarray(p).resize((size, size), Image.Resampling.NEAREST)
