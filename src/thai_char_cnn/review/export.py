"""Hand the review's decisions to the pipeline as `configs/image_rulings.csv` rows.

The review DB saves every click; this is the separate, deliberate step that changes what
`03_split` trains on. Only decisions the reviewers agree on become rulings:

- consensus `wrong_class` -> `reassign` to the proposed class
- consensus `drop`        -> `drop`
- `ok` / `unsure` / `flag` / `conflict` -> **no row** (the folder label stands)

By default a ruling needs at least two agreeing reviewers; `min_agree=1` accepts one.
A decision on an item covers every byte-identical copy in the same folder.

Rows other tools wrote (e.g. 02's `cross_class_near` queue) are kept as they are. This app only
replaces rows with `source == "audit_app"`. If another tool has already ruled a path, that ruling
wins and the path is listed in the report instead.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd

from ..labels import load_image_rulings
from ..paths import CONFIGS
from .catalog import Catalog
from .store import Store, consensus, now

SOURCE = "audit_app"
COLS = ["path", "ruling", "correct_class", "source", "note", "decided_at"]


def proposed_rows(cat: Catalog, store: Store, min_agree: int = 2) -> pd.DataFrame:
    cur = store.current()
    cons = consensus(cur)
    it = cat.items.join(cons, on="path")
    it = it[it.state.isin(["wrong_class", "drop"]) & (it.n_agree >= min_agree)]
    notes = cur[cur.note != ""].groupby("path").note.agg(" | ".join)
    t = now()
    rows = []
    for r in it.itertuples():
        note = f"{r.votes}" + (f" -- {notes[r.path]}" if r.path in notes.index else "")
        for p in r.copies:
            rows.append(dict(path=p, ruling="reassign" if r.state == "wrong_class" else "drop",
                             correct_class=int(r.proposed_class) if r.state == "wrong_class" else pd.NA,
                             source=SOURCE, note=note, decided_at=t))
    return pd.DataFrame(rows, columns=COLS)


def export(cat: Catalog, store: Store, configs: Path = CONFIGS, min_agree: int = 2) -> dict:
    new = proposed_rows(cat, store, min_agree)
    f = configs / "image_rulings.csv"
    prior = (pd.read_csv(f, keep_default_na=False, na_values=[""]) if f.exists()
             else pd.DataFrame(columns=COLS))
    prior = prior.reindex(columns=list(dict.fromkeys([*COLS, *prior.columns])))
    others = prior[prior.source != SOURCE]
    # another tool's *decided* row wins; its blank (deferred) rows are not decisions and give way
    held = set(others.loc[others.ruling.notna(), "path"])
    skipped = sorted(set(new.path) & held)
    new = new[~new.path.isin(held)]
    others = others[~others.path.isin(new.path)]
    out = pd.concat([others, new.reindex(columns=others.columns)], ignore_index=True)
    out["correct_class"] = pd.to_numeric(out.correct_class, errors="coerce").astype("Int64")

    known = set(cat.classes.class_folder)
    bad = set(new.correct_class.dropna().astype(int)) - known
    assert not bad, f"proposed classes not in the class list: {bad}"
    assert set(new.path) <= set(cat.images.path), "export names paths not in the manifest"

    # write beside the target, validate with the reader 03 uses, then swap in atomically
    with tempfile.TemporaryDirectory(dir=configs) as d:
        tmp = Path(d) / "image_rulings.csv"
        out.to_csv(tmp, index=False)
        load_image_rulings(Path(d))
        os.replace(tmp, f)
    store.export_logs(configs)
    return dict(written=len(new), reassign=int((new.ruling == "reassign").sum()),
                drop=int((new.ruling == "drop").sum()), kept_other_sources=len(others),
                skipped_already_ruled=skipped)
