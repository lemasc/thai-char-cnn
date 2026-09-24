"""Progress numbers for the statistics tab, computed from the catalog and the current decisions."""

import pandas as pd

from .catalog import Catalog
from .store import Store, consensus

STATE_COLS = ["ok", "flag", "wrong_class", "drop", "unsure", "conflict"]


def item_states(cat: Catalog, cons: pd.DataFrame) -> pd.DataFrame:
    """items + consensus columns (state '' when nobody has decided)."""
    it = cat.items[["path", "folder"]].join(cons, on="path")
    it["state"] = it.state.fillna("")
    it["n_reviewers"] = it.n_reviewers.fillna(0).astype(int)
    return it


def class_stats(cat: Catalog, store: Store, cons: pd.DataFrame | None = None) -> pd.DataFrame:
    cons = consensus(store.current()) if cons is None else cons
    it = item_states(cat, cons)
    counts = pd.crosstab(it.folder, it.state).reindex(columns=STATE_COLS, fill_value=0)
    df = cat.classes.set_index("class_folder")[["character"]].copy()
    df["n_raw"] = cat.images.groupby("folder").size()
    df["n_unique"] = it.groupby("folder").size()
    df["reviewed"] = it[it.state != ""].groupby("folder").size()
    df = df.join(counts).fillna(0)
    num = ["n_raw", "n_unique", "reviewed", *STATE_COLS]
    df[num] = df[num].astype(int)
    df["pct_reviewed"] = (100 * df.reviewed / df.n_unique.clip(lower=1)).round(1)
    st = store.class_status().set_index("class_folder")
    df["status"] = st.status.reindex(df.index).fillna("not_started")
    df["status_by"] = st.reviewer.reindex(df.index).fillna("")
    df["status_note"] = st.note.reindex(df.index).fillna("")
    df["last_activity"] = it.groupby("folder").last_at.max().reindex(df.index).fillna("")
    df = df.reset_index().rename(columns={"index": "class_folder"})
    return df[["class_folder", "character", "status", "pct_reviewed", "n_raw", "n_unique", "reviewed",
               *STATE_COLS, "status_by", "status_note", "last_activity"]]


def reassign_matrix(cat: Catalog, cons: pd.DataFrame) -> pd.DataFrame:
    """Folder -> proposed class, over images whose consensus is wrong_class."""
    it = item_states(cat, cons)
    w = it[it.state == "wrong_class"]
    if w.empty:
        return pd.DataFrame(columns=["from", "to", "images"])
    m = w.groupby(["folder", "proposed_class"]).size().rename("images").reset_index()
    m["from"] = m.folder.map(cat.label_of)
    m["to"] = m.proposed_class.astype(int).map(cat.label_of)
    return m.sort_values("images", ascending=False)[["from", "to", "images"]]


def reviewer_stats(store: Store) -> pd.DataFrame:
    cur = store.current()
    cols = ["reviewer", "images", "ok", "flag", "wrong_class", "drop", "unsure", "last_at", "agreement"]
    if cur.empty:
        return pd.DataFrame(columns=cols)
    df = pd.crosstab(cur.reviewer, cur.verdict).reindex(columns=cols[2:7], fill_value=0)
    df.insert(0, "images", cur.groupby("reviewer").size())
    df["last_at"] = cur.groupby("reviewer").created_at.max()
    # agreement: of this reviewer's final verdicts on images someone else also finalised, the
    # share where every final verdict matches
    cons = consensus(cur)
    fin = cur[cur.verdict.isin(["ok", "wrong_class", "drop"])]
    shared = fin.path.map(fin.groupby("path").reviewer.nunique()) > 1
    s = fin[shared.to_numpy()]
    agree = (s.path.map(cons.state) != "conflict").groupby(s.reviewer).mean()
    df["agreement"] = (100 * agree).round(1).reindex(df.index)
    return df.reset_index()[cols]


def conflicts(cat: Catalog, cons: pd.DataFrame) -> pd.DataFrame:
    it = cat.items.reset_index(names="item").join(cons, on="path")
    c = it[it.state == "conflict"]
    return pd.DataFrame({"item": c.item, "class": c.folder.map(cat.label_of), "path": c.path,
                         "votes": c.votes, "last_at": c.last_at}).sort_values("last_at", ascending=False)
