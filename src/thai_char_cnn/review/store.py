"""Where review decisions live: one SQLite file shared by every session of the app.

Both tables are append-only logs -- a click is one INSERT, nothing is ever updated or deleted, so
the full history survives and "undo" is just a newer row (`verdict = 'clear'`). The *current*
decision of a reviewer on an image is their latest row. WAL mode plus one short connection per call
keeps concurrent sessions safe.

Consensus over an image's current decisions (per reviewer), highest first:

- `conflict`  -- reviewers gave different final verdicts (ok / wrong_class→X / drop)
- `flag`      -- someone marked it suspicious and has not resolved it
- the single final verdict everyone who gave one agrees on (`ok`, `wrong_class`, `drop`)
- `unsure`    -- only unsure votes
"""

import shutil
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import DATA

DB_PATH = DATA / "review" / "review.db"

VERDICTS = ("ok", "flag", "wrong_class", "drop", "unsure", "clear")
FINAL = ("ok", "wrong_class", "drop")
STATUSES = ("not_started", "in_progress", "needs_review", "completed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS image_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    folder_class INTEGER NOT NULL,
    reviewer TEXT NOT NULL,
    verdict TEXT NOT NULL,
    proposed_class INTEGER,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_dec_path ON image_decisions(path, reviewer, id);
CREATE INDEX IF NOT EXISTS ix_dec_folder ON image_decisions(folder_class);
CREATE TABLE IF NOT EXISTS class_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_folder INTEGER NOT NULL,
    status TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
"""

_LATEST = """
SELECT d.path, d.folder_class, d.reviewer, d.verdict, d.proposed_class, d.note, d.created_at
FROM image_decisions d
JOIN (SELECT MAX(id) AS id FROM image_decisions {where} GROUP BY path, reviewer) l ON d.id = l.id
WHERE d.verdict != 'clear'
"""

DECISION_COLS = ["path", "folder_class", "reviewer", "verdict", "proposed_class", "note", "created_at"]


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path = DB_PATH, backup: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if backup and self.path.exists():
            with closing(self._connect()) as src, closing(sqlite3.connect(self.path.with_suffix(".db.bak"))) as dst:
                src.backup(dst)
        with closing(self._connect()) as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def _query(self, sql: str, params=()) -> pd.DataFrame:
        with closing(self._connect()) as c:
            return pd.read_sql_query(sql, c, params=params)

    # ---- writes -----------------------------------------------------------------------------

    def record_many(self, rows: list[dict], reviewer: str) -> int:
        """rows: dicts with path, folder_class, verdict and optional proposed_class, note."""
        reviewer = reviewer.strip()
        assert reviewer, "a decision needs a reviewer"
        t = now()
        vals = []
        for r in rows:
            v = r["verdict"]
            assert v in VERDICTS, f"unknown verdict {v!r}"
            pc = r.get("proposed_class")
            assert (v == "wrong_class") == (pc is not None), "proposed_class goes with wrong_class, only"
            vals.append((r["path"], int(r["folder_class"]), reviewer, v,
                         None if pc is None else int(pc), r.get("note") or "", t))
        with closing(self._connect()) as c, c:
            c.executemany("INSERT INTO image_decisions (path, folder_class, reviewer, verdict, proposed_class,"
                          " note, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)", vals)
        return len(vals)

    def record(self, path: str, folder_class: int, reviewer: str, verdict: str,
               proposed_class: int | None = None, note: str = "") -> None:
        self.record_many([dict(path=path, folder_class=folder_class, verdict=verdict,
                               proposed_class=proposed_class, note=note)], reviewer)

    def set_class_status(self, class_folder: int, status: str, reviewer: str, note: str = "") -> None:
        assert status in STATUSES, f"unknown status {status!r}"
        assert reviewer.strip(), "a status needs a reviewer"
        with closing(self._connect()) as c, c:
            c.execute("INSERT INTO class_status (class_folder, status, reviewer, note, created_at)"
                      " VALUES (?, ?, ?, ?, ?)", (int(class_folder), status, reviewer.strip(), note or "", now()))

    # ---- reads ------------------------------------------------------------------------------

    def current(self, folder: int | None = None, paths: list[str] | None = None) -> pd.DataFrame:
        """Each reviewer's latest non-cleared decision per image."""
        if paths is not None:
            if not paths:
                return pd.DataFrame(columns=DECISION_COLS)
            where, params = f"WHERE path IN ({','.join('?' * len(paths))})", list(paths)
        elif folder is not None:
            where, params = "WHERE folder_class = ?", [int(folder)]
        else:
            where, params = "", []
        df = self._query(_LATEST.format(where=where), params)
        df["proposed_class"] = df.proposed_class.astype("Int64")
        return df

    def log(self) -> pd.DataFrame:
        return self._query("SELECT * FROM image_decisions ORDER BY id")

    def status_log(self) -> pd.DataFrame:
        return self._query("SELECT * FROM class_status ORDER BY id")

    def class_status(self) -> pd.DataFrame:
        """Latest manual status per class: class_folder, status, reviewer, note, created_at."""
        return self._query("SELECT s.class_folder, s.status, s.reviewer, s.note, s.created_at FROM class_status s"
                           " JOIN (SELECT MAX(id) AS id FROM class_status GROUP BY class_folder) l ON s.id = l.id")

    def export_logs(self, configs: Path) -> None:
        """Plain-CSV copies of both logs, so the decisions can be tracked in git."""
        for name, df in (("review_log.csv", self.log()), ("review_class_status.csv", self.status_log())):
            tmp = configs / f".{name}.tmp"
            df.to_csv(tmp, index=False)
            shutil.move(tmp, configs / name)


def consensus(cur: pd.DataFrame) -> pd.DataFrame:
    """Per path: state, proposed_class, n_reviewers, n_agree (reviewers backing the final verdict),
    votes (short text), last_at."""
    cols = ["state", "proposed_class", "n_reviewers", "n_agree", "votes", "last_at"]
    if cur.empty:
        return pd.DataFrame(columns=cols).rename_axis("path")
    wrong = cur.verdict == "wrong_class"
    key = cur.verdict.where(~wrong, "wrong_class:" + cur.proposed_class.astype("string"))
    by = cur.groupby("path", sort=False)
    out = pd.DataFrame({"n_reviewers": by.size(), "last_at": by.created_at.max()})
    out["votes"] = (cur.reviewer + ": " + key.str.replace("wrong_class:", "→")).groupby(cur.path).agg(", ".join)

    fin = cur.verdict.isin(FINAL)
    fkey = key[fin].groupby(cur.path[fin])
    n_keys = fkey.nunique().reindex(out.index, fill_value=0)
    first = fkey.first().reindex(out.index).astype("string")
    flagged = (cur.verdict == "flag").groupby(cur.path).any().reindex(out.index)

    out["state"] = np.select([n_keys > 1, flagged, n_keys == 1], ["conflict", "flag", first.str.split(":").str[0]],
                             "unsure")
    one = (n_keys == 1) & ~flagged
    out["n_agree"] = fkey.size().reindex(out.index, fill_value=0).where(one, 0).astype(int)
    pc = first.str.split(":").str[1].where(out.state == "wrong_class")
    out["proposed_class"] = pd.to_numeric(pc, errors="coerce").astype("Int64")
    return out[cols].rename_axis("path")
