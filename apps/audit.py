"""Shared image-review app: scan each class, flag and correct labels, look up similar images.

    uv run python apps/audit.py                       # http://localhost:7860
    uv run python apps/audit.py --host 0.0.0.0        # let colleagues on the network in
    AUDIT_USERS="alice:pw,bob:pw" uv run python apps/audit.py --host 0.0.0.0   # with logins
    AUDIT_USERS="alice:pw,bob:pw" uv run python apps/audit.py --share          # public *.gradio.live link

Every click is saved immediately to `data/<dataset>/review/review.db` (one SQLite file shared by all
sessions). Nothing reaches the pipeline until someone presses **Export** on the last tab, which
writes agreed decisions to `configs/image_rulings.csv` (source `audit_app`) and the full decision
log to `configs/review_log.csv`. `AUDIT_DB=/some/file.db` points the app at another database.
"""

import argparse
import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from thai_char_cnn.paths import CONFIGS
from thai_char_cnn.review import export, stats
from thai_char_cnn.review.catalog import Catalog
from thai_char_cnn.review.index import Index
from thai_char_cnn.review.store import DB_PATH, STATUSES, Store, consensus

USERS = dict(u.split(":", 1) for u in os.environ.get("AUDIT_USERS", "").split(",") if ":" in u)

CAT = Catalog()
STORE = Store(Path(os.environ.get("AUDIT_DB", DB_PATH)))
print("loading similarity index (built on first run, ~15 s) ...")
IX = Index(CAT)
ITEMS = CAT.items
ITEM_ROWS = ITEMS.row.to_numpy()
BY_FOLDER = {f: np.flatnonzero(ITEMS.folder.to_numpy() == f) for f in CAT.classes.class_folder}
CLASS_CHOICES = [(lab, int(f)) for f, lab in zip(CAT.classes.class_folder, CAT.classes.label, strict=True)]
DOWNLOADS = Path(tempfile.mkdtemp(prefix="audit_dl_"))

MARK = {"ok": "✓", "flag": "⚑", "wrong_class": "→", "drop": "✗", "unsure": "?", "conflict": "⚠"}
ORDERS = ["model doubt first", "like confirmed wrong", "outliers first", "random", "file name"]
BAD = ["wrong_class", "drop"]
FILTERS = ["not reviewed by me", "not reviewed by anyone", "all", "flagged", "conflicts",
           "wrong class / drop", "unsure", "model disagrees", "outliers"]


def lab(folder) -> str:
    return CAT.label_of.get(int(folder), str(folder))


def char(folder) -> str:
    """the character alone (falls back to the folder id) -- for captions too narrow for both"""
    return lab(folder).split()[-1]


def reviewer(name: str, request: gr.Request | None, required: bool = True) -> str:
    if USERS and request is not None and request.username:
        return request.username
    name = (name or "").strip()
    if required and not name:
        raise gr.Error("Type your name in the box at the top before reviewing — every decision is attributed.")
    return name


# ---- page building --------------------------------------------------------------------------

def like_wrong(rows: np.ndarray, bad: np.ndarray, good: np.ndarray, k: int = 5) -> np.ndarray:
    """how much closer each image sits to the team's confirmed wrong-class / drop images of its
    folder than to the confirmed OK ones (mean cosine of the k nearest of each). A folder label the
    model has memorised hides from "model doubt first"; this finds it from the verdicts instead."""
    def near(ref):
        s = IX.emb[rows] @ IX.emb[ref].T
        n = min(k, s.shape[1])
        return -np.partition(-s, n - 1, axis=1)[:, :n].mean(1)
    return near(bad) - near(good) if len(good) else near(bad)


def page_list(folder: int, order: str, filt: str, me: str) -> tuple[list[int], str]:
    """item indices in display order, plus a note when the order had to fall back"""
    idx = BY_FOLDER[int(folder)]
    it = ITEMS.iloc[idx]
    cur = STORE.current(folder=int(folder))
    cons = consensus(cur)
    state = it.path.map(cons.state).fillna("").to_numpy()
    mine = set(cur.path[cur.reviewer == me])
    rows = it.row.to_numpy()
    keep = {
        "not reviewed by me": ~it.path.isin(mine).to_numpy(),
        "not reviewed by anyone": state == "",
        "all": np.ones(len(it), bool),
        "flagged": state == "flag",
        "conflicts": state == "conflict",
        "wrong class / drop": np.isin(state, BAD),
        "unsure": state == "unsure",
        "model disagrees": IX.top_folder[rows, 0] != int(folder),
        "outliers": (it.outlier != "").to_numpy(),
    }[filt]
    bad, good = rows[np.isin(state, BAD)], rows[state == "ok"]
    note = ""
    idx, rows, it = idx[keep], rows[keep], it[keep]
    if order == "like confirmed wrong" and len(bad):
        o = np.argsort(-like_wrong(rows, bad, good), kind="stable")
    elif order in ("model doubt first", "like confirmed wrong"):
        if order != "model doubt first":
            note = " · *no confirmed wrong class / drop in this class yet — ordered by model doubt*"
        o = np.argsort(IX.p_folder[rows], kind="stable")
    elif order == "outliers first":
        o = np.lexsort((IX.p_folder[rows], (it.outlier == "").to_numpy()))
    elif order == "random":
        o = np.random.default_rng(42 + int(folder)).permutation(len(idx))
    else:
        o = np.argsort(it.file_name.to_numpy(), kind="stable")
    return idx[o].tolist(), note


def caption(k: int, mine: dict, cons: pd.DataFrame) -> str:
    r = ITEMS.iloc[k]
    parts = []
    v = mine.get(r.path)
    if v is not None:
        parts.append(MARK[v[0]] + (char(v[1]) if v[0] == "wrong_class" else ""))
    if r.path in cons.index:
        c = cons.loc[r.path]
        others = c.n_reviewers - (v is not None)
        if c.state == "conflict":
            parts.append("⚠")
        elif others > 0:
            parts.append(f"👥{others}")
    top = IX.top_folder[r.row, 0]
    if top != r.folder:
        parts.append(f"model:{char(top)}")
    if r.n_copies > 1:
        parts.append(f"×{r.n_copies}")
    return " ".join(parts) or f"{r.width}×{r.height}"


def render_gallery(page_items: list[int], me: str, true_size: bool, cell: int = 96):
    paths = ITEMS.path.iloc[page_items].tolist()
    cur = STORE.current(paths=paths)
    cons = consensus(cur)
    m = cur[cur.reviewer == me]
    mine = {p: (v, pc) for p, v, pc in zip(m.path, m.verdict, m.proposed_class, strict=True)}
    return [(CAT.thumb(p, cell, true_size, mine.get(p, ("",))[0]), caption(k, mine, cons))
            for k, p in zip(page_items, paths, strict=True)]


def class_header(folder: int) -> str:
    folder = int(folder)
    it = ITEMS.iloc[BY_FOLDER[folder]]
    state = it.path.map(consensus(STORE.current(folder=folder)).state).fillna("")
    n = state.value_counts()
    n_raw = int((CAT.images.folder == folder).sum())
    pct = 100 * (state != "").mean()
    st = STORE.class_status().set_index("class_folder")
    s = st.loc[folder] if folder in st.index else None
    return (f"### {lab(folder)}\n"
            f"**{len(it):,}** unique images ({n_raw:,} files) · reviewed **{pct:.1f}%** · "
            + " · ".join(f"{MARK[k]} {n.get(k, 0)}" for k in ("ok", "flag", "wrong_class", "drop", "unsure", "conflict"))
            + "  \nstatus **" + ("not_started" if s is None else s.status) + "**"
            + ("" if s is None else f" (by {s.reviewer})" + (f" — {s.note}" if s.note else "")))


def load_page(folder, order, filt, page, size, true_size, name, request: gr.Request):
    me = reviewer(name, request, required=False)
    lst, note = page_list(folder, order, filt, me)
    size = int(size)
    n_pages = max(1, -(-len(lst) // size))
    page = int(min(max(0, page), n_pages - 1))
    items = lst[page * size:(page + 1) * size]
    info = f"page **{page + 1} / {n_pages}** · {len(lst):,} images match *{filt}*" + note
    return render_gallery(items, me, true_size), items, page, info, class_header(folder)


# ---- detail panel ---------------------------------------------------------------------------

def vote_md(nb) -> str:
    total = int(nb.vote.sum())
    lines = [f"`{lab(f):<6}` {'█' * round(20 * n / max(1, total))} {n}/{total}" for f, n in nb.vote.items()]
    return "**Class vote among the neighbours**\n\n" + "  \n".join(lines)


def neighbours(k, signal, scope, n, true_size):
    if k is None:
        return [], [], ""
    r = ITEMS.iloc[int(k)]
    nb = IX.neighbours(int(r.row), signal, scope, int(n))
    fmt = (lambda s: f"rms {s:.1f}") if signal == "pixel" else (lambda s: f"{s:.2f}".lstrip("0"))
    gal = [(CAT.thumb(CAT.images.path.iloc[j], 96, true_size), f"{char(IX.folder[j])} {fmt(s)}")
           for j, s in zip(nb.rows, nb.score, strict=True)]
    return gal, nb.rows.tolist(), vote_md(nb)


def compare(k, proposed):
    if k is None:
        return []
    r = ITEMS.iloc[int(k)]
    out = [(CAT.big(r.path, 144), "this image")]
    for f, what in ((r.folder, "folder"), (proposed, "→")):
        if f is not None and (p := CAT.prototype_image(int(f))) is not None:
            out.append((p, f"{what} {lab(f)}"))
    return out


def detail(k, me: str):
    """big image, info, model, decisions, proposed default, compare strip"""
    if k is None:
        return None, "*Switch to **Inspect** and click an image.*", "", "", None, []
    r = ITEMS.iloc[int(k)]
    info = [f"**{lab(r.folder)}** · `{r.path}` · {r.width}×{r.height} px · filename group `{r.writer or '?'}`"
            f" · split **{r.split or '?'}**" + (f" · {r.n_copies} identical copies in this folder" if r.n_copies > 1 else "")]
    if r.outlier:
        info.append(f"outlier: {r.outlier}")
    if r.sha_group_spans_classes:
        info.append("⚠ a byte-identical copy sits in another class folder")
    elif r.near_dup_group_spans_classes:
        info.append("⚠ a near-duplicate (RMS ≤ 4) sits in another class folder")
    if r.prior_ruling:
        info.append(f"already in configs: {r.prior_ruling}")

    tops = ", ".join(f"**{lab(f)}** {p:.0%}" for f, p in zip(IX.top_folder[r.row], IX.top_p[r.row], strict=True) if p >= 0.005)
    # the split holds out filename groups, not fonts: a val image may share its font with train
    # images, so even a val prediction is a second opinion, not an independent one
    trust = {"train": "⚠ seen in training — the prediction partly memorises the folder label",
             "val": "held-out filename group — not trained on this image (its font may still be in train)",
             }.get(r.split, "not used in training")
    model = f"**Model** ({IX.meta['name']}): {tops}  \nP(folder class) = {IX.p_folder[r.row]:.1%} · {trust}"

    cur = STORE.current(paths=[r.path])
    if cur.empty:
        dec = "**Decisions:** none yet"
    else:
        c = consensus(cur).loc[r.path]
        rows = [f"- **{x.reviewer}**: {x.verdict}" + (f" → {lab(x.proposed_class)}" if x.verdict == "wrong_class" else "")
                + (f" — *{x.note}*" if x.note else "") + f" <sub>{x.created_at}</sub>" for x in cur.itertuples()]
        dec = f"**Decisions** — team: **{c.state}**\n" + "\n".join(rows)

    # prefill only a real alternative: the model's choice when it disagrees, or a runner-up it
    # gives >= 5% -- otherwise the "2" key would reassign to a class nobody suggested
    t, tp = IX.top_folder[r.row], IX.top_p[r.row]
    proposed = int(t[0]) if t[0] != r.folder else (int(t[1]) if tp[1] >= 0.05 else None)
    mine = cur[cur.reviewer == me]
    if len(mine) and mine.verdict.iloc[0] == "wrong_class":
        proposed = int(mine.proposed_class.iloc[0])
    return CAT.big(r.path), "  \n".join(info), model, dec, proposed, compare(k, proposed)


def open_item(k, name, signal, scope, n, true_size, auto_sim, request: gr.Request):
    me = reviewer(name, request, required=False)
    big, info, model, dec, proposed, cmp = detail(k, me)
    sim = neighbours(k, signal, scope, n, true_size) if auto_sim and k is not None else ([], [], "")
    return k, big, info, model, dec, proposed, cmp, *sim


# ---- wiring ---------------------------------------------------------------------------------

KEYS_JS = """
<script>
document.addEventListener('keydown', (e) => {
  const t = e.target;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
  const id = {'1': 'btn-ok', '2': 'btn-wrong', '3': 'btn-drop', '4': 'btn-unsure', 'f': 'btn-flag',
              'j': 'btn-next-item', 'k': 'btn-prev-item', 'n': 'btn-next-page', 'p': 'btn-prev-page'}[e.key];
  const b = id && document.getElementById(id);
  if (b) { b.click(); e.preventDefault(); }
});
</script>
"""

CSS = """
#gallery .caption-label, #neigh .caption-label { font-size: 12px; }
/* Gradio sizes the grid as `columns` x minmax(100px, 1fr), which overflows sideways in a narrow
   column. Fit as many columns as the width allows instead; extra thumbnails wrap downward. */
#gallery .grid-container, #neigh .grid-container {
  grid-template-columns: repeat(auto-fill, minmax(84px, 1fr));
  grid-template-rows: none;
  grid-auto-rows: auto;
}
#cmp .grid-container { grid-template-columns: repeat(3, minmax(0, 1fr)); grid-template-rows: none; grid-auto-rows: auto; }
#gallery .grid-wrap, #neigh .grid-wrap, #cmp .grid-wrap { overflow-x: hidden; }
.tile { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 10px 14px; min-width: 130px; }
.tile .v { font-size: 22px; font-weight: 600; }
.tile .l { font-size: 12px; opacity: .75; }
.tiles { display: flex; flex-wrap: wrap; gap: 10px; }
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="Dataset audit") as demo:
        name_saved = gr.BrowserState("", storage_key="audit_reviewer")
        page_items = gr.State([])
        cur_item = gr.State(None)
        neigh_rows = gr.State([])

        with gr.Row():
            gr.Markdown("## Dataset audit — shared review")
            name = gr.Textbox(label="Reviewer", placeholder="your name", scale=0, min_width=200,
                              visible=not USERS)

        with gr.Tabs() as tabs:
            # ======================= review ========================================================
            with gr.Tab("Review", id="review"):
                with gr.Row():
                    folder = gr.Dropdown(CLASS_CHOICES, value=CLASS_CHOICES[0][1], label="Class", scale=2)
                    order = gr.Dropdown(ORDERS, value=ORDERS[0], label="Order", scale=1)
                    filt = gr.Dropdown(FILTERS, value=FILTERS[0], label="Show", scale=1)
                    size = gr.Slider(24, 144, value=24, step=4, label="Per page", scale=1)
                    true_size = gr.Checkbox(False, label="True relative size", scale=0, min_width=120,
                                            info="keep glyph sizes comparable")
                header = gr.Markdown()
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            mode = gr.Radio(["Scan: click toggles ⚑", "Inspect: click opens"], value="Scan: click toggles ⚑",
                                            show_label=False, scale=2)
                            prev_page = gr.Button("◀ page", elem_id="btn-prev-page", size="sm", scale=0)
                            next_page = gr.Button("page ▶", elem_id="btn-next-page", size="sm", scale=0)
                            page_ok = gr.Button("✓ Mark rest of page OK → next", variant="primary", size="sm", scale=1)
                        page_info = gr.Markdown()
                        page = gr.State(0)
                        gallery = gr.Gallery(columns=12, height="auto", allow_preview=False, object_fit="contain",
                                             show_label=False, elem_id="gallery", interactive=False)
                        gr.Markdown("Border = **your** verdict: <span style='color:#2ea043'>■ ok</span> "
                                    "<span style='color:#da3633'>■ flag</span> <span style='color:#8957e5'>■ wrong class</span> "
                                    "<span style='color:#3c3c3c'>■ drop</span> <span style='color:#e68c14'>■ unsure</span> · "
                                    "caption: 👥n others decided · ⚠ reviewers disagree · model:X model prefers X · ×n identical copies. "
                                    "Keys: **1** ok · **2** wrong class · **3** drop · **4** unsure · **f** flag · **j/k** next/prev image · **n/p** page")
                        with gr.Accordion("Class status (manual)", open=False):
                            with gr.Row():
                                status = gr.Radio(list(STATUSES), label="Status of this class", scale=2)
                                status_note = gr.Textbox(label="Note", scale=2)
                                status_save = gr.Button("Save status", scale=0)

                    with gr.Column(scale=2):
                        with gr.Row():
                            big = gr.Image(show_label=False, height=220, interactive=False, scale=1)
                            cmp = gr.Gallery(columns=3, height=220, allow_preview=False, show_label=False, scale=2, elem_id="cmp",
                                             object_fit="contain", interactive=False)
                        info = gr.Markdown("*Switch to **Inspect** and click an image.*")
                        model_md = gr.Markdown()
                        with gr.Row():
                            b_ok = gr.Button("1 · OK", elem_id="btn-ok", variant="primary", size="sm")
                            b_drop = gr.Button("3 · Drop", elem_id="btn-drop", size="sm")
                            b_unsure = gr.Button("4 · Unsure", elem_id="btn-unsure", size="sm")
                            b_flag = gr.Button("f · Flag", elem_id="btn-flag", size="sm")
                            b_clear = gr.Button("Clear mine", size="sm")
                        with gr.Row():
                            proposed = gr.Dropdown(CLASS_CHOICES, label="Correct class", scale=2)
                            b_wrong = gr.Button("2 · Wrong class →", elem_id="btn-wrong", variant="stop", scale=1)
                        with gr.Row():
                            note = gr.Textbox(label="Note (optional, saved with the next verdict)", scale=3)
                            advance = gr.Checkbox(True, label="Auto-advance", scale=1)
                        with gr.Row():
                            b_prev = gr.Button("◀ prev (k)", elem_id="btn-prev-item", size="sm")
                            b_next = gr.Button("next (j) ▶", elem_id="btn-next-item", size="sm")
                        decisions = gr.Markdown()
                        with gr.Accordion("Find similar images", open=True):
                            with gr.Row():
                                signal = gr.Radio(["cnn", "pixel", "both"], value="cnn", label="Signal",
                                                  info="cnn: same character · pixel: same strokes")
                                scope = gr.Radio(["all", "other classes", "same class"], value="all", label="Search in")
                            with gr.Row():
                                k_nb = gr.Slider(8, 60, value=24, step=4, label="How many")
                                auto_sim = gr.Checkbox(True, label="Search on open")
                                b_sim = gr.Button("Find similar", size="sm")
                            vote = gr.Markdown()
                            neigh = gr.Gallery(columns=6, height="auto", allow_preview=False, show_label=False,
                                               elem_id="neigh", object_fit="contain", interactive=False)
                            gr.Markdown("*Click a neighbour to open it.*")

            # ======================= statistics ====================================================
            with gr.Tab("Statistics", id="stats") as stats_tab:
                with gr.Row():
                    refresh = gr.Button("Refresh", size="sm", scale=0)
                    gr.Markdown("*Updates every 30 s.*")
                tiles = gr.HTML()
                bar = gr.BarPlot(x="class", y="pct_reviewed", color="status", y_title="% reviewed",
                                 x_label_angle=-60, height=320, y_lim=[0, 100],
                                 color_map={"not_started": "#9aa0a6", "in_progress": "#3b82f6",
                                            "needs_review": "#e68c14", "completed": "#2ea043"})
                class_table = gr.Dataframe(label="Per class", interactive=False, show_search="filter",
                                           max_height=520)
                with gr.Row():
                    reassign_table = gr.Dataframe(label="Proposed reassignments (team consensus): folder → class",
                                                  interactive=False)
                    reviewer_table = gr.Dataframe(label="Reviewers (agreement = share of their verdicts on "
                                                  "images others also judged that don't conflict)", interactive=False)
                with gr.Row():
                    dl_log = gr.DownloadButton("Download decision log (CSV)", size="sm")
                    dl_classes = gr.DownloadButton("Download per-class stats (CSV)", size="sm")

            # ======================= conflicts & export ============================================
            with gr.Tab("Conflicts & export", id="export"):
                gr.Markdown("### Conflicts\nReviewers gave different verdicts. Click a row to open it in the Review tab.")
                conflict_table = gr.Dataframe(interactive=False, max_height=360)
                conflict_refresh = gr.Button("Refresh conflicts", size="sm")
                gr.Markdown(
                    "### Export to the pipeline\n"
                    "Writes consensus **wrong class → `reassign`** and **drop → `drop`** rows to "
                    "`configs/image_rulings.csv` (source `audit_app`), expanded to identical copies, plus the full "
                    "log to `configs/review_log.csv` and `configs/review_class_status.csv`. OK / unsure / flagged / "
                    "conflicting images get no row, so their folder label stands. Rows from other tools are kept; "
                    "a path they already ruled is skipped and listed. Re-run `03_split` afterwards.")
                with gr.Row():
                    min_agree = gr.Radio([("≥ 2 reviewers agree", 2), ("1 reviewer is enough", 1)], value=2,
                                         label="A decision becomes a ruling when")
                    preview_btn = gr.Button("Preview", size="sm")
                    export_btn = gr.Button("Export", variant="primary", size="sm")
                export_md = gr.Markdown()
                preview_table = gr.Dataframe(interactive=False, max_height=360)

        # ---------------- callbacks ----------------------------------------------------------------
        page_inputs = [folder, order, filt, page, size, true_size, name]
        page_outputs = [gallery, page_items, page, page_info, header]
        detail_outputs = [cur_item, big, info, model_md, decisions, proposed, cmp, neigh, neigh_rows, vote]
        sim_inputs = [signal, scope, k_nb, true_size, auto_sim]

        def status_of(f):
            s = STORE.class_status().set_index("class_folder")
            if int(f) in s.index:
                return s.loc[int(f), "status"], s.loc[int(f), "note"]
            return "not_started", ""

        def new_class(f, o, fl, _page, sz, ts, nm, request: gr.Request):
            return (*load_page(f, o, fl, 0, sz, ts, nm, request), *status_of(f))

        for ev in (folder.change, order.change, filt.change, size.release, true_size.change):
            ev(new_class, page_inputs, [*page_outputs, status, status_note])
        demo.load(lambda s: s, name_saved, name).then(new_class, page_inputs, [*page_outputs, status, status_note])
        name.change(lambda n: n, name, name_saved)

        def turn(delta):
            def f(fo, o, fl, p, sz, ts, nm, request: gr.Request):
                return load_page(fo, o, fl, p + delta, sz, ts, nm, request)
            return f
        prev_page.click(turn(-1), page_inputs, page_outputs)
        next_page.click(turn(+1), page_inputs, page_outputs)

        def page_all_ok(fo, o, fl, p, sz, ts, nm, items, request: gr.Request):
            me = reviewer(nm, request)
            paths = ITEMS.path.iloc[items].tolist()
            decided = set(STORE.current(paths=paths).query("reviewer == @me").path)
            rows = [dict(path=pa, folder_class=int(ITEMS.folder.iloc[k]), verdict="ok")
                    for k, pa in zip(items, paths, strict=True) if pa not in decided]
            n = STORE.record_many(rows, me) if rows else 0
            gr.Info(f"Marked {n} images OK")
            # a "not reviewed" filter has already dropped this page out of the list
            nxt = p if fl in ("not reviewed by me", "not reviewed by anyone") else p + 1
            return load_page(fo, o, fl, nxt, sz, ts, nm, request)
        page_ok.click(page_all_ok, [*page_inputs, page_items], page_outputs)

        def on_gallery(evt: gr.SelectData, md, items, nm, ts, sig, sc, n, ts2, auto, request: gr.Request):
            k = items[evt.index]
            if md.startswith("Scan"):
                me = reviewer(nm, request)
                r = ITEMS.iloc[k]
                mine = STORE.current(paths=[r.path]).query("reviewer == @me")
                is_flag = len(mine) and mine.verdict.iloc[0] == "flag"
                STORE.record(r.path, r.folder, me, "clear" if is_flag else "flag")
                return render_gallery(items, me, ts), *([gr.skip()] * len(detail_outputs))
            return gr.skip(), *open_item(k, nm, sig, sc, n, ts2, auto, request)
        # every click counts while scanning fast: queue them rather than drop them, and don't
        # cover the grid with a progress overlay that swallows the next click
        gallery.select(on_gallery, [mode, page_items, name, true_size, *sim_inputs], [gallery, *detail_outputs],
                       trigger_mode="multiple", show_progress="hidden")

        def act(verdict):
            def f(k, items, nm, prop, nt, adv, ts, sig, sc, n, ts2, auto, request: gr.Request):
                me = reviewer(nm, request)
                if k is None:
                    raise gr.Error("Open an image first (Inspect mode, then click it).")
                r = ITEMS.iloc[int(k)]
                if verdict == "wrong_class":
                    if prop is None:
                        raise gr.Error("Choose the correct class first.")
                    if int(prop) == int(r.folder):
                        raise gr.Error("That is the folder's own class — use OK instead.")
                STORE.record(r.path, r.folder, me, verdict,
                             int(prop) if verdict == "wrong_class" else None, nt if verdict != "clear" else "")
                nxt = k
                if adv and verdict != "clear" and k in items and items.index(k) + 1 < len(items):
                    nxt = items[items.index(k) + 1]
                return (render_gallery(items, me, ts), "", *open_item(nxt, nm, sig, sc, n, ts2, auto, request))
            return f
        act_inputs = [cur_item, page_items, name, proposed, note, advance, true_size, *sim_inputs]
        for btn, v in ((b_ok, "ok"), (b_wrong, "wrong_class"), (b_drop, "drop"), (b_unsure, "unsure"),
                       (b_flag, "flag"), (b_clear, "clear")):
            btn.click(act(v), act_inputs, [gallery, note, *detail_outputs])

        def step(delta):
            def f(k, items, nm, sig, sc, n, ts, auto, request: gr.Request):
                if not items:
                    return (gr.skip(),) * len(detail_outputs)
                i = items.index(k) + delta if k in items else 0
                return open_item(items[min(max(i, 0), len(items) - 1)], nm, sig, sc, n, ts, auto, request)
            return f
        b_prev.click(step(-1), [cur_item, page_items, name, *sim_inputs], detail_outputs)
        b_next.click(step(+1), [cur_item, page_items, name, *sim_inputs], detail_outputs)

        b_sim.click(neighbours, [cur_item, signal, scope, k_nb, true_size], [neigh, neigh_rows, vote])
        for c in (signal, scope):
            c.change(neighbours, [cur_item, signal, scope, k_nb, true_size], [neigh, neigh_rows, vote])
        k_nb.release(neighbours, [cur_item, signal, scope, k_nb, true_size], [neigh, neigh_rows, vote])
        proposed.input(compare, [cur_item, proposed], cmp)

        def on_neigh(evt: gr.SelectData, rows, nm, sig, sc, n, ts, auto, request: gr.Request):
            return open_item(int(CAT.item_of_row[rows[evt.index]]), nm, sig, sc, n, ts, auto, request)
        neigh.select(on_neigh, [neigh_rows, name, *sim_inputs], detail_outputs)

        def save_status(f, st, nt, nm, request: gr.Request):
            if st is None:
                raise gr.Error("Pick a status.")
            STORE.set_class_status(f, st, reviewer(nm, request), nt)
            gr.Info(f"{lab(f)} → {st}")
            return class_header(f)
        status_save.click(save_status, [folder, status, status_note, name], header)

        # ---- statistics
        def stats_view():
            cons = consensus(STORE.current())
            cs = stats.class_stats(CAT, STORE, cons)
            n_items, reviewed = int(cs.n_unique.sum()), int(cs.reviewed.sum())
            by_status = cs.status.value_counts()

            def tile(v, lbl):
                return f"<div class='tile'><div class='v'>{v}</div><div class='l'>{lbl}</div></div>"
            html = "<div class='tiles'>" + "".join([
                tile(f"{int(cs.n_raw.sum()):,}", "image files"),
                tile(f"{n_items:,}", "unique images to review"),
                tile(f"{reviewed:,}", "reviewed (≥ 1 verdict)"),
                tile(f"{100 * reviewed / max(1, n_items):.1f}%", "complete"),
                tile(f"{int(cs.flag.sum()):,}", "⚑ flagged, unresolved"),
                tile(f"{int(cs.wrong_class.sum()):,}", "→ wrong class"),
                tile(f"{int(cs["drop"].sum()):,}", "✗ drop"),
                tile(f"{int(cs.conflict.sum()):,}", "⚠ conflicts"),
                tile(f"{by_status.get('completed', 0)} / {len(cs)}", "classes completed"),
                tile(f"{by_status.get('needs_review', 0)}", "classes need review"),
                tile(f"{by_status.get('in_progress', 0)}", "classes in progress"),
            ]) + "</div>"
            plot = cs.assign(**{"class": cs.class_folder.map(lab)})[["class", "pct_reviewed", "status"]]
            log_f, cls_f = DOWNLOADS / "review_log.csv", DOWNLOADS / "review_class_stats.csv"
            STORE.log().to_csv(log_f, index=False)
            cs.to_csv(cls_f, index=False)
            return (html, plot, cs, stats.reassign_matrix(CAT, cons), stats.reviewer_stats(STORE),
                    str(log_f), str(cls_f))
        stats_outputs = [tiles, bar, class_table, reassign_table, reviewer_table, dl_log, dl_classes]
        refresh.click(stats_view, None, stats_outputs)
        stats_tab.select(stats_view, None, stats_outputs)
        gr.Timer(30).tick(stats_view, None, stats_outputs)

        # ---- conflicts & export
        def conflict_view():
            return stats.conflicts(CAT, consensus(STORE.current()))
        conflict_refresh.click(conflict_view, None, conflict_table)
        demo.load(conflict_view, None, conflict_table)

        def on_conflict(evt: gr.SelectData, table, nm, sig, sc, n, ts, auto, request: gr.Request):
            k = int(table.iloc[evt.index[0]]["item"])
            return gr.Tabs(selected="review"), *open_item(k, nm, sig, sc, n, ts, auto, request)
        conflict_table.select(on_conflict, [conflict_table, name, *sim_inputs], [tabs, *detail_outputs])

        def preview(ma):
            rows = export.proposed_rows(CAT, STORE, int(ma))
            return (f"**{len(rows)}** rows would be written ({(rows.ruling == 'reassign').sum()} reassign, "
                    f"{(rows.ruling == 'drop').sum()} drop)."), rows

        def do_export(ma, nm, request: gr.Request):
            who = reviewer(nm, request)
            rep = export.export(CAT, STORE, CONFIGS, int(ma))
            sk = rep["skipped_already_ruled"]
            return (f"Exported by **{who}**: {rep['written']} rows ({rep['reassign']} reassign, {rep['drop']} drop); "
                    f"kept {rep['kept_other_sources']} rows from other tools."
                    + (f"  \nSkipped {len(sk)} paths another tool already ruled: " + ", ".join(f"`{p}`" for p in sk[:20])
                       if sk else ""))
        preview_btn.click(preview, min_agree, [export_md, preview_table])
        export_btn.click(do_export, [min_agree, name], export_md)
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="also serve through a public *.gradio.live tunnel (72 h link)")
    a = ap.parse_args()
    if a.share and not USERS:
        print("warning: --share without AUDIT_USERS -- anyone with the link can read and write reviews")
    demo = build()
    demo.queue(default_concurrency_limit=16).launch(
        server_name=a.host, server_port=a.port, share=a.share, auth=list(USERS.items()) or None,
        allowed_paths=[str(CAT.cache / "review_thumbs"), str(DOWNLOADS)],
        head=KEYS_JS, css=CSS, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
