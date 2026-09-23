# Visual check — the human half of the dataset audit.
#
#   uv run marimo edit notebooks/02_visual_check.py
#
# Reads the manifest written by 01_dataset_audit.ipynb. Writes only to configs/.

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visual check

    The audit notebook measured everything that can be measured. What is left needs eyes:

    1. **What character is each class?** — the manifest has bare folder IDs by design.
    2. **How near is "the same image"?** — §10 of the audit computed two near-duplicate
       signals exhaustively and deliberately chose no threshold.
    3. **Which label wins** when one image carries two?
    4. **Is an outlier actually wrong**, or merely unusual?

    Exact (byte-identical) duplicates were already settled arithmetically, so this notebook
    shows one canonical copy of each and you never have to think about them.

    > This notebook writes **only** to `configs/`. It never touches `raw/` or the manifest.
    """)
    return


@app.cell
def _():
    import io
    from datetime import UTC, datetime
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageDraw

    _here = Path(__file__).resolve().parent
    ROOT = _here.parent
    DATASET = "baseline"
    RAW = ROOT / "data" / DATASET / "raw"
    MANIFEST = ROOT / "data" / DATASET / "manifest"
    CACHE = ROOT / "data" / DATASET / "cache"
    CONFIGS = ROOT / "configs"
    CONFIGS.mkdir(exist_ok=True)
    return (
        CACHE,
        CONFIGS,
        Image,
        ImageDraw,
        MANIFEST,
        RAW,
        ROOT,
        UTC,
        datetime,
        io,
        mo,
        np,
        pd,
    )


@app.cell
def _(MANIFEST, pd):
    images = pd.read_csv(MANIFEST / "images.csv", keep_default_na=False, na_values=[""])
    classes = pd.read_csv(MANIFEST / "classes.csv")
    cross = pd.read_csv(MANIFEST / "findings_cross_class_exact.csv")
    cross_near = pd.read_csv(MANIFEST / "findings_cross_class_near.csv")
    outliers = pd.read_csv(MANIFEST / "findings_outliers.csv")
    # Pair tables are regenerable and not tracked in git, so they may be absent on a fresh
    # clone. Fall back to empty frames -- section 4 then explains itself instead of crashing.
    def _load_pairs(name, cols):
        f = MANIFEST / "pairs" / f"{name}.csv.gz"
        return pd.read_csv(f) if f.exists() else pd.DataFrame({c: [] for c in cols})

    pairs_phash = _load_pairs("pairs_phash", ["i", "j", "hamming"])
    pairs_pixel = _load_pairs("pairs_pixel", ["i", "j", "rms"])
    pairs_available = (MANIFEST / "pairs" / "pairs_pixel.csv.gz").exists()

    canonical = images[images.is_sha_canonical].reset_index(drop=True)
    class_ids = sorted(classes.class_folder.tolist())
    return (
        canonical,
        class_ids,
        classes,
        cross,
        cross_near,
        images,
        outliers,
        pairs_available,
        pairs_phash,
        pairs_pixel,
    )


@app.cell(hide_code=True)
def _(canonical, classes, cross, cross_near, images, mo, outliers):
    mo.hstack(
        [
            mo.stat(f"{len(images):,}", label="images"),
            mo.stat(f"{len(canonical):,}", label="canonical (exact dups collapsed)"),
            mo.stat(f"{len(classes)}", label="classes"),
            mo.stat(f"{cross.sha_group_id.nunique()}", label="cross-class dup groups"),
            mo.stat(f"{cross_near.near_dup_group_id.nunique()}", label="cross-class near groups"),
            mo.stat(f"{len(outliers):,}", label="outliers queued"),
        ],
        justify="start",
        gap=2,
    )
    return


@app.cell
def _(Image, ImageDraw, RAW, ROOT, io, np):
    from PIL import ImageFont

    # PIL's default bitmap font has no Thai glyphs (renders tofu boxes), so captions with
    # Thai characters need a bundled font that does.
    _FONT_PATH = ROOT / "assets" / "fonts" / "sarabun" / "THSarabun.ttf"
    _font_cache = {}

    def _font(size):
        if size not in _font_cache:
            _font_cache[size] = ImageFont.truetype(str(_FONT_PATH), size)
        return _font_cache[size]

    def load_gray(rel_path):
        with Image.open(RAW / rel_path) as im:
            return np.asarray(im.convert("L"), np.uint8)

    def fit(arr, cell, pad=4):
        # scale a glyph into a square cell, preserving aspect, padded white.
        h, w = arr.shape
        inner = cell - 2 * pad
        s = min(inner / w, inner / h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        canvas = Image.new("L", (cell, cell), 255)
        canvas.paste(Image.fromarray(arr).resize((nw, nh), Image.Resampling.LANCZOS),
                     ((cell - nw) // 2, (cell - nh) // 2))
        return canvas

    def montage(items, ncol=12, cell=72, label_h=22, scale=1):
        # items: list of (2-D uint8 array, caption)
        cell, label_h = cell * scale, label_h * scale
        if not items:
            return Image.new("L", (cell, cell), 255)
        nrow = int(np.ceil(len(items) / ncol))
        sheet = Image.new("L", (ncol * cell, nrow * (cell + label_h)), 255)
        draw = ImageDraw.Draw(sheet)
        _cap_font = _font(max(8, label_h - 4)) if label_h > 6 else None
        for k, (arr, cap) in enumerate(items):
            r, c = divmod(k, ncol)
            y = r * (cell + label_h)
            sheet.paste(fit(arr, cell), (c * cell, y))
            if cap:
                draw.text((c * cell + 2 * scale, y + cell + scale), str(cap), fill=0, font=_cap_font)
        return sheet

    def png(img, scale=1):
        if scale != 1:
            img = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
        buf = io.BytesIO()
        img.convert("L").save(buf, "PNG")
        return buf.getvalue()

    return fit, load_gray, montage, png


@app.cell
def _(CACHE, canonical, class_ids, fit, load_gray, np):
    # Mean-image prototype per class. Averaging cancels stroke noise and shows the canonical glyph.
    # Cached: recomputing loads ~10k images and would stall every reactive re-run.
    # Samples are letterboxed via fit() (scale-to-fit, aspect preserved) before averaging --
    # a plain square stretch would flatten exactly the aspect-ratio cues that separate classes
    # like า/ๅ or ฤ/ป's tails from the rest, which is the wrong thing to average away in the
    # image billed as "primary evidence" for §3.
    PROTO_N, PROTO_S = 150, 48
    # parameters (and the transform) are in the filename so changing them cannot silently
    # reuse a stale cache built under an older transform
    _cache = CACHE / f"prototypes_n{PROTO_N}_s{PROTO_S}_fitwhite.npz"

    if _cache.exists():
        _z = np.load(_cache)
        prototypes = {int(k): _z[k] for k in _z.files}
    else:
        prototypes = {}
        for _fid in class_ids:
            _paths = sorted(canonical.loc[canonical.class_folder == _fid, "path"])
            _rng = np.random.default_rng(42 + _fid)
            if len(_paths) > PROTO_N:
                _paths = [_paths[i] for i in _rng.permutation(len(_paths))[:PROTO_N]]
            _acc = np.zeros((PROTO_S, PROTO_S), np.float64)
            for _p in _paths:
                _acc += np.asarray(fit(load_gray(_p), PROTO_S, pad=2), np.float64)
            _a = _acc / max(1, len(_paths))
            _a = (_a - _a.min()) / max(1e-6, _a.max() - _a.min()) * 255
            prototypes[_fid] = _a.astype(np.uint8)
        np.savez_compressed(_cache, **{str(k): v for k, v in prototypes.items()})
    return (prototypes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Every class at a glance

    Mean image per class, contrast-stretched. Averaging up to 150 samples cancels individual
    handwriting and leaves the canonical glyph — this is the primary evidence for §3.
    """)
    return


@app.cell(hide_code=True)
def _(class_char, class_ids, classes, mo, montage, png, prototypes):
    _n = dict(zip(classes.class_folder, classes.n_images_dedup, strict=True))
    def _cap(f):
        ch = class_char.get(f, "")
        return f"{f} {ch} n={_n[f]}" if ch else f"{f} n={_n[f]}"
    _items = [(prototypes[f], _cap(f)) for f in class_ids]
    mo.image(png(montage(_items, ncol=12, cell=80, scale=2)), width="100%",
             caption="Mean-image prototype per class (sha-deduped, seed 42)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Class browser

    A prototype hides within-class variation; the random grid shows it. Look here for mislabels
    and for classes that are visually two different things.
    """)
    return


@app.cell
def _(class_char, class_ids, mo):
    def _opt(f):
        ch = class_char.get(f, "")
        return f"{f} {ch}" if ch else str(f)
    class_pick = mo.ui.dropdown(
        options={_opt(f): f for f in class_ids}, value=_opt(class_ids[0]), label="class folder")
    grid_n = mo.ui.slider(8, 96, value=36, step=4, label="samples shown")
    mo.hstack([class_pick, grid_n], justify="start", gap=2)
    return class_pick, grid_n


@app.cell(hide_code=True)
def _(
    Image,
    canonical,
    class_char,
    class_pick,
    classes,
    grid_n,
    load_gray,
    mo,
    montage,
    np,
    png,
    prototypes,
):
    _fid = class_pick.value
    _ch = class_char.get(_fid, "")
    _paths = sorted(canonical.loc[canonical.class_folder == _fid, "path"])
    _rng = np.random.default_rng(42 + _fid)
    _sel = [_paths[i] for i in _rng.permutation(len(_paths))[: grid_n.value]]
    _row = classes[classes.class_folder == _fid].iloc[0]

    mo.vstack([
        mo.hstack([
            mo.image(png(Image.fromarray(prototypes[_fid]), scale=3),
                     caption=f"prototype · class {_fid}" + (f" · {_ch}" if _ch else "")),
            mo.md(
                f"""
                **class {_fid}**{f" — {_ch}" if _ch else ""} — {int(_row.n_images_dedup):,} canonical images
                ({int(_row.n_images_raw):,} raw, {int(_row.n_redundant_copies):,} redundant)

                size: {int(_row.width_min)}–{int(_row.width_max)} × {int(_row.height_min)}–{int(_row.height_max)} px ·
                brightness {_row.brightness_mean:.0f} · contrast {_row.contrast_mean:.0f}

                cross-class duplicate images: **{int(_row.n_cross_class_dup)}**
                """
            ),
        ], justify="start", gap=2),
        mo.image(png(montage([(load_gray(p), "") for p in _sel], ncol=12, cell=64, label_h=2)),
                 width="100%", caption=f"{len(_sel)} random samples, seed 42"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Record the class labels

    The table below is **seeded with a hypothesis, not an answer**: `class_folder` appears to equal
    the decimal **TIS-620** byte value of the Thai character (equivalently, `class_folder + 0xD60`
    is its Unicode code point). That covers the consonants `161–206` (including ฤ/ฦ), the vowel and
    tone-mark block `207–239`, and the digit block `240–249` (`๐–๙`).

    That is a pattern in the numbering, not evidence about the pixels. Check each row against
    its prototype above before accepting it. Clear the `character` cell to mark a class
    undecided — an empty cell means *unknown*, and unknown is a legitimate answer.

    Edit, then press **Save**. Writes `configs/class_labels.csv`.
    """)
    return


@app.cell
def _():
    # Canonical Thai reference data (domain knowledge, not a claim about this dataset).
    CONSONANTS = [
        ("ก", "ko kai"), ("ข", "kho khai"), ("ฃ", "kho khuat"), ("ค", "kho khwai"),
        ("ฅ", "kho khon"), ("ฆ", "kho rakhang"), ("ง", "ngo ngu"), ("จ", "cho chan"),
        ("ฉ", "cho ching"), ("ช", "cho chang"), ("ซ", "so so"), ("ฌ", "cho choe"),
        ("ญ", "yo ying"), ("ฎ", "do chada"), ("ฏ", "to patak"), ("ฐ", "tho than"),
        ("ฑ", "tho nangmontho"), ("ฒ", "tho phuthao"), ("ณ", "no nen"), ("ด", "do dek"),
        ("ต", "to tao"), ("ถ", "tho thung"), ("ท", "tho thahan"), ("ธ", "tho thong"),
        ("น", "no nu"), ("บ", "bo baimai"), ("ป", "po pla"), ("ผ", "pho phung"),
        ("ฝ", "fo fa"), ("พ", "pho phan"), ("ฟ", "fo fan"), ("ภ", "pho samphao"),
        ("ม", "mo ma"), ("ย", "yo yak"), ("ร", "ro rua"),
        ("ฤ", "ru"), ("ล", "lo ling"), ("ฦ", "rue"),
        ("ว", "wo waen"), ("ศ", "so sala"), ("ษ", "so rusi"), ("ส", "so sua"),
        ("ห", "ho hip"), ("ฬ", "lo chula"), ("อ", "o ang"), ("ฮ", "ho nokhuk"),
    ]
    # Vowels, tone marks and other signs that fill the TIS-620 code page between the
    # consonant block (ending 0xCE/206) and the digit block (starting 0xF0/240).
    # None marks the four TIS-620 positions (0xDB-0xDE / 219-222) left unassigned.
    OTHER_SIGNS = [
        ("ฯ", "paiyannoi", "Thai Punctuation"),
        ("ะ", "sara a", "Thai Vowel"),
        ("ั", "mai han-akat", "Thai Vowel"),
        ("า", "sara aa", "Thai Vowel"),
        ("ำ", "sara am", "Thai Vowel"),
        ("ิ", "sara i", "Thai Vowel"),
        ("ี", "sara ii", "Thai Vowel"),
        ("ึ", "sara ue", "Thai Vowel"),
        ("ื", "sara uee", "Thai Vowel"),
        ("ุ", "sara u", "Thai Vowel"),
        ("ู", "sara uu", "Thai Vowel"),
        ("ฺ", "phinthu", "Thai Diacritic"),
        None, None, None, None,
        ("฿", "baht sign", "Thai Symbol"),
        ("เ", "sara e", "Thai Vowel"),
        ("แ", "sara ae", "Thai Vowel"),
        ("โ", "sara o", "Thai Vowel"),
        ("ใ", "sara ai maimuan", "Thai Vowel"),
        ("ไ", "sara ai maimalai", "Thai Vowel"),
        ("ๅ", "lakkhangyao", "Thai Vowel"),
        ("ๆ", "maiyamok", "Thai Punctuation"),
        ("็", "maitaikhu", "Thai Diacritic"),
        ("่", "mai ek", "Thai Tone Mark"),
        ("้", "mai tho", "Thai Tone Mark"),
        ("๊", "mai tri", "Thai Tone Mark"),
        ("๋", "mai chattawa", "Thai Tone Mark"),
        ("์", "thanthakhat", "Thai Diacritic"),
        ("ํ", "nikhahit", "Thai Diacritic"),
        ("๎", "yamakkan", "Thai Diacritic"),
        ("๏", "fongman", "Thai Punctuation"),
    ]
    DIGITS = [("๐", "sun"), ("๑", "nueng"), ("๒", "song"), ("๓", "sam"), ("๔", "si"),
              ("๕", "ha"), ("๖", "hok"), ("๗", "chet"), ("๘", "paet"), ("๙", "kao")]
    return CONSONANTS, DIGITS, OTHER_SIGNS


@app.cell
def _(CONFIGS, CONSONANTS, DIGITS, OTHER_SIGNS, class_ids, classes, pd):
    def seed_row(fid):
        # the TIS-620 hypothesis: class_folder appears to equal the decimal TIS-620
        # byte value of the character. A suggestion to check, never an assertion.
        if 161 <= fid <= 160 + len(CONSONANTS):
            ch, name = CONSONANTS[fid - 161]
            return ch, "Thai Consonant", f"hypothesis: TIS-620 0x{fid:02X} -> {name}"
        if 207 <= fid <= 206 + len(OTHER_SIGNS):
            entry = OTHER_SIGNS[fid - 207]
            if entry is None:
                return "", "", f"no hypothesis: TIS-620 0x{fid:02X} is unassigned"
            ch, name, cat = entry
            return ch, cat, f"hypothesis: TIS-620 0x{fid:02X} -> {name}"
        if 240 <= fid <= 249:
            ch, name = DIGITS[fid - 240]
            return ch, "Thai Digit", f"hypothesis: TIS-620 0x{fid:02X} -> {name}"
        return "", "", "no hypothesis: outside the TIS-620 Thai block"

    _existing = (pd.read_csv(CONFIGS / "class_labels.csv", keep_default_na=False)
                 if (CONFIGS / "class_labels.csv").exists() else None)
    _n = dict(zip(classes.class_folder, classes.n_images_dedup, strict=True))

    _rows = []
    for _f in class_ids:
        _ch, _cat, _note = seed_row(_f)
        if _existing is not None and _f in set(_existing.class_folder):
            _p = _existing[_existing.class_folder == _f].iloc[0]
            _ch, _cat, _note = _p.character, _p.category, _p.note
            _conf = _p.confidence
        else:
            _conf = ""
        _rows.append(dict(class_folder=_f, n_images=_n[_f], character=_ch,
                          category=_cat, confidence=_conf, note=_note))
    label_seed = pd.DataFrame(_rows)
    return (label_seed,)


@app.cell
def _(label_seed):
    # Best-known character per class, for annotating ids elsewhere in this notebook: the
    # saved decision if one exists, otherwise the seeded TIS-620 hypothesis. Not proof —
    # still check it against the prototype in §1/§2 before trusting a caption.
    class_char = dict(zip(label_seed.class_folder, label_seed.character.fillna(""), strict=True))
    return (class_char,)


@app.cell
def _(label_seed, mo):
    label_editor = mo.ui.data_editor(label_seed, label="class labels",
                                     editable_columns=["character", "category", "confidence", "note"])
    label_editor
    return (label_editor,)


@app.cell
def _(mo):
    save_labels = mo.ui.button(label="Save class labels", value=0, on_click=lambda v: v + 1)
    save_labels
    return (save_labels,)


@app.cell(hide_code=True)
def _(CONFIGS, UTC, datetime, label_editor, mo, pd, save_labels):
    if save_labels.value:
        _df = pd.DataFrame(label_editor.value).copy()
        _df["decided_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        _df.to_csv(CONFIGS / "class_labels.csv", index=False)
        _decided = int((_df.character.astype(str).str.strip() != "").sum())
        _out = mo.md(f"Saved **{len(_df)}** rows to `configs/class_labels.csv` — "
                     f"{_decided} with a character, {len(_df) - _decided} left undecided.")
    else:
        _out = mo.md("*Not saved yet.*")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Near-duplicate threshold

    The audit computed both signals over all 1.97 billion pairs and chose no cut. Choose it here,
    by looking.

    - **pHash Hamming** — the prior pipeline's signal. Note it only moves in steps of 2: pHash
      sets bits against the median of 64 coefficients, so every hash has exactly 32 bits set and
      all pairwise distances are even. A threshold of 1 accepts exactly what 0 accepts.
    - **Pixel RMS** — mean gray-level difference on a common 16×16 grid. RMS 4 means the two
      images differ by 4 gray levels per pixel on average.

    The pair that matters is the one where the two signals **disagree**. Look at those.
    """)
    return


@app.cell
def _(mo, pairs_available, pairs_phash, pairs_pixel):
    pairs = pairs_phash.merge(pairs_pixel, on=["i", "j"], how="outer")
    missing_pairs = None if pairs_available else mo.callout(
        mo.md(
            """
            **The near-duplicate pair tables are not on disk.**

            They are ~14 MB of regenerable data, so they are not tracked in git. Run
            `notebooks/01_dataset_audit.ipynb` once (about 75 seconds) and this section will
            populate. Everything else on this page works without them.
            """
        ),
        kind="warn",
    )
    missing_pairs
    return (pairs,)


@app.cell
def _(mo):
    t_phash = mo.ui.slider(0, 4, value=0, step=2, label="pHash Hamming ≤")
    t_pixel = mo.ui.slider(0.0, 32.0, value=4.0, step=0.5, label="pixel RMS ≤")
    pair_view = mo.ui.radio(
        options=["both agree", "only pHash", "only pixel"], value="only pHash",
        label="show pairs where", inline=True)
    mo.vstack([mo.hstack([t_phash, t_pixel], justify="start", gap=2), pair_view])
    return pair_view, t_phash, t_pixel


@app.cell(hide_code=True)
def _(mo, pair_view, pairs, t_phash, t_pixel):
    _ph = pairs.hamming.le(t_phash.value).fillna(False)
    _px = pairs.rms.le(t_pixel.value).fillna(False)
    sel = {"both agree": _ph & _px, "only pHash": _ph & ~_px, "only pixel": _px & ~_ph}[pair_view.value]
    shown = pairs[sel].sort_values("rms", na_position="last").reset_index(drop=True)

    mo.vstack([
        mo.hstack([
            mo.stat(f"{int((_ph & _px).sum()):,}", label="both agree"),
            mo.stat(f"{int((_ph & ~_px).sum()):,}", label="only pHash"),
            mo.stat(f"{int((_px & ~_ph).sum()):,}", label="only pixel"),
            mo.stat(f"{int((_ph | _px).sum()):,}", label="union"),
        ], justify="start", gap=2),
        mo.md(f"**{len(shown):,}** pairs in the selected set."),
    ])
    return (shown,)


@app.cell
def _(mo, shown):
    pair_page = mo.ui.slider(0, max(1, (len(shown) - 1) // 8), value=0, step=1,
                             label=f"page of 8 (of {max(1, -(-len(shown)//8))})")
    pair_page
    return (pair_page,)


@app.cell(hide_code=True)
def _(Image, class_char, images, load_gray, mo, np, pair_page, png, shown):
    def pair_strip(row):
        a, b = images.iloc[int(row.i)], images.iloc[int(row.j)]
        ga, gb = load_gray(a.path), load_gray(b.path)
        h = max(ga.shape[0], gb.shape[0], 24)

        def _pad(g):
            out = np.full((h, g.shape[1]), 255, np.uint8)
            out[: g.shape[0]] = g
            return out

        gap = np.full((h, 6), 128, np.uint8)
        strip = Image.fromarray(np.hstack([_pad(ga), gap, _pad(gb)]))
        hd = "-" if np.isnan(row.hamming) else f"{int(row.hamming)}"
        rd = "-" if np.isnan(row.rms) else f"{row.rms:.2f}"
        _ca, _cb = class_char.get(a.class_folder, ""), class_char.get(b.class_folder, "")
        return mo.vstack([
            mo.image(png(strip, scale=4), width=320),
            mo.md(f"`{a.class_folder}` {_ca} vs `{b.class_folder}` {_cb} · "
                  f"hamming **{hd}** · rms **{rd}**"),
        ])

    _page = shown.iloc[pair_page.value * 8 : pair_page.value * 8 + 8]
    _cards = [pair_strip(r) for r in _page.itertuples()]
    _rows = [mo.hstack(_cards[i : i + 2], justify="start", gap=2) for i in range(0, len(_cards), 2)]
    mo.vstack(_rows) if len(_page) else mo.md("*No pairs.*")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Decision: RMS is the near-duplicate signal, not pHash.** These are isolated single-character
    > glyphs — simple enough that two different writers' strokes can coincidentally land close in
    > RMS by chance, and pHash's distance is coarse (quantized in steps of 2) and its DCT resize can
    > collapse distinct simple glyphs together. pHash is a candidate filter only; RMS makes the call.
    >
    > **Threshold: RMS ≤ 4**, matching the empirical gap from §10 of the audit — true duplicates
    > (the same sample re-exported, resized, or recompressed) cluster tightly under this value, then
    > there is a gap before genuinely different samples that merely look similar.
    >
    > **Group, don't delete.** A redundant sample barely moves a CNN's training signal, so
    > thinning near-dups out of the dataset buys little. The actual risk is a near-duplicate pair
    > landing on both sides of a train/val/test split, which inflates the validation/test metric with
    > memorization instead of generalization. So treat each RMS ≤ 4 group as a unit that must stay
    > entirely within one split — keep every image, just keep its duplicates together.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Cross-class exact duplicates

    Byte-identical images sitting in two different class folders. One of the two labels is
    wrong, and no amount of arithmetic can say which — the pixels are the same pixels.

    Record a ruling per group: which class folder is correct, or `hold` if the glyph is genuinely
    ambiguous. Writes `configs/cross_class_rulings.csv`.
    """)
    return


@app.cell
def _(cross, mo):
    xgroups = sorted(cross.sha_group_id.unique())
    xpick = mo.ui.dropdown(options={str(g): g for g in xgroups}, value=str(xgroups[0]),
                           label="duplicate group")
    xpick
    return xgroups, xpick


@app.cell(hide_code=True)
def _(class_char, cross, load_gray, mo, montage, png, xpick):
    _g = cross[cross.sha_group_id == xpick.value]
    def _cap(f):
        ch = class_char.get(f, "")
        return f"cls {f} {ch}" if ch else f"cls {f}"
    _table = _g[["path", "class_folder"]].assign(character=_g.class_folder.map(class_char))
    mo.vstack([
        mo.image(png(montage([(load_gray(r.path), _cap(r.class_folder)) for r in _g.itertuples()],
                             ncol=6, cell=96, scale=2)),
                 caption=f"group {xpick.value} · classes {_g.class_folder.unique().tolist()}"),
        mo.ui.table(_table, selection=None),
    ])
    return


@app.cell
def _(CONFIGS, cross, pd, xgroups):
    _existing = (pd.read_csv(CONFIGS / "cross_class_rulings.csv", keep_default_na=False)
                 if (CONFIGS / "cross_class_rulings.csv").exists() else None)
    _rows = []
    for _g in xgroups:
        _sub = cross[cross.sha_group_id == _g]
        _prev = (_existing[_existing.sha_group_id == _g].iloc[0]
                 if _existing is not None and _g in set(_existing.sha_group_id) else None)
        _rows.append(dict(
            sha_group_id=_g,
            classes=",".join(map(str, sorted(_sub.class_folder.unique()))),
            n_images=len(_sub),
            correct_class=("" if _prev is None else _prev.correct_class),
            ruling=("" if _prev is None else _prev.ruling),
            note=("" if _prev is None else _prev.note)))
    ruling_seed = pd.DataFrame(_rows)
    return (ruling_seed,)


@app.cell
def _(mo, ruling_seed):
    ruling_editor = mo.ui.data_editor(ruling_seed, label="cross-class rulings",
                                      editable_columns=["correct_class", "ruling", "note"])
    ruling_editor
    return (ruling_editor,)


@app.cell
def _(mo):
    save_rulings = mo.ui.button(label="Save rulings", value=0, on_click=lambda v: v + 1)
    save_rulings
    return (save_rulings,)


@app.cell(hide_code=True)
def _(CONFIGS, UTC, datetime, mo, pd, ruling_editor, save_rulings):
    if save_rulings.value:
        _df = pd.DataFrame(ruling_editor.value).copy()
        _df["decided_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        _df.to_csv(CONFIGS / "cross_class_rulings.csv", index=False)
        _o = mo.md(f"Saved **{len(_df)}** rulings to `configs/cross_class_rulings.csv`.")
    else:
        _o = mo.md("*Not saved yet.*")
    _o
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Cross-class near-duplicate queue

    These leakage groups connect images from more than one class through the decided RMS ≤ 4
    rule. Unlike the exact-duplicate queue, their images can be distinct, so rule each image:
    `keep` keeps its folder label, `reassign` supplies the correct class folder, and `drop`
    excludes it. Leave `ruling` blank to defer it.

    This is a label decision, not a split decision: all members remain in one leakage group even
    after a ruling. Writes `configs/image_rulings.csv`.
    """)
    return


@app.cell
def _(cross_near, mo):
    ngroups = sorted(cross_near.near_dup_group_id.unique())
    npick = mo.ui.dropdown(options={str(g): g for g in ngroups}, value=str(ngroups[0]),
                           label="near-duplicate group")
    npick
    return (npick,)


@app.cell(hide_code=True)
def _(class_char, cross_near, load_gray, mo, montage, npick, png):
    _g = cross_near[cross_near.near_dup_group_id == npick.value]

    def _cap(r):
        ch = class_char.get(r.class_folder, "")
        return f"cls {r.class_folder} {ch} · sha {r.sha_group_id}" if ch else f"cls {r.class_folder} · sha {r.sha_group_id}"

    _table = _g[["path", "class_folder", "sha_group_id", "sha_group_spans_classes"]].copy()
    _table["character"] = _g.class_folder.map(class_char)
    mo.vstack([
        mo.md(f"**{len(_g)} images · {_g.n_distinct_sha.iloc[0]} distinct exact-image groups · "
              f"classes {_g.classes_in_group.iloc[0]}**"),
        mo.image(png(montage([(load_gray(r.path), _cap(r)) for r in _g.itertuples()],
                              ncol=6, cell=96, scale=2, label_h=14)), width="100%",
                 caption=f"near-duplicate group {npick.value} · connected under RMS <= 4"),
        mo.ui.table(_table, selection=None),
    ])
    return


@app.cell
def _(CONFIGS, cross_near, pd):
    _path = CONFIGS / "image_rulings.csv"
    _existing = pd.read_csv(_path, keep_default_na=False) if _path.exists() else None
    _rows = []
    for _r in cross_near.itertuples():
        _prev = (_existing[_existing.path == _r.path].iloc[0]
                 if _existing is not None and _r.path in set(_existing.path) else None)
        _rows.append(dict(
            near_dup_group_id=_r.near_dup_group_id,
            path=_r.path,
            class_folder=_r.class_folder,
            classes_in_group=_r.classes_in_group,
            ruling="" if _prev is None else _prev.ruling,
            correct_class="" if _prev is None else _prev.correct_class,
            source="cross_class_near" if _prev is None else _prev.get("source", "cross_class_near"),
            note="" if _prev is None else _prev.get("note", ""),
        ))
    near_ruling_seed = pd.DataFrame(_rows)
    return (near_ruling_seed,)


@app.cell
def _(mo, near_ruling_seed):
    near_ruling_editor = mo.ui.data_editor(
        near_ruling_seed,
        label="cross-class near-duplicate image rulings",
        editable_columns=["ruling", "correct_class", "source", "note"],
    )
    near_ruling_editor
    return (near_ruling_editor,)


@app.cell
def _(mo):
    save_near_rulings = mo.ui.button(label="Save image rulings", value=0, on_click=lambda v: v + 1)
    save_near_rulings
    return (save_near_rulings,)


@app.cell(hide_code=True)
def _(
    CONFIGS,
    UTC,
    cross_near,
    datetime,
    mo,
    near_ruling_editor,
    pd,
    save_near_rulings,
):
    if save_near_rulings.value:
        _df = pd.DataFrame(near_ruling_editor.value).copy()
        _df["decided_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        _out = _df[["path", "ruling", "correct_class", "source", "note", "decided_at"]]
        _path = CONFIGS / "image_rulings.csv"
        # This view owns its queue, not any image decisions recorded elsewhere.
        _prior = pd.read_csv(_path, keep_default_na=False) if _path.exists() else pd.DataFrame()
        _prior = _prior[~_prior.path.isin(cross_near.path)] if "path" in _prior else _prior
        _prior = _prior.reindex(columns=_out.columns, fill_value="")
        pd.concat([_prior, _out], ignore_index=True).to_csv(_path, index=False)
        _decided = int((_out.ruling.astype(str).str.strip() != "").sum())
        _message = mo.md(f"Saved **{len(_out)}** rows to `configs/image_rulings.csv` — "
                         f"{_decided} decided, {len(_out) - _decided} deferred.")
    else:
        _message = mo.md("*Not saved yet.*")
    _message
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Outliers

    Flagged by robust z-score (per class) on width, height, aspect, file size and brightness, plus
    two absolute checks on ink coverage — see §9 of the audit notebook. An outlier is *unusual*,
    not *wrong* — that is what this view is for.

    Filter by class to check a single class's outliers against each other, not the whole queue —
    useful when a reason's flags turn out to cluster in one or two classes.
    """)
    return


@app.cell
def _(class_char, class_ids, mo, outliers):
    _reasons = sorted({r.split("(")[0] for rs in outliers.reasons for r in rs.split("; ")})
    def _opt(f):
        ch = class_char.get(f, "")
        return f"{f} {ch}" if ch else str(f)
    out_class = mo.ui.dropdown(
        options={"(all)": "(all)", **{_opt(f): f for f in class_ids}}, value="(all)", label="class")
    out_reason = mo.ui.dropdown(options=["(all)"] + _reasons, value="(all)", label="reason")
    mo.hstack([out_class, out_reason], justify="start", gap=2)
    return out_class, out_reason


@app.cell
def _(out_class, out_reason, outliers):
    out_filtered = outliers
    if out_class.value != "(all)":
        out_filtered = out_filtered[out_filtered.class_folder == out_class.value]
    if out_reason.value != "(all)":
        out_filtered = out_filtered[out_filtered.reasons.str.contains(out_reason.value, regex=False)]
    return (out_filtered,)


@app.cell
def _(mo, out_filtered):
    out_page = mo.ui.slider(0, max(1, (len(out_filtered) - 1) // 48), value=0, step=1,
                            label=f"page of 48 (of {max(1, -(-len(out_filtered) // 48))})")
    out_page
    return (out_page,)


@app.cell(hide_code=True)
def _(class_char, load_gray, mo, montage, out_class, out_filtered, out_page, out_reason, png):
    _pg = out_filtered.iloc[out_page.value * 48 : out_page.value * 48 + 48]
    def _cap(f):
        ch = class_char.get(f, "")
        return f"{f} {ch}" if ch else str(f)
    mo.vstack([
        mo.md(f"**{len(out_filtered):,}** images flagged" +
              ("" if out_class.value == "(all)" else f" in class `{out_class.value}`") +
              ("" if out_reason.value == "(all)" else f" for `{out_reason.value}`") +
              f" · showing {len(_pg)}"),
        mo.image(png(montage([(load_gray(r.path), _cap(r.class_folder)) for r in _pg.itertuples()],
                             ncol=12, cell=72, scale=2)),
                 width="100%")
        if len(_pg) else mo.md("*Nothing on this page.*"),
    ])
    return


if __name__ == "__main__":
    app.run()
