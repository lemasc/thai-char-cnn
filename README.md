# thai-char-cnn

Thai handwritten character classification: dataset exploration, a writer-level train/val split,
and a baseline CNN.

```bash
uv sync     # installs deps and the shared `thai_char_cnn` package (src/) in editable mode
```

## Dataset

Download to `data/{dataset_name}/raw/`.

| Name | Download URL |
| --- | --- |
| `baseline` | https://drive.google.com/drive/folders/1mnGr83VaVpsJKVfQJD-na5kVwOORBtLS |

`raw/` is read-only and gitignored. Everything else under `data/{name}/` is derived from it.

## The notebooks

Exploration is split by the *kind* of work, not by topic:

> **`01_dataset_audit.ipynb` owns derived data. `02_visual_check.py` owns human decisions.**
> Decisions flow back as small files under `configs/`, which the audit reads on its next run.
> `03_split.ipynb` applies those decisions to produce the split; `04_train.ipynb` trains on it.
> Logic shared between notebooks lives in `src/thai_char_cnn/`.

### 1 · `notebooks/01_dataset_audit.ipynb` — arithmetic & logical checks

```bash
uv run jupyter lab notebooks/01_dataset_audit.ipynb
```

Measures only. Filesystem discovery, file inventory, a single-pass scan (geometry, colour mode,
intensity stats, SHA-256, pHash, a 16×16 downsample), integrity checks, exact-duplicate resolution,
class imbalance, outliers, and both near-duplicate signals computed exhaustively over all 1.97
billion pairs. Ends with a self-check that every published number ties out.

It deliberately does **not** assign semantic labels, choose a near-duplicate threshold, drop
anything, or pronounce the dataset fit for training. Runs in about a minute; the scan is cached
against a content signature of `raw/`.

Sole writer of `data/{name}/manifest/`:

| File | Contents |
| --- | --- |
| `images.csv` | Per-image index — the canonical table. `path` is relative to `raw/`; row order is the index used by the pair tables. `near_dup_group_id` is the leakage group a split must keep whole. `writer_id` / `session` / `sheet` are parsed from the file name (`{src}_{num}{session}_{sheet}_{idx}.jpg`, `Copy of ` stripped); `writer_id` is the unit the split is made on. |
| `classes.csv` | Per-class counts (raw and exact-dup-collapsed), geometry and brightness stats |
| `folder_inventory.csv`, `non_images.csv` | What is on disk, including stray non-image files |
| `findings_cross_class_exact.csv` | Byte-identical images under two different labels — an adjudication queue |
| `findings_cross_class_near.csv` | Near-duplicate groups (per `configs/near_dup.json`) spanning two classes, one row per image — an adjudication queue |
| `findings_outliers.csv` | Robust-z outliers, with per-metric reasons |
| `dist_phash.csv`, `dist_pixel.csv` | Full near-duplicate distance distributions |
| `pairs/pairs_{phash,pixel}.csv.gz` | Near-duplicate candidate pairs, as row indices into `images.csv`. **Not tracked** — ~14 MB, rebuilt in ~21s |
| `run.json` | Versions, config, timings, self-check results |

### 2 · `notebooks/02_visual_check.py` — visual judgement

```bash
uv run marimo edit notebooks/02_visual_check.py
```

Reads the manifest, shows one canonical copy per exact-duplicate group, and writes only to
`configs/`. Covers: an all-class prototype sheet, a per-class sample browser, a label recorder
seeded with the collation-offset hypothesis, a near-duplicate threshold tuner that contrasts the two
signals, a cross-class duplicate adjudicator, and an outlier browser.

### 3 · `notebooks/03_split.ipynb` — train / val split

```bash
uv run jupyter lab notebooks/03_split.ipynb
```

Applies the rulings in `configs/` to get one label per image, then splits **by writer** (parsed from
the file name), so validation measures handwriting the model has never seen. There is no test split:
the instructor holds the real test set, and val is used for model selection, so its score is
optimistic. Classes with fewer than `min_val_class_images` distinct images go wholly to train and
are flagged *not validated*. They are the only images allowed to break writer purity, and they are
counted.

The writer assignment is **frozen** in `split/writers.csv` and reused while `configs/split.json` is
unchanged. New rulings then move only the images they touch, never whole writers. Byte-identical
glyphs shared across writers (1,000 SHA groups, mostly tiny bars that many writers drew
identically) are *reported* as cross-split twins, not enforced: joining writers through them
would chain 37 of 52 writers into one block. Runs in a few seconds.

Sole writer of `data/{name}/split/` (tracked):

| File | Contents |
| --- | --- |
| `writers.csv` | writer → `train`/`val`, with the hash of the split config it was built under |
| `images.csv` | One row per manifest image: `writer_id`, `orig_class`, `label_class`, `class_idx`, `split`, `included`, `exclude_reason`, `ruling_source`, `forced_train`, `is_split_canonical` (one row per SHA, label and split — copies count once per side) |
| `classes.csv` | The fixed class index (`class_idx` 0..71 from the audit's folder list, never renumbered by rulings), `train_n`, `val_n`, `status` ∈ `validated` / `weak` (val < 5) / `not_validated` |
| `run.json` | `split_id` (changes exactly when what a model trains or is scored on changes), counts, search summary, self-checks |

### 4 · `notebooks/04_train.ipynb` — experiments as data

```bash
uv run jupyter lab notebooks/04_train.ipynb
EPOCHS=1 uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_train.ipynb   # smoke run
VERIFY=1 uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_train.ipynb   # + reproducibility check
```

An experiment is a named row of config overrides on a shared `BASE`. The notebook defines no
training logic: it calls `thai_char_cnn.train.fit`, which caches each run under
`runs/<hash(config, seed, split_id, code_hash)>/`, so a rerun loads finished runs and adding a row
trains only that row. `code_hash` covers the training-side modules of `src/thai_char_cnn/`, so
editing them invalidates the cache rather than serving results from older code. `RETRAIN=1`
retrains every row regardless; `VERIFY=1` retrains the first run into a scratch folder and checks
it reproduces the cached one (it does, bit for bit, on the RTX 3060). Sections: data check with an augmentation montage, experiment table, one train
loop, a leaderboard (val macro-F1 over validated classes, mean ± sd across seeds; only runs on the
current `split_id` are ranked), and a reproducibility check, and a deep dive (per-class F1, top confused pairs, error gallery)
on the experiment named by `FOCUS`.

`runs/` is gitignored. Each run holds `config.json`, `history.csv`, `metrics.json`,
`per_class.csv`, `confusion.npy`, `val_predictions.csv`, `model.pt`.

### Decision files under `configs/`

| File | Written by | Read by | Contents |
| --- | --- | --- | --- |
| `class_labels.csv` | 02 | 03, 04 | Folder → Thai character |
| `near_dup.json` | 02 | 01 | Near-duplicate RMS threshold |
| `cross_class_rulings.csv` | 02 | 03 | Per exact cross-class SHA group: `reassign` to `correct_class`, or `hold` (excluded) |
| `image_rulings.csv` | 02 (optional) | 03 | Per image: `path, ruling, correct_class, source, note`, `ruling` ∈ `keep` / `reassign` / `drop`. Overrides group rulings; a missing file or row means keep |
| `split.json` | by hand | 03 | `unit` (`writer`), `val_fraction`, `min_val_class_images`, `seed`, `search_iters`, `unruled_cross_class` (`drop` or `keep`: what happens to an image in a near-dup group that still carries two labels after the rulings) |
| `augment.json` | `05_augment_preview.py` (planned; optional) | 04 | Any `AugmentConfig` field (`rotation_deg`, `shear_deg`, `scale_min`, `scale_max`, `translate_frac`, `stroke_p`, `elastic_p`, `elastic_alpha`, `elastic_sigma`). Missing → conservative defaults, with stroke and elastic off |

## Notes from the current `baseline` audit

- 72 class folders spanning 161–249, with 17 IDs absent; 62,707 images and 4 stray files.
- All images are 8-bit grayscale, sizes vary widely (988 distinct `w×h`, 2–138 px wide).
- 2,579 redundant exact copies; 11 SHA-256 groups span two classes — those are labelling
  contradictions, left unresolved for `02_visual_check.py`.
- pHash sets bits against the median of 64 DCT coefficients, so a hash with no tied coefficients
  carries exactly 32 set bits — and equal popcounts can only differ by an even number of bits.
  62,644 of 62,707 hashes are like that. Measured directly: **0 pairs sit at Hamming distance 1**,
  so the prior pipeline's `phash_hamming_threshold = 1` merged nothing beyond exact hash equality.
  Pixel RMS distance is the finer-grained signal; both are provided.
- `imagehash.phash` is used directly rather than reimplemented. A numpy DCT matched it on 62,702 of
  62,707 images but broke exactly-tied coefficients on the five smallest glyphs, flipping 16 bits.
  Ties are load-bearing in pHash, so the reference implementation is the right one to depend on.
