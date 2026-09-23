# thai-char-cnn

Thai handwritten character classification. This repo currently covers **dataset exploration**;
no model is trained here yet.

## Dataset

Download to `data/{dataset_name}/raw/`.

| Name | Download URL |
| --- | --- |
| `baseline` | https://drive.google.com/drive/folders/1mnGr83VaVpsJKVfQJD-na5kVwOORBtLS |

`raw/` is read-only and gitignored. Everything else under `data/{name}/` is derived from it.

## The two notebooks

Exploration is split by the *kind* of work, not by topic:

> **`01_dataset_audit.ipynb` owns derived data. `02_visual_check.py` owns human decisions.**
> Decisions flow back as small files under `configs/`, which the audit reads on its next run.

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
| `images.csv` | Per-image index — the canonical table. `path` is relative to `raw/`; row order is the index used by the pair tables. `near_dup_group_id` is the leakage group a split must keep whole. |
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
