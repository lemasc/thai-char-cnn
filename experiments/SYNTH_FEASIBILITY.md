# Synthetic dataset: feasibility review (2026-09-24)

Dataset: `data/synthetic/raw/` has 16,200 PNGs (64×64, L mode) in 81 folders × 200 images,
rendered from 39 Windows Thai fonts (≈5 renders per font per class, with rotation, shear, blur
and weight augmentation). It comes with `classes.json` and `labels.csv`.
Reproduce: `uv run python experiments/synth_feasibility.py`. Results go to `runs_synth/`.

> Snapshot: numbers use the pre-audit split at `7e1737e` with `letterbox` and no augmentation.
> Main has since moved to the cleaned split and `stretch`, so re-measure before comparing.

## Class mapping: compatible
- Every `classes.json` entry satisfies `char == bytes([folder]).decode("tis-620")` and `tis620 == folder`.
- All 72 baseline folders are present, and their characters match `configs/class_labels.csv` exactly.
- `labels.csv` has 16,200 rows. All files exist, and folder, char and font agree for every row.
- 9 extra classes are not in baseline: 165 ฅ, 166 ฆ, 172 ฌ, 174 ฎ, 198 ฦ, 208 ะ, 211 ำ, 218 ฺ, 235 ๋.
  That is 1,800 images, dropped here because the model head is fixed to baseline `class_idx` 0..71.
- There are no byte-identical duplicates. Visual spot checks found no tofu boxes or dotted-circle
  placeholders.
- The strongest check is training on synthetic data only. That model scores
  **0.79 ± 0.02 macro-F1 on real val** and **84% on the 262 real images of the 16 rare /
  not-validated classes** (images it never saw). A shifted or mismatched mapping would score close to 0.

## Domain gap: large
- Baseline images are tight ink crops (~20 px, pixelated, one print style). Synthetic images are padded
  64×64 renders in many styles. **Synthetic images must be cropped to their ink bbox** before
  `letterbox`, or the glyph scale and the geometry features (`log w, log h, w/h`) will not match.
  The script does this.
- The best real-only model labels only 56% of synthetic images correctly. The errors are
  look-alike pairs (ฃ→ข, ฑ→ท, ๙→์, ๅ→า), not a systematic offset.
- 2,411 images (15%) have ink touching the canvas edge. Some are truly clipped: for example, ฑ
  loses its head and becomes two strokes, and แ is cut off.

## Effect on real val (3 seeds, small_cnn, letterbox, no aug)

| arm | train n | val macro-F1 | val acc | F1, validated classes with <100 train |
|---|---|---|---|---|
| real | 48,535 | 0.9724 ± 0.0013 | 0.9771 | 0.9624 |
| synthetic only | 12,269 | 0.7907 ± 0.0244 | 0.8160 | 0.6923 |
| real + synthetic (all) | 62,935 | 0.9697 ± 0.0034 | 0.9760 | 0.9533 |
| real + synthetic, clipped dropped | 60,804 | 0.9722 ± 0.0009 | 0.9779 | 0.9590 |
| real + synthetic, clipped dropped, rare classes only | 53,667 | 0.9720 ± 0.0010 | 0.9790 | 0.9577 |

## Verdict
The dataset can be used, with preprocessing. Labels line up with TIS-620 and with baseline.
- It does **not** improve the measurable val score. Unfiltered, it slightly hurts (−0.003).
  With clipped images dropped, the result is within seed noise.
- Its value is for the 16 rare / not-validated classes (ฃ ฑ ฤ ฬ ฮ ๔–๙ …). Real val cannot
  score those, but the synthetic-only result (84% on unseen real rare images) shows they transfer.
- If adopted: crop to the ink bbox, drop edge-clipped renders, and keep synthetic images **train-only**
  (never in val). Decide separately whether the 9 extra classes belong in the class index.
  The hidden test set's class list is unknown.
