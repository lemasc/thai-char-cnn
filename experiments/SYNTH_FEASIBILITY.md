# Synthetic dataset: feasibility review (2026-09-24, rerun 2026-09-25)

Dataset: `data/synthetic/raw/` has 16,200 PNGs (64×64, L mode) in 81 folders × 200 images,
rendered from 39 Windows Thai fonts (≈5 renders per font per class, with rotation, shear, blur
and weight augmentation). It comes with `classes.json` and `labels.csv`.
Reproduce: `uv run python experiments/synth_feasibility.py` (add `AUGMENT=1` for tiered
augmentation, `PREPROCESS=letterbox` for the old default). Results go to `runs_synth/<config>/`.

Current numbers use the cleaned split `d0a0d7ada85a`, with `stretch` and with and without tiered
augmentation. The first run (pre-audit split `49bb16a21863`, `letterbox`, no augmentation) is kept
at the end for comparison.

## Class mapping: compatible
- Every `classes.json` entry satisfies `char == bytes([folder]).decode("tis-620")` and `tis620 == folder`.
- All 72 baseline folders are present, and their characters match `configs/class_labels.csv` exactly.
- `labels.csv` has 16,200 rows. All files exist, and folder, char and font agree for every row.
- 9 extra classes are not in baseline: 165 ฅ, 166 ฆ, 172 ฌ, 174 ฎ, 198 ฦ, 208 ะ, 211 ำ, 218 ฺ, 235 ๋.
  That is 1,800 images, dropped here because the model head is fixed to baseline `class_idx` 0..71.
- There are no byte-identical duplicates. Visual spot checks found no tofu boxes or dotted-circle
  placeholders.
- The strongest check is training on synthetic data only. That model scores
  **0.79 ± 0.03 macro-F1 on real val** and **84% on the 260 real images of the 16 rare /
  not-validated classes** (images it never saw). A shifted or mismatched mapping would score close to 0.

## Domain gap: large
- Baseline images are tight ink crops (~20 px, pixelated, one print style). Synthetic images are padded
  64×64 renders in many styles. **Synthetic images must be cropped to their ink bbox** before
  `stretch` or `letterbox`, or the glyph scale and the geometry features (`log w, log h, w/h`) will not match.
  The script does this.
- The best real-only `small_cnn` on the current split labels only 62% of synthetic images correctly
  (56% on the pre-audit split). The errors are look-alike pairs (ฑ→ท 0%, ฃ→ข 1%, ๙→็, ๗→ฟ, ๅ→า),
  not a systematic offset.
- 2,411 images (15%) have ink touching the canvas edge. Some are truly clipped: for example, ฑ
  loses its head and becomes two strokes, and แ is cut off.

## Effect on real val (3 seeds, small_cnn, stretch, cleaned split)

Macro-F1 over validated classes, mean ± SD over seeds 42 / 137 / 271. "Small" is the mean F1 of
validated classes with fewer than 100 real train images.

| arm | train n | stretch F1 | small | stretch + augment F1 | small |
|---|---|---|---|---|---|
| real | 48,400 | 0.9889 ± 0.0020 | 0.9846 | 0.9894 ± 0.0013 | 0.9872 |
| synthetic only | 12,269 | 0.7922 ± 0.0308 | 0.6890 | 0.8249 ± 0.0299 | 0.7326 |
| real + synthetic (all) | 62,800 | 0.9874 ± 0.0013 | 0.9830 | 0.9857 ± 0.0021 | 0.9738 |
| real + synthetic, clipped dropped | 60,669 | 0.9896 ± 0.0018 | 0.9871 | 0.9865 ± 0.0013 | 0.9744 |
| real + synthetic, clipped dropped, rare classes only | 53,517 | 0.9880 ± 0.0017 | 0.9816 | 0.9852 ± 0.0016 | 0.9749 |

The `real` arm reproduces `reports/training-decisions.md` exactly (0.9889 / 0.9894).
Synthetic-only accuracy on the 260 unseen real rare images is 0.845 without augmentation and
0.844 with it.

## Verdict
The dataset can be used, with preprocessing. Labels line up with TIS-620 and with baseline.
- It does **not** improve the measurable val score. The best arm (clipped dropped, `stretch`) is
  +0.0007 over real-only, within seed noise. Unfiltered, it costs −0.0015.
- **It works against tiered augmentation.** With `augment=True`, every real + synthetic arm scores
  below real-only (−0.003 to −0.004), and the small validated classes drop most (0.987 → 0.974).
  The renders already carry rotation, shear and blur, so augmenting them again likely pushes them
  further from real glyphs. That is untested: an arm that augments only the real rows would check it.
- Its value is for the 16 rare / not-validated classes (ฃ ฑ ฤ ฬ ฮ ๔–๙ …). Real val cannot
  score those, but the synthetic-only result (84% on unseen real rare images, stable across both
  splits and both configs) shows they transfer.
- If adopted: crop to the ink bbox, drop edge-clipped renders, and keep synthetic images **train-only**
  (never in val). Decide separately whether the 9 extra classes belong in the class index.
  The hidden test set's class list is unknown.
- Pretrained backbones are untested here. Rerun against them once they land (E0 already skips
  models this script can't build).

## First run (2026-09-24): pre-audit split, letterbox, no augmentation

| arm | train n | val macro-F1 | val acc | F1, validated classes with <100 train |
|---|---|---|---|---|
| real | 48,535 | 0.9724 ± 0.0013 | 0.9771 | 0.9624 |
| synthetic only | 12,269 | 0.7907 ± 0.0244 | 0.8160 | 0.6923 |
| real + synthetic (all) | 62,935 | 0.9697 ± 0.0034 | 0.9760 | 0.9533 |
| real + synthetic, clipped dropped | 60,804 | 0.9722 ± 0.0009 | 0.9779 | 0.9590 |
| real + synthetic, clipped dropped, rare classes only | 53,667 | 0.9720 ± 0.0010 | 0.9790 | 0.9577 |
