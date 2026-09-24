# Training decisions

Experiments retired from `notebooks/04_train.ipynb` and the results that retired them. Scores are
val macro-F1 over validated classes, mean ± SD over seeds 42 / 137 / 271, small_cnn at 32 px.
Their runs stay in `runs/` (listed as "older code" once the code hash moves on).

## 2026-09-24 · Default input: stretch, not letterbox

Split `d0a0d7ada85a` (after the data audit), code `fdba95a2a0a6`.

| Experiment | Macro-F1 | Per seed |
| --- | --- | --- |
| stretch | 0.9889 ± 0.0020 | 0.9891 / 0.9868 / 0.9907 |
| letterbox | 0.9858 ± 0.0022 | 0.9850 / 0.9841 / 0.9883 |

Stretch wins on every seed, here and on the pre-audit split `49bb16a21863` (0.9747 vs 0.9724).
The expected cost, lost aspect cues for า/ๅ, did not appear: ๅ improved under stretch on both
splits. Stretch is somewhat worse on ฏ, ฐ and ู/ุ (up to −0.03 F1) and much better on ิ and ๓.
`letterbox` stays in the notebook as the control; rerun this comparison for any new backbone or
input size.

## 2026-09-24 · Retired: letterbox ablations and geometry-only augmentation

Same split and code.

| Experiment | Macro-F1 | Why retired |
| --- | --- | --- |
| letterbox+augment | 0.9858 ± 0.0018 | No gain over letterbox; replaced by stretch+augment (0.9894 ± 0.0013) |
| letterbox+geometry | 0.9857 ± 0.0010 | No gain over letterbox, which already keeps aspect; retested as stretch+geometry |
| letterbox+augment-uniform | 0.9840 ± 0.0039 | 0.0018 below tiered, within noise (one seed at 0.9796); retested as stretch+augment-uniform |
| stretch+augment-geometry-only | 0.9871 ± 0.0004 | 0.0023 below full augment, more than the seed spread: the photometric part of augment helps |

Open at retirement: stretch vs stretch+augment is a tie (0.9889 vs 0.9894).

## 2026-09-24 · Tested on stretch: geometry features and uniform augmentation

Same split and code. Both were rerun on stretch after the letterbox versions above.

| Experiment | Macro-F1 | Per seed | Compared with |
| --- | --- | --- | --- |
| stretch+geometry | 0.9881 ± 0.0007 | 0.9884 / 0.9885 / 0.9873 | stretch 0.9889 ± 0.0020 |
| stretch+augment-uniform | 0.9842 ± 0.0024 | 0.9868 / 0.9819 / 0.9840 | stretch+augment 0.9894 ± 0.0013 |

- **Geometry features add nothing on stretch either** (−0.0008, within noise). They win back part
  of stretch's loss on ฏ (+0.018) but lose on ็, ญ and ฐ. Retired.
- **Per-class tiers matter.** Uniform augmentation is below tiered on every seed (−0.0052) and below
  unaugmented stretch too. The biggest losses are ๅ (−0.048), า (−0.044), ฐ and ๓, which are the
  classes the tiers augment most gently. Retired; `configs/augment.json` tiers stay.
