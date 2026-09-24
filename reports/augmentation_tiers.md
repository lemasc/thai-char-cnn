# Tiered augmentation: what we found so far

*Status as of 2026-09-24. Dataset `baseline`, split `49bb16a21863`, model `small_cnn` at 32 px letterbox.*

## Summary

Per-class tiered augmentation now matches or slightly beats the un-augmented letterbox baseline.
The gain is small, and it only appeared after four characters were moved out of the full-strength
tier. The first version, with the tiers exactly as designed, was clearly worse. It taught the
model to send clean ว images to า.

The best configuration so far is round A4 below. Its gain over the baseline is below one percent of
macro-F1 and comparable to the seed spread. Treat it as "no longer hurts, possibly helps a little",
not as a proven improvement. Stretch preprocessing without augmentation is still the best row in
the leaderboard. Stretch combined with augmentation has not been tried yet.

## What was built

Augmentation lives in `src/thai_char_cnn/augment.py`. Every class is looked up in a
character-to-tier table, and each tier has its own profile.

| tier | who | treatment |
| --- | --- | --- |
| 1 | glyphs with no close visual neighbour | full range: ±10° rotation, shear, elastic, erosion/dilation, cutout, blur, noise, brightness/contrast |
| 2 | hook/loop confusable pairs (ผ/ฝ, บ/ป, ศ/ษ/ส ...) | ±5° rotation, rare small shear, no cutout, mild blur |
| 3 | ascender/descender letters (ป ฝ ฟ ฤ ญ ฏ ...) | Tier 1 rotation, small vertical shift, bbox jitter never trims top/bottom |
| 4 | tone marks and vowel diacritics | ±2.5° rotation, no shear, elastic, stroke change, blur or cutout |
| 5 | extreme near-duplicates (า/ๅ) | ±2° rotation and almost nothing else |

A character in several tiers gets the most conservative value of every setting. Geometry is one
resample that never pushes ink out of the frame. The tier assignments can be overridden in
`configs/augment.json` without touching code, which is how every round below was run.

## How it was measured

- Each configuration was trained with three seeds (42, 137, 271), everything else fixed.
- The headline score is validation macro-F1 over validated classes, the notebook's leaderboard metric.
- Confusion counts are summed over the three seeds' validation confusion matrices.
- The comparison point is the `letterbox` row: the same model and preprocessing with augmentation off.

## Results

| round | tier change on top of the previous round | macro-F1 | seed sd | accuracy | val errors |
| --- | --- | --- | --- | --- | --- |
| baseline | augmentation off | 0.9724 | 0.0013 | 0.9771 | 826 |
| A1 | default tiers | 0.9683 | 0.0014 | 0.9676 | 1171 |
| A2 | ว, จ: Tier 1 → 4 | 0.9696 | 0.0038 | 0.9766 | 844 |
| A3 | ด, ต → 4; ว → 5 | 0.9735 | 0.0007 | 0.9776 | 808 |
| A4 | ใ: Tier 2 → 5 | **0.9731** | **0.0003** | **0.9782** | **788** |

For reference, the other un-augmented rows are stretch at 0.9747 and letterbox with geometry
features at 0.9738.

Confusions between the pairs that drove each decision, both directions summed:

| pair | baseline | A1 | A2 | A3 | A4 |
| --- | --- | --- | --- | --- | --- |
| ว / า | 148 | 456 | 157 | 157 | 140 |
| า / ๅ | 213 | 219 | 205 | 183 | 208 |
| ใ / า | 86 | 102 | 106 | 106 | 88 |
| ด / ต | 45 | 44 | 46 | 45 | 44 |
| ั / ้ | 48 | 21 | 20 | 31 | 19 |
| Tier 1 classes, total errors | 126 | 165 | 162 | 141 | 147 |

## Findings

1. **Per-class augmentation strength leaks the label.** In A1, true ว was predicted as า 426
   times against 68 at baseline. ว got the full Tier 1 treatment while า got almost none, so a
   clean-looking image became evidence for า. Validation images are never augmented, so every
   clean ว looked a little more like า. Moving ว, and later other characters, to the same tier
   as the thing it is confused with removed most of the damage.
2. **Matching the tier of a confused pair works better than softening one side.** ว in Tier 4
   (A2) brought ว back near baseline. Putting ว in Tier 5 with า (A3) brought its errors back to
   baseline. Putting ใ in Tier 5 with า (A4) cut ใ/า from 106 to 88, about the baseline level,
   and ใ's own errors fell from 84 to 58, below the baseline's 74.
3. **Small marks benefit clearly.** ั/้ confusions are less than half the baseline in every
   augmented round except A3.
4. **ด/ต is not an augmentation problem.** It stays at about 45 in every configuration,
   including Tier 4 for both letters. It needs a different fix: resolution, data or a
   dedicated check.
5. **า/ๅ moves around but stays the largest single confusion.** It improved in A3 and returned
   near baseline in A4. The tier rules already say this pair needs non-augmentation fixes more
   than augmentation tuning.
6. **Tier 1 classes still carry a cost.** Their total errors stay above baseline in every
   augmented round. The remaining difference between tiers in noise, blur and photometric
   settings is the most likely cause, for the same reason as finding 1.

## Caveats

- **The tier changes were chosen by looking at validation results.** The same validation set
  then scores them, so the gains in A2 to A4 are optimistic. Only the hidden test set can confirm them.
- **Validation is also optimistic for a data reason.** The dataset is rendered fonts, and fonts
  are shared across writer IDs within a filename prefix. The split is by writer ID, so many
  validation glyphs have a near-identical twin in train. Augmentation's benefit on truly unseen
  fonts could be larger or smaller than measured here.
- **Differences between A3, A4 and the baseline are small.** Both gains are smaller than one
  baseline seed standard deviation. Three seeds are enough to see A1 is worse, not to rank A3 against A4.

## Control: uniform augmentation (no tiers)

*Added 2026-09-24.* Every earlier round compared tiers against no augmentation. This control
gives every class the same profile (`configs/augment_uniform.json`, row
`letterbox+augment-uniform` in `04_train`): the pre-tier global defaults, ±8° rotation, ±5°
shear, 0.9-1.1 scale, 4% shift, with stroke, elastic, bbox jitter, blur, noise, cutout and
brightness/contrast off. It uses the same pipeline and seeds as A4.

| row | macro-F1 | seed sd | accuracy | val errors | ว/า | า/ๅ | ใ/า | ด/ต | ั/้ | Tier 1 errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (off) | 0.9724 | 0.0013 | 0.9771 | 826 | 148 | 213 | 86 | 45 | 48 | 126 |
| A4 (tiered) | 0.9731 | 0.0003 | 0.9782 | 788 | 140 | 208 | 88 | 44 | 19 | 147 |
| uniform | 0.9721 | 0.0013 | 0.9737 | 950 | 168 | 328 | 91 | 64 | 28 | 114 |

- **Macro-F1 can't separate the three.** They are within about one seed sd of each other.
- **The errors show where tiers matter.** Uniform geometry hurts the near-duplicates: า/ๅ goes up
  by more than half (213 → 328) and ด/ต by 40% (45 → 64), and total val errors are the highest of
  any row. Tier 5's near-zero geometry for า/ๅ is doing real work.
- **Uniform is better for Tier 1 classes** (114 errors, versus 126 for baseline and 147 for A4).
  This supports finding 6: A4's extra noise, blur and photometric changes on Tier 1, or the
  difference in strength between tiers, costs the robust classes something.
- **Marks improve with either kind of augmentation** (ั/้ 48 → 28 uniform, 19 tiered).

Taken together: keep per-tier *geometry* limits, which protect the confusable pairs. Next-step 1
below, the same non-geometric settings for every tier, is now the most promising change.

## Current configuration

`configs/augment.json` holds these overrides on top of the default tier table in `augment.py`:

| character | default tier | now |
| --- | --- | --- |
| ว | 1 | 5 |
| จ | 1 | 4 |
| ด | 2 | 4 |
| ต | 2 | 4 |
| ใ | 2 | 5 |
| า | 5 | 5 (explicit) |

## Suggested next steps

1. Use the same noise, blur and brightness/contrast settings for every tier, and keep only the
   geometric limits per tier. This targets the remaining Tier 1 cost in finding 6.
2. Try augmentation with stretch preprocessing, the current best un-augmented setup.
3. Look at ด/ต and า/ๅ outside augmentation, for example at higher input resolution.
4. Before trusting sub-percent gains, check them on a font-aware split or the hidden test set.
