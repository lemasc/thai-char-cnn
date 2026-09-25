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

## 2026-09-25 · Transfer learning: how much of the ImageNet backbone to freeze

Split `d0a0d7ada85a`, code `ac396e469e8b`. ResNet-18 at 64 px with `IMAGENET1K_V1` weights,
tiered augmentation and `lr=1e-3`. The three rows share this recipe and differ only in
`freeze_through`. Frozen modules get no gradients and keep their ImageNet BatchNorm statistics.

| Experiment | Trains | Trainable params | Macro-F1 | Per seed | Best epoch (mean) |
| --- | --- | --- | --- | --- | --- |
| resnet18-pretrained@64 (full fine-tune) | everything | 11.2 M | 0.9872 ± 0.0004 | 0.9867 / 0.9875 / 0.9875 | 18 |
| resnet18-pretrained-frozen-l2@64 | `layer3`, `layer4`, head | 10.5 M | 0.9870 ± 0.0004 | 0.9871 / 0.9866 / 0.9873 | 24 |
| resnet18-pretrained-frozen@64 (head only) | dropout + linear head | 37 k | 0.8561 ± 0.0025 | 0.8543 / 0.8590 / 0.8551 | 32 |

- **The head-only probe is 0.13 below full fine-tuning.** ImageNet features pooled from the 2×2
  `layer4` map don't separate Thai characters linearly. The loss is broad: 27 of 56 validated
  classes drop by more than 0.1 F1. The worst are ฉ (−0.71), ๓ (−0.47), ๑ (−0.39) and ๆ (−0.37),
  which are shapes that differ by small loops and tails. Accuracy falls less than macro-F1
  (0.902 vs 0.989), so the damage is concentrated in the smaller classes.
- **Freezing the stem, `layer1` and `layer2` costs nothing** (−0.0002, within noise; no class moves
  more than 0.03 F1). The low-level ImageNet filters transfer to handwriting; the deeper stages are
  the part that has to adapt. Freezing them only saves 6% of the trainable parameters, though,
  because most of ResNet-18's weights sit in `layer3` and `layer4`.
- **The probe is a lower bound on what ImageNet features can do.** It kept improving late (best
  epochs 29–36 of 40, rarely early-stopped), so a head-specific recipe with a higher learning rate
  and more epochs would score somewhat higher. It was kept on the shared recipe so that the three
  rows differ only in what is frozen.
- **Freezing does not speed up training here.** CPU-side augmentation is the bottleneck, and the
  frozen runs trained for more epochs (13–16 min per 3 seeds vs 11 for full fine-tuning).

All three rows stay in the notebook for the report. None of them beats the from-scratch
`resnet18@64` or `small_cnn` rows from the ResNet comparison on the previous code (0.985–0.991). Those
rows haven't been retrained on this code yet, so the ranking waits for that rerun.

## 2026-09-25 · CNN + MLP: an MLP classifier on the pooled features

Split `d0a0d7ada85a`, train version `1.p2`. Each `+mlp` row repeats a row above with the single linear
head replaced by `mlp_hidden` (dropout, then `Linear -> BatchNorm1d -> ReLU -> Dropout`, then the output
layer); everything else is the same recipe.

| Experiment | Hidden | Macro-F1 | Compared with (linear head) | Δ |
| --- | --- | --- | --- | --- |
| stretch+augment+mlp | 256 | 0.9889 ± 0.0018 | stretch+augment 0.9894 ± 0.0013 | −0.0005 |
| resnet18@64+mlp | 512 | 0.9894 ± 0.0014 | resnet18@64 0.9884 ± 0.0029 | +0.0010 |
| resnet18-pretrained-frozen+mlp@64 | 512 | 0.8905 ± 0.0082 | resnet18-pretrained-frozen@64 0.8561 ± 0.0025 | +0.0344 |

- **End to end, the MLP head changes nothing.** Both gaps are inside the seed spread. When the
  convolutional layers are trained, they already make the pooled features linearly separable, so an
  extra nonlinear layer on top has nothing left to fix.
- **Separating the jobs, with a frozen ImageNet extractor and a trained MLP classifier, recovers only a
  quarter of the probe's gap.** It is +0.034 over the linear probe but still about 0.10 below every
  model whose convolutional layers were trained. The features, not the classifier, are what's missing.
  It was still improving at the end (best epoch 36 of 40), so it is a lower bound, as with the probe.

## 2026-09-25 · Pruned to four rows before merging

Split `d0a0d7ada85a`, train version `1.p2`. Results for every row, retired or kept, are compared in
`reports/final-report.md`, which lists the run ids.

Kept: `stretch+augment` (the recommended model), `stretch` (its no-augmentation control; the two
are still tied at 0.9894 vs 0.9889), `resnet18@64` and `resnet18-pretrained@64` (the reference points
for any future backbone). The deep dive now focuses on `stretch+augment`.

| Retired | Macro-F1 | Why retired |
| --- | --- | --- |
| letterbox | 0.9858 ± 0.0022 | Stretch wins on every seed, on this split and the pre-audit one |
| resnet18-pretrained-frozen-l2@64 | 0.9870 ± 0.0004 | Freezing through `layer2` costs nothing, but saves only 6% of trainable parameters |
| resnet18-pretrained-frozen@64 | 0.8561 ± 0.0025 | A frozen ImageNet backbone is 0.13 below any trained one |
| resnet18-pretrained-frozen+mlp@64 | 0.8905 ± 0.0082 | An MLP recovers only a quarter of the frozen backbone's gap |
| stretch+augment+mlp | 0.9889 ± 0.0018 | MLP head is within noise of the linear head (−0.0005) |
| resnet18@64+mlp | 0.9894 ± 0.0014 | Ties for first, but within noise of `resnet18@64` (+0.0010) |

- The ranking that the transfer-learning entry above left open is settled: on this code, the
  from-scratch `resnet18@64` (0.9884 ± 0.0029) and `small_cnn` rows tie, and none of the pretrained
  rows beats them.
- Open at retirement: letterbox was only tested on `small_cnn`, never on ResNet-18 at 64 px.
