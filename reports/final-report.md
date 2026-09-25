# Thai character classification: final report

*As of 2026-09-25. Dataset `baseline`, split `d0a0d7ada85a`, train version `1.p2`. Every number below
comes from cached runs in `runs/`. The run ids are in the appendix, so the numbers can still be traced
if rows are later removed from `notebooks/04_train.ipynb`.*

## Summary

- **Best validation score: macro-F1 0.9894, accuracy 0.9902.** Two configurations share it: the small
  CNN with tiered augmentation (`stretch+augment`) and ResNet-18 trained from scratch with an MLP head
  (`resnet18@64+mlp`). The gap between them is 0.0000 ± 0.0014.
- **Recommended model: `stretch+augment`.** It scores the same as the ResNet with 38× fewer parameters
  (296 k vs 11.5 M), trains 2.5× faster and is among the best calibrated (ECE 0.0030).
- **Most of the configurations that train their convolutional layers tie.** The eight rows from
  `letterbox` to `resnet18@64+mlp` all fall between 0.9858 and 0.9894 macro-F1, a range about three
  seed standard deviations wide. Neither model size, ImageNet pretraining nor an MLP head moved the
  score by more than the seed noise.
- **Two choices clearly matter.** Stretching the glyph to a square beats letterboxing it. Freezing the
  ImageNet backbone costs about 0.10–0.13 macro-F1.
- **One confusion pair dominates the remaining errors.** า (sara aa) against ๅ (lakkhangyao) accounts
  for 62% of the best models' validation errors. No architecture change fixed it.

## 1 · Setup

### Data and split

| | |
| --- | --- |
| Images | 62,707 in 72 class folders; 62,480 kept after the audit and rulings (227 dropped or held) |
| Train / val | 48,400 / 11,996 images (exact copies counted once per side), split by `{src}_{num}` filename group (43 groups in train, 9 in val) |
| Classes | 56 *validated*; 5 *weak* (fewer than 5 val images: ฝ ฯ แ ๊ ๘); 11 *not validated* (no val images: ฃ ฑ ฤ ฬ ฮ ึ ๔ ๕ ๖ ๗ ๙) |

Validation groups are held out whole, so val measures filename groups that were never seen in
training. It is **not** a test set. It was used for early stopping and for choosing between
experiments, so every number here is somewhat optimistic. The instructor holds the real test set.
Nothing here measures the 11 not-validated classes.

### Metrics

All metrics are computed on the val set with the checkpoint from the best epoch.

| Metric | Definition | Why it's reported |
| --- | --- | --- |
| **Macro-F1** (headline) | Unweighted mean of per-class F1 over the 56 validated classes | The classes are imbalanced (17 to 3,884 train images per validated class), so each class gets equal weight |
| Macro-F1 (all) | The same, over all 61 classes with val images, the weak ones included | Shows how noisy the weak classes are |
| **Accuracy** | Share of val images classified correctly | Dominated by the ten biggest classes |
| Macro precision / recall | Unweighted means over the validated classes. Macro recall equals balanced accuracy | Separates missed images from false alarms |
| Weighted F1 | Per-class F1 weighted by val support | Sits between accuracy and macro-F1 |
| Worst-class F1 | Lowest F1 among validated classes, averaged over seeds | Usually ๑, which has only 7 val images, so one error costs about 0.07 |
| Errors | Misclassified val images, out of 11,996 | |
| NLL | Mean −log p(true class) | Penalises confident mistakes |
| ECE | Expected calibration error, 15 equal-width confidence bins | How far confidence is from the actual hit rate |

Every configuration was trained with three seeds (42, 137 and 271) and is reported as mean ± SD.
With only three seeds, **a gap smaller than about 0.003 macro-F1 is within noise**.

### Training recipe (shared)

AdamW, weight decay 5e-4, batch size 256, one warm-up epoch, then cosine decay over up to 40 epochs.
Early stopping has patience 8 on val macro-F1. No label smoothing, mixed precision on, RTX 3060.
The small CNN uses 32 px input at `lr=3e-3`. ResNet-18 uses 64 px input at `lr=1e-3`, because its
stem downsamples 4×. The ResNet stem takes one channel (see *Transfer learning* below for how it is
initialised from ImageNet).

- **`small_cnn`**: three blocks, each two 3×3 conv + BN + ReLU and a max-pool (32 → 64 → 128
  channels), then global average pooling, dropout 0.3 and a linear head. 296 k parameters.
- **`resnet18`**: torchvision ResNet-18 with the same dropout + linear head. 11.2 M parameters.
- **Tiered augmentation** (`augment=True`): augmentation strength depends on the class. Robust glyphs
  get the full range: rotation, shear, elastic distortion, stroke thickness, cutout, blur, noise and
  contrast. Confusable pairs, ascender/descender letters, marks and the า/ๅ pair get progressively
  gentler settings. See `reports/augmentation_tiers.md`.

### Transfer learning

Transfer learning reuses weights learned on a large dataset as the starting point for a new task,
instead of starting from random values. The question here is whether features learned on ImageNet
(1.28 M colour photos in 1,000 classes) help with single grey handwritten Thai glyphs, which look
nothing like photos. Only ResNet-18 is tested, because the small CNN has no pretrained weights.

**Loading the weights.** The ResNet-18 backbone starts from torchvision's `IMAGENET1K_V1` weights,
and the 1,000-class ImageNet head is replaced with a new, randomly initialised head for the 72
classes. ImageNet's first convolution expects three colour channels, but these images have one. The
one-channel stem kernel is the sum of the three RGB kernels, which gives exactly the same output as
feeding the grey image into all three channels. Inputs are 64 px, because the ImageNet stem
downsamples 4× and a 32 px glyph would shrink to 8 × 8 before `layer1`. Pixels are normalised with
this dataset's train mean and SD, not ImageNet's.

**How much to reuse.** ResNet-18 is a stem followed by four stages (`layer1` to `layer4`). The early
stages detect generic patterns such as edges and strokes, and the late stages detect larger shapes
specific to what the network was trained on. Freezing a stage keeps its ImageNet weights fixed: it
gets no gradients, and its BatchNorm layers stay in eval mode, so they keep the ImageNet running
statistics. The experiments move the freeze point from nothing to the whole backbone:

| Experiment | Initialisation | Frozen | Trained | Trainable params |
| --- | --- | --- | --- | --- |
| `resnet18@64` | random | nothing | everything | 11.2 M |
| `resnet18-pretrained@64` | ImageNet | nothing (full fine-tune) | everything | 11.2 M |
| `resnet18-pretrained-frozen-l2@64` | ImageNet | stem, `layer1`, `layer2` | `layer3`, `layer4`, head | 10.5 M |
| `resnet18-pretrained-frozen@64` | ImageNet | whole backbone | linear head (linear probe) | 37 k |
| `resnet18-pretrained-frozen+mlp@64` | ImageNet | whole backbone | MLP head, 512 hidden | 301 k |

The random-init row is the baseline. Comparing it with full fine-tuning shows whether ImageNet is a
better *starting point*. The frozen rows show how far ImageNet's *features* can go on their own.

**Same recipe on purpose.** All five rows use the shared recipe above: `lr=1e-3`, the same tiered
augmentation and the same 40-epoch schedule. Transfer learning is often run with a lower learning
rate for the pretrained layers or longer training for frozen backbones. Neither was done here, so
that the rows differ only in initialisation and in what is frozen. The results are in §3.4.

## 2 · Results

Mean over three seeds, sorted by macro-F1. **Bold** marks the best value in each column; for the
error columns, lower is better.

| Experiment | Model | Macro-F1 | Accuracy | Macro-P | Macro-R | Weighted F1 | Worst class | Errors | NLL | ECE | Params (trainable) | Min / run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `stretch+augment` | small CNN | **0.9894 ± 0.0013** | 0.9902 ± 0.0013 | 0.9911 | **0.9885** | 0.9902 | 0.853 | 118 | 0.036 | **0.0030** | 296 k | 1.8 |
| `resnet18@64+mlp` | ResNet-18, scratch, MLP 512 | **0.9894 ± 0.0014** | 0.9902 ± 0.0023 | **0.9913** | 0.9884 | 0.9903 | 0.849 | 117 | 0.043 | 0.0050 | 11.5 M | 4.6 |
| `stretch` | small CNN | 0.9889 ± 0.0020 | 0.9904 ± 0.0015 | 0.9907 | 0.9881 | 0.9905 | 0.854 | 115 | **0.035** | 0.0039 | 296 k | 1.0 |
| `stretch+augment+mlp` | small CNN, MLP 256 | 0.9889 ± 0.0018 | 0.9889 ± 0.0018 | 0.9905 | 0.9883 | 0.9889 | 0.853 | 134 | 0.038 | 0.0031 | 339 k | 1.5 |
| `resnet18@64` | ResNet-18, scratch | 0.9884 ± 0.0029 | 0.9883 ± 0.0021 | **0.9913** | 0.9861 | 0.9884 | **0.893** | 140 | 0.046 | 0.0034 | 11.2 M | 3.3 |
| `resnet18-pretrained@64` | ResNet-18, ImageNet, full fine-tune | 0.9872 ± 0.0004 | 0.9888 ± 0.0012 | 0.9904 | 0.9853 | 0.9889 | 0.833 | 134 | 0.049 | 0.0048 | 11.2 M | 3.7 |
| `resnet18-pretrained-frozen-l2@64` | ResNet-18, ImageNet, stem–`layer2` frozen | 0.9870 ± 0.0004 | **0.9907 ± 0.0010** | 0.9900 | 0.9852 | **0.9907** | 0.833 | **112** | 0.054 | 0.0049 | 11.2 M (10.5 M) | 4.5 |
| `letterbox` | small CNN | 0.9858 ± 0.0022 | 0.9896 ± 0.0019 | 0.9896 | 0.9831 | 0.9896 | 0.794 | 125 | 0.036 | 0.0029 | 296 k | 0.8 |
| `resnet18-pretrained-frozen+mlp@64` | ResNet-18, ImageNet, backbone frozen, MLP 512 | 0.8905 ± 0.0082 | 0.9360 ± 0.0029 | 0.9273 | 0.8811 | 0.9361 | 0.357 | 768 | 0.201 | 0.0075 | 11.5 M (301 k) | 5.5 |
| `resnet18-pretrained-frozen@64` | ResNet-18, ImageNet, backbone frozen (linear probe) | 0.8561 ± 0.0025 | 0.9017 ± 0.0015 | 0.9084 | 0.8457 | 0.9008 | 0.286 | 1,179 | 0.336 | 0.0092 | 11.2 M (37 k) | 5.4 |

Macro-F1 over all 61 classes with val images is 0.984–0.989 for the top eight rows, 0–0.004 below the
validated-only figure: the weak classes are slightly harder and much noisier. In every row, macro
precision is higher than macro recall. The models miss small classes more often than they
falsely predict them.

## 3 · Findings

### 3.1 Input: stretch to a square instead of letterboxing

| | Macro-F1 | Accuracy | Worst class |
| --- | --- | --- | --- |
| `stretch` | 0.9889 ± 0.0020 | 0.9904 | 0.854 |
| `letterbox` | 0.9858 ± 0.0022 | 0.9896 | 0.794 |

Stretch wins on every seed, and it also won on the split used before the data audit. It helps most on
ิ and ๓ and costs a little on ฏ, ฐ and ู/ุ. The expected cost was lost aspect-ratio cues for า/ๅ,
but that didn't happen: ๅ got better under stretch. Geometry features (the original log width, log
height and aspect ratio, fed to the head) didn't help either input mode: `stretch+geometry` scored
0.9881 and `letterbox+geometry` 0.9857.

### 3.2 Augmentation: per-class tiers are needed, but the gain is within noise

| | Macro-F1 | Accuracy | Errors |
| --- | --- | --- | --- |
| `stretch` (no augmentation) | 0.9889 ± 0.0020 | 0.9904 | 115 |
| `stretch+augment` (tiered) | 0.9894 ± 0.0013 | 0.9902 | 118 |
| `stretch+augment-uniform` (one profile for every class) | 0.9842 ± 0.0024 | 0.9851 | 178 |
| `stretch+augment-geometry-only` (tiered, no photometric) | 0.9871 ± 0.0004 | 0.9887 | 135 |

Tiered augmentation ties with no augmentation on this val set (+0.0005). Uniform augmentation is
clearly worse (−0.0052, below no augmentation on every seed). Its worst losses are on ๅ, า, ฐ and ๓,
the classes the tiers treat most gently: strong rotation and shear turn one of these glyphs into its
neighbour. Dropping the photometric part (blur, noise, contrast) also costs 0.0023.

Val comes from the same collection as train, so it can't show whether augmentation helps on a
different distribution. The hidden test set may differ more. That, and not the val score, is the
reason to keep augmentation in the recommended model.

### 3.3 Capacity: ResNet-18 doesn't beat the small CNN

`resnet18@64` (0.9884 ± 0.0029) and `stretch+augment` (0.9894 ± 0.0013) use the same augmentation and
tie. ResNet-18 has 38× the parameters and takes 1.8× as long per run: 8.6 s per epoch against 3.7 s,
with CPU-side augmentation as the bottleneck. It is the most seed-sensitive row (SD 0.0029). Its one
consistent advantage is the rare class ๑ (0.923 against 0.863), which has 7 val images, so the
difference is about one image.

### 3.4 Transfer learning: ImageNet weights don't help, and freezing them hurts

| Frozen | Trainable | Macro-F1 | Accuracy | Errors |
| --- | --- | --- | --- | --- |
| nothing (random init, `resnet18@64`) | 11.2 M | 0.9884 ± 0.0029 | 0.9883 | 140 |
| nothing (ImageNet init, full fine-tune) | 11.2 M | 0.9872 ± 0.0004 | 0.9888 | 134 |
| stem, `layer1`, `layer2` | 10.5 M | 0.9870 ± 0.0004 | 0.9907 | 112 |
| whole backbone, MLP head | 301 k | 0.8905 ± 0.0082 | 0.9360 | 768 |
| whole backbone, linear head | 37 k | 0.8561 ± 0.0025 | 0.9017 | 1,179 |

- **ImageNet initialisation doesn't beat random initialisation** (−0.0012, within noise). It does
  make the result much more stable across seeds: SD 0.0004 against 0.0029.
- **The early ImageNet stages transfer.** Freezing the stem through `layer2` costs nothing, and this
  row has the fewest errors and the highest accuracy of any experiment. The saving is small, though:
  most of ResNet-18's weights are in `layer3` and `layer4`.
- **The late ImageNet stages don't transfer.** With the whole backbone frozen, macro-F1 falls by
  0.13. Accuracy falls much less (0.90), so the damage is concentrated in the small classes: 42 of the
  56 validated classes drop below 0.95 F1. The worst are ฉ, ๓, ๑ and ๆ, which differ from their
  neighbours by small loops and tails. An MLP head on the frozen features recovers about a quarter of
  the gap. Both frozen rows were still improving at 40 epochs (best epochs 32 and 36), so they are
  lower bounds. They kept the shared recipe so that the rows differ only in what is frozen.

### 3.5 MLP classifier head: no effect when the backbone is trained

| | Linear head | MLP head | Δ |
| --- | --- | --- | --- |
| small CNN + augment | 0.9894 ± 0.0013 | 0.9889 ± 0.0018 | −0.0005 |
| ResNet-18 scratch | 0.9884 ± 0.0029 | 0.9894 ± 0.0014 | +0.0010 |
| ResNet-18 frozen ImageNet | 0.8561 ± 0.0025 | 0.8905 ± 0.0082 | +0.0344 |

When the convolutional layers are trained, they already make the pooled features linearly separable,
so an extra nonlinear layer has nothing left to fix. The MLP only helps when the features are fixed,
and even then it can't make up for features that weren't learned on this data.

## 4 · Error analysis

### Where the errors are

Validation errors summed over the three seeds (35,988 predictions per configuration):

| Confusion | `stretch+augment` | `resnet18@64+mlp` | frozen linear probe |
| --- | --- | --- | --- |
| า → ๅ | 92 | 126 | 215 |
| ๅ → า | 127 | 91 | 204 |
| ค → ด | 21 | 13 | — |
| ้ → ั | 14 | 17 | — |
| **า/ๅ share of all errors** | **62%** (219 / 354) | **62%** (217 / 351) | 12% (419 / 3,538) |

า and ๅ are nearly the same stroke. In isolation, they are separated mainly by height and by how
the stroke sits relative to the baseline, which a tight crop mostly removes. Both top models make the
same number of า/ๅ errors and differ only in which direction they lean. Other confusions are small
and mostly between glyphs that differ by one loop or tail (ค/ด, ข/บ, ด/ต) or between marks (้/ั).

In the notebook's deep dive (`stretch`, seed 42), several of the most confident errors (p ≥ 0.99) are
images that a ruling moved to another class, where the model predicts the original folder: for example,
`209/…` images relabelled ้ and predicted ั. Others come from the same few filename groups (`be_002`,
`be_016`, `be_018`, `bc_001`). Some of the remaining error may be label noise, not model error; these
images are worth a second review.

### Weakest validated classes

Per-class F1, mean over seeds:

| Class | Val images | `stretch` | `stretch+augment` | `resnet18@64` | `resnet18@64+mlp` | ResNet ImageNet fine-tune |
| --- | --- | --- | --- | --- | --- | --- |
| ๑ | 7 | 0.863 | 0.863 | 0.923 | 0.889 | 0.833 |
| ๅ | 405 | 0.914 | 0.907 | 0.893 | 0.912 | 0.897 |
| ุ | 17 | 0.936 | 0.933 | 0.951 | 0.953 | 0.944 |
| า | 713 | 0.945 | 0.947 | 0.935 | 0.945 | 0.938 |
| ู | 26 | 0.941 | 0.949 | 0.946 | 0.927 | 0.936 |
| ้ | 192 | 0.977 | 0.982 | 0.969 | 0.976 | 0.976 |

Every other validated class is at or above 0.97 F1 in the top seven rows, except ๓ under frozen-`layer2`
(0.939); under `letterbox`, ๓ drops to 0.909. Four or five classes are below 0.95 in each of the top
eight rows, and they are the same classes: the า/ๅ pair and small, similar glyphs with few val images.

## 5 · Recommendation

Use **`stretch+augment`**: small CNN, 32 px stretch input, tiered augmentation, `lr=3e-3`.

- It ties for the best macro-F1 (0.9894) and accuracy (0.9902).
- It is among the best calibrated (ECE 0.0030, against 0.0050 for the ResNet) and is 38× smaller and
  2.5× faster to train than the ResNet-18 that ties with it.
- It keeps augmentation. On this val set that is a tie, but it is the cheaper way to hedge against a
  hidden test set that differs from the training collection. This is a judgement call, not a measured
  result.

Choosing one seed because it scored best on val (seed 137, 0.9907) adds about 0.001 of optimism.
Treat 0.989 ± 0.001 as the expected val-level score. Expect the hidden test score to be lower by an
unknown amount.

### What would move the score next

1. **Separate า and ๅ.** They account for 62% of the errors. Experiments to try: a crop that keeps
   the glyph's position and size on the original sheet, or geometry features specific to this pair.
   Plain geometry features didn't help.
2. **Continue the label review** on the most confident errors. Several of them involve ruled
   relabels, where either the ruling or the model is wrong.
3. **Get val data for the 11 not-validated classes.** Nothing here says how well they work.

## 6 · Limitations

- **There is no test set.** Val was used for early stopping and for choosing between experiments.
- **Three seeds per row.** Differences smaller than about 0.003 macro-F1 aren't meaningful. A 0.001
  difference in which row ranks first is noise.
- **Val shares the collection with train.** Held-out filename groups aren't verified to be different
  writers, and 1,000 byte-identical glyphs appear on both sides of the split (reported, not removed).
  Val is likely to be easier than unseen handwriting.
- **The frozen rows used the shared recipe.** Their scores are lower bounds on what frozen ImageNet
  features can do.
- **The weak classes have fewer than 5 val images.** Their F1 is close to noise, and they are left out
  of the headline.

## Appendix · Run ids

All runs are on split `d0a0d7ada85a` and train version `1.p2`, with 40 max epochs. Each list gives the
seeds in order 42 / 137 / 271.

| Experiment | Overrides on `DEFAULT_CONFIG` | Runs |
| --- | --- | --- |
| `stretch` | — | `28d32bd7713a` `fc42a6285144` `c324b6b57c75` |
| `letterbox` | `preprocess="letterbox"` | `22844fd09224` `e5f072a4d928` `7c1602d6f217` |
| `stretch+augment` | `augment=True` | `1e94eb59c37d` `c80f69cf5af9` `f4747ccb4bb3` |
| `stretch+augment+mlp` | `augment=True, mlp_hidden=[256]` | `1374c4ef49c7` `2be925402f3b` `65680c09345a` |
| `resnet18@64` | `model="resnet18", img_size=64, augment=True, lr=1e-3` | `d12f65132771` `95033e7c4f6d` `bad6a5679718` |
| `resnet18@64+mlp` | as above + `mlp_hidden=[512]` | `444b12fe23f7` `6d049189118d` `1c36e791ed9e` |
| `resnet18-pretrained@64` | as `resnet18@64` + `pretrained="IMAGENET1K_V1"` | `583115bf51ce` `a23d54ff1e27` `243eb9a86def` |
| `resnet18-pretrained-frozen-l2@64` | as above + `freeze_through="layer2"` | `739c97f68631` `b5838415926c` `367dc628e3b3` |
| `resnet18-pretrained-frozen@64` | as above + `freeze_through="layer4"` | `fa72fa08ea40` `100c904a7306` `28bf256d2632` |
| `resnet18-pretrained-frozen+mlp@64` | as above + `mlp_hidden=[512]` | `c1658c311c74` `aa2189c2878e` `9508d589e92e` |

The retired rows in §3.1 and §3.2 (`stretch+geometry`, `letterbox+geometry`, `stretch+augment-uniform`,
`stretch+augment-geometry-only`) are on the same split and train version. They are documented in
`reports/training-decisions.md`.

The extra metrics (macro precision and recall, weighted F1, NLL, ECE, confusion sums) are computed from
each run's `confusion.npy` and `val_predictions.csv`. Accuracy and macro-F1 are read from `metrics.json`
and match the notebook's leaderboard.
