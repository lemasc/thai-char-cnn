"""Training-time augmentation -- the seam for the (deferred) augmentation preview.

Nothing here has been checked against pixels yet, so the defaults are deliberately conservative
and the riskier ops (stroke width, elastic) are implemented but off. The preview notebook will
import `AugmentConfig`, `load_augment_config` and `build_train_transform`, and write only
`configs/augment.json`; training code never changes when that file does.

No flips: a mirrored Thai glyph is a different glyph or none at all.
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms import v2

WHITE = 255


@dataclass(frozen=True)
class AugmentConfig:
    rotation_deg: float = 8.0          # +/- degrees
    shear_deg: float = 5.0             # +/- degrees, x only
    scale_min: float = 0.9
    scale_max: float = 1.1
    translate_frac: float = 0.04       # of the image side
    stroke_p: float = 0.0              # probability of a 3x3 erode or dilate of the ink
    elastic_p: float = 0.0             # probability of an elastic distortion
    elastic_alpha: float = 8.0
    elastic_sigma: float = 3.0


def load_augment_config(path: Path) -> AugmentConfig:
    """`configs/augment.json` when present (unknown keys like `note` are ignored), else defaults."""
    if not path.exists():
        return AugmentConfig()
    known = {f.name for f in fields(AugmentConfig)}
    return AugmentConfig(**{k: v for k, v in json.loads(path.read_text()).items() if k in known})


def augment_params(cfg: AugmentConfig) -> dict:
    """What a run hashes. The code that applies them is covered by the run's `code_hash`."""
    return asdict(cfg)


class RandomStroke(torch.nn.Module):
    """Thicken or thin the ink by one pixel. Ink is dark, so thickening is a min-filter."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0 or torch.rand(()) >= self.p:
            return x
        f = x.float().unsqueeze(0)
        if torch.rand(()) < 0.5:
            f = -F.max_pool2d(-f, 3, stride=1, padding=1)     # dilate ink
        else:
            f = F.max_pool2d(f, 3, stride=1, padding=1)       # erode ink
        return f.squeeze(0).to(x.dtype)


class OnInk(torch.nn.Module):
    """Run a geometric op on the inverted image (ink bright, paper 0), then invert back.

    torchvision blends `fill` in with an interpolated mask after sampling with zero padding, so a
    white fill leaves a faint grey line wherever the image edge lands. With paper at 0 the padding
    *is* the background and the seam disappears.
    """

    def __init__(self, op: torch.nn.Module):
        super().__init__()
        self.op = op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return WHITE - self.op(WHITE - x)


def build_train_transform(cfg: AugmentConfig) -> v2.Compose:
    """uint8 (1, H, W) tensor -> uint8 (1, H, W) tensor. What enters from outside is white paper."""
    ops = [OnInk(v2.RandomAffine(degrees=cfg.rotation_deg,
                                 translate=(cfg.translate_frac, cfg.translate_frac),
                                 scale=(cfg.scale_min, cfg.scale_max),
                                 shear=(-cfg.shear_deg, cfg.shear_deg),
                                 interpolation=v2.InterpolationMode.BILINEAR, fill=0))]
    if cfg.stroke_p > 0:
        ops.append(RandomStroke(cfg.stroke_p))
    if cfg.elastic_p > 0:
        ops.append(v2.RandomApply([OnInk(v2.ElasticTransform(alpha=cfg.elastic_alpha, sigma=cfg.elastic_sigma,
                                                             fill=0))], p=cfg.elastic_p))
    return v2.Compose(ops)
