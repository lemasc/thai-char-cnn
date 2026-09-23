"""A small CNN baseline.

Three conv blocks (two 3x3 conv + BN + ReLU, then max-pool), global average pooling, dropout, one
linear head. `use_geometry` concatenates the image's original log width, log height and aspect
before the head: letterboxing a tight ink crop throws away its absolute size, which is the main
cue separating ่ from ๅ/า and ุ from ู.
"""

import torch
from torch import nn


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int, width: int = 32, dropout: float = 0.3,
                 use_geometry: bool = False, n_geo: int = 3):
        super().__init__()
        self.use_geometry = use_geometry
        self.features = nn.Sequential(_block(1, width), _block(width, 2 * width), _block(2 * width, 4 * width),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.geo = nn.Sequential(nn.Linear(n_geo, 16), nn.ReLU(inplace=True)) if use_geometry else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * width + (16 if use_geometry else 0), n_classes))

    def forward(self, x: torch.Tensor, geo: torch.Tensor | None = None) -> torch.Tensor:
        h = self.features(x)
        if self.use_geometry:
            h = torch.cat([h, self.geo(geo)], 1)
        return self.head(h)


def build_model(cfg: dict, n_classes: int) -> nn.Module:
    assert cfg["model"] == "small_cnn", f"unknown model {cfg['model']!r}"
    return SmallCNN(n_classes, width=cfg["width"], dropout=cfg["dropout"], use_geometry=cfg["use_geometry"])
