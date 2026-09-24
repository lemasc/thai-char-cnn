"""Models: a small CNN baseline and ResNet-18.

`small_cnn`: three conv blocks (two 3x3 conv + BN + ReLU, then max-pool), global average pooling,
dropout, one linear head. `use_geometry` concatenates the image's original log width, log height and
aspect before the head: letterboxing a tight ink crop throws away its absolute size, which is the
main cue separating ่ from ๅ/า and ุ from ู.

`resnet18`: torchvision's ResNet-18 with a one-channel stem and the same dropout + linear head.
`pretrained` names a torchvision weights tag (e.g. `"IMAGENET1K_V1"`) or is None for random init.
With weights, the stem's kernel is the ImageNet RGB kernel summed over its input channels, which is
the same as feeding the grey image to all three. The ImageNet stem downsamples 4x, so it is meant
for inputs of 64 px and up.
"""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


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


class ResNet18(nn.Module):
    def __init__(self, n_classes: int, pretrained: str | None = None, dropout: float = 0.3):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights[pretrained] if pretrained else None)
        rgb = net.conv1.weight.detach()
        net.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        if pretrained:
            with torch.no_grad():
                net.conv1.weight.copy_(rgb.sum(1, keepdim=True))
        net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, n_classes))
        self.net = net

    def forward(self, x: torch.Tensor, geo: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(x)


def build_model(cfg: dict, n_classes: int) -> nn.Module:
    if cfg["model"] == "small_cnn":
        assert cfg["pretrained"] is None, "small_cnn has no pretrained weights"
        return SmallCNN(n_classes, width=cfg["width"], dropout=cfg["dropout"], use_geometry=cfg["use_geometry"])
    if cfg["model"] == "resnet18":
        assert not cfg["use_geometry"], "resnet18 does not take geometry features"
        return ResNet18(n_classes, pretrained=cfg["pretrained"], dropout=cfg["dropout"])
    raise ValueError(f"unknown model {cfg['model']!r}")
