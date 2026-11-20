"""Defined comparators, not claims of recovering unspecified manuscript baselines."""
import torch
import torch.nn.functional as F
from torch import nn

from crackvit.models.vit import refinement


class ConvSeg(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.network = nn.Sequential(refinement(3, base), nn.MaxPool2d(2),
                                     refinement(base, base * 2), nn.MaxPool2d(2),
                                     refinement(base * 2, base * 4),
                                     nn.Conv2d(base * 4, 1, 1))

    def forward(self, x):
        return F.interpolate(self.network(x), x.shape[-2:], mode="bilinear", align_corners=False)


class UNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.down = nn.ModuleList([refinement(3, base), refinement(base, 2 * base),
                                   refinement(2 * base, 4 * base)])
        self.bridge = refinement(4 * base, 8 * base)
        self.up = nn.ModuleList([refinement(12 * base, 4 * base),
                                 refinement(6 * base, 2 * base), refinement(3 * base, base)])
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bridge(x)
        for block, skip in zip(self.up, reversed(skips)):
            x = F.interpolate(x, skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        return self.head(x)


class ResNetSeg(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        backbone = resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.head = nn.Sequential(refinement(512, 128), nn.Conv2d(128, 1, 1))

    def forward(self, x):
        return F.interpolate(self.head(self.features(x)), x.shape[-2:], mode="bilinear", align_corners=False)
