"""Explicit segmentation model factory; no implicit pretrained downloads."""
import torch.nn as nn

from crackvit.config import ModelConfig
from crackvit.models.baselines import ConvSeg, ResNetSeg, UNet
from crackvit.models.vit import ViTSegmenter


def build_model(config: ModelConfig):
    if config.name == "vit":
        return ViTSegmenter(config)
    if config.name == "cnn":
        return ConvSeg(config.base_channels)
    if config.name == "unet":
        return UNet(config.base_channels)
    if config.name == "resnet":
        return ResNetSeg()
    if config.name == "fcn":
        from torchvision.models.segmentation import fcn_resnet50
        return FCNAdapter(fcn_resnet50(weights=None, weights_backbone=None, num_classes=1))
    raise ValueError(f"Unsupported model {config.name}")


class FCNAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]
