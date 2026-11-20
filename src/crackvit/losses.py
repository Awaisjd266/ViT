"""Numerically stable equal-weight BCE and soft Dice segmentation loss."""
import torch
import torch.nn.functional as F
from torch import nn


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=1.0, smooth=1e-6):
        super().__init__()
        if min(bce_weight, dice_weight) < 0 or bce_weight + dice_weight <= 0 or pos_weight <= 0:
            raise ValueError("Invalid loss weights")
        self.bce_weight, self.dice_weight, self.smooth = bce_weight, dice_weight, smooth
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape or targets.ndim != 4 or targets.shape[1] != 1:
            raise ValueError("Expected matching [B,1,H,W] logits and targets")
        probabilities = logits.float().sigmoid().flatten(1)
        truth = targets.float().flatten(1)
        dice = (2 * (probabilities * truth).sum(1) + self.smooth) / (
            probabilities.sum(1) + truth.sum(1) + self.smooth)
        bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float(),
                                                 pos_weight=self.pos_weight)
        return self.bce_weight * bce + self.dice_weight * (1 - dice).mean()
