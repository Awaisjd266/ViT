"""Pre-norm global ViT encoder and progressive dense prediction decoder."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from crackvit.config import ModelConfig


def refinement(in_channels: int, out_channels: int) -> nn.Sequential:
    groups = math.gcd(8, out_channels)
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                         nn.GroupNorm(groups, out_channels), nn.GELU(),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                         nn.GroupNorm(groups, out_channels), nn.GELU())


class ViTSegmenter(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.embed_dim % cfg.heads or cfg.patch_size & (cfg.patch_size - 1):
            raise ValueError("Invalid attention dimension or patch size")
        self.patch_size = cfg.patch_size
        self.patch = nn.Conv2d(3, cfg.embed_dim, cfg.patch_size, stride=cfg.patch_size)
        self.position = nn.Parameter(torch.zeros(1, cfg.embed_dim, 16, 16))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(cfg.embed_dim, cfg.heads,
                                           int(cfg.embed_dim * cfg.mlp_ratio), cfg.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.depth, norm=nn.LayerNorm(cfg.embed_dim),
                                             enable_nested_tensor=False)
        # Avoid identical initial parameters in cloned PyTorch encoder layers.
        for block in self.encoder.layers:
            for name, param in block.named_parameters():
                if param.ndim > 1:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
        self.project = nn.Conv2d(cfg.embed_dim, cfg.decoder_channels, 1)
        stages, channels = [], cfg.decoder_channels
        for _ in range(int(math.log2(cfg.patch_size))):
            next_channels = max(16, channels // 2)
            stages.append(refinement(channels, next_channels))
            channels = next_channels
        self.stages = nn.ModuleList(stages)
        self.local = refinement(3, channels) if cfg.local_skip else None
        self.head = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        p = self.patch_size
        padded = F.pad(x, (0, (-w) % p, 0, (-h) % p), mode="replicate")
        grid = self.patch(padded)
        gh, gw = grid.shape[-2:]
        pos = F.interpolate(self.position, size=(gh, gw), mode="bicubic", align_corners=False)
        tokens = (grid + pos).flatten(2).transpose(1, 2)
        features = self.encoder(tokens).transpose(1, 2).reshape(x.shape[0], -1, gh, gw)
        features = self.project(features)
        for block in self.stages:
            features = block(F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False))
        if self.local is not None:
            features = features + self.local(padded)
        return self.head(features)[..., :h, :w]
