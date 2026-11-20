"""Strict, versionable experiment configuration.

Hyperparameters absent from AS.docx are explicit engineering choices, not recovered settings.
"""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    manifest: str = "datasets/manifests/deepcrack.csv"
    root: str = "datasets/raw/deepcrack"
    height: int = 384
    width: int = 544
    augment: bool = True
    workers: int = 0
    max_train: int | None = None
    max_val: int | None = None


@dataclass
class ModelConfig:
    name: str = "vit"
    patch_size: int = 16
    embed_dim: int = 192
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    decoder_channels: int = 128
    local_skip: bool = False
    base_channels: int = 32


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    pos_weight: float = 1.0
    clip_grad: float = 1.0
    patience: int = 20
    amp: bool = False
    seed: int = 42
    device: str = "cpu"
    threads: int = 4


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    threshold: float = 0.5
    output: str = "runs/deepcrack-vit"
    purpose: str = "research"

    def validate(self) -> None:
        d, m, t = self.data, self.model, self.train
        if min(d.height, d.width, m.patch_size, m.embed_dim, m.depth, m.heads,
               m.decoder_channels, m.base_channels, t.epochs, t.batch_size, t.threads) <= 0:
            raise ValueError("Dimensions, epochs, batch size and threads must be positive")
        if m.embed_dim % m.heads or m.patch_size & (m.patch_size - 1):
            raise ValueError("embed_dim must divide heads; patch_size must be a power of two")
        if not 0 <= self.threshold <= 1 or not 0 <= m.dropout < 1:
            raise ValueError("Invalid probability or dropout")
        if t.learning_rate <= 0 or t.weight_decay < 0 or t.clip_grad <= 0 or t.pos_weight <= 0:
            raise ValueError("Invalid optimizer or loss parameter")
        if min(t.bce_weight, t.dice_weight) < 0 or t.bce_weight + t.dice_weight <= 0:
            raise ValueError("At least one nonnegative loss weight must be positive")
        if d.workers < 0 or t.patience < 1 or any(
            x is not None and x <= 0 for x in (d.max_train, d.max_val)
        ):
            raise ValueError("Invalid worker, subset or patience setting")
        if m.name not in {"vit", "cnn", "resnet", "fcn", "unet"}:
            raise ValueError(f"Unknown model: {m.name}")
        if self.purpose not in {"research", "smoke"}:
            raise ValueError("purpose must be research or smoke")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def from_dict(raw: dict) -> ExperimentConfig:
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping")
    allowed = {f.name for f in fields(ExperimentConfig)}
    if set(raw) - allowed:
        raise ValueError(f"Unknown configuration keys: {set(raw) - allowed}")
    raw = dict(raw)
    for name, cls in (("data", DataConfig), ("model", ModelConfig), ("train", TrainConfig)):
        if name in raw:
            if not isinstance(raw[name], dict) or set(raw[name]) - {f.name for f in fields(cls)}:
                raise ValueError(f"Unknown or invalid {name} settings")
            raw[name] = cls(**raw[name])
    cfg = ExperimentConfig(**raw)
    cfg.validate()
    return cfg


def load_config(path: str | Path) -> ExperimentConfig:
    return from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
