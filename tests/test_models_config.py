import pytest
import torch

from crackvit.config import ExperimentConfig, ModelConfig, from_dict
from crackvit.models import build_model


@pytest.mark.parametrize("name", ["vit", "cnn", "resnet", "fcn", "unet"])
def test_all_models_dense_shape_and_backward(name):
    torch.set_num_threads(2)
    cfg = ModelConfig(name=name, patch_size=8, embed_dim=24, depth=1, heads=3,
                      decoder_channels=16, base_channels=8)
    model = build_model(cfg)
    x = torch.randn(2, 3, 35, 51)
    y = model(x)
    assert y.shape == (2, 1, 35, 51) and torch.isfinite(y).all()
    y.mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_local_skip_and_patch_boundary():
    cfg = ModelConfig(local_skip=True, patch_size=16, embed_dim=24, depth=1, heads=3,
                      decoder_channels=16)
    model = build_model(cfg).eval()
    with torch.inference_mode():
        assert model(torch.randn(1, 3, 33, 65)).shape == (1, 1, 33, 65)


@pytest.mark.parametrize("raw", [{"bogus": 1}, {"model": {"heads": 5}},
                                 {"model": {"patch_size": 7}}, {"train": {"batch_size": 0}},
                                 {"train": {"bce_weight": 0, "dice_weight": 0}}])
def test_invalid_config(raw):
    with pytest.raises(ValueError):
        from_dict(raw)


def test_config_roundtrip():
    c = ExperimentConfig()
    assert from_dict(c.as_dict()) == c
