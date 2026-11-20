"""Native-resolution overlap-tile prediction with no test-time label access."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from crackvit.data.dataset import preprocess
from crackvit.evaluation.runner import restored_model
from crackvit.utils import seed_everything, sha256_file, write_json


def starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile <= 0 or not 0 <= overlap < tile:
        raise ValueError("Tile must be positive and 0 <= overlap < tile")
    positions = list(range(0, max(1, length - tile + 1), tile - overlap))
    end = max(0, length - tile)
    if positions[-1] != end:
        positions.append(end)
    return positions


@torch.inference_mode()
def tiled_probabilities(model, image: Image.Image, stats: dict, tile=256, overlap=64,
                        device="cpu") -> np.ndarray:
    w, h = image.size
    values, weights = np.zeros((h, w), np.float64), np.zeros((h, w), np.float64)
    model.eval()
    for top in starts(h, tile, overlap):
        for left in starts(w, tile, overlap):
            tw, th = min(tile, w - left), min(tile, h - top)
            crop = image.crop((left, top, left + tw, top + th))
            x = preprocess(crop, th, tw, stats).unsqueeze(0).to(device)
            probability = model(x).sigmoid()[0, 0].cpu().numpy()
            if not np.isfinite(probability).all():
                raise FloatingPointError("Nonfinite model probabilities during inference")
            # Positive cosine blending avoids seams and zero-weight boundary pixels.
            window = np.outer(np.hanning(th + 2)[1:-1], np.hanning(tw + 2)[1:-1])
            values[top:top + th, left:left + tw] += probability * window
            weights[top:top + th, left:left + tw] += window
    if np.any(weights <= 0):
        raise RuntimeError("Inference tiling left uncovered pixels")
    return (values / weights).astype(np.float32)


def infer(checkpoint: Path, image_path: Path, output: Path, device="cpu", tile=256,
          overlap=64, threshold=None) -> dict:
    model, ck, cfg = restored_model(checkpoint, device)
    seed_everything(cfg.train.seed, cfg.train.threads)
    threshold = cfg.threshold if threshold is None else threshold
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must lie in [0,1]")
    with Image.open(image_path) as f:
        image = f.convert("RGB")
    probability = tiled_probabilities(model, image, ck["stats"], tile, overlap, device)
    mask = probability >= threshold
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "probabilities.npy", probability)
    Image.fromarray((mask * 255).astype(np.uint8)).save(output / "mask.png")
    rgb = np.asarray(image).copy()
    rgb[mask] = (0.5 * rgb[mask] + 0.5 * np.array([255, 30, 90])).astype(np.uint8)
    Image.fromarray(rgb).save(output / "overlay.png")
    result = {"image_sha256": sha256_file(image_path), "checkpoint_sha256": sha256_file(checkpoint),
              "threshold": threshold, "tile": tile, "overlap": overlap,
              "height": probability.shape[0], "width": probability.shape[1],
              "note": "Predictions are model outputs, not annotations or measured performance"}
    write_json(output / "inference.json", result)
    return result
