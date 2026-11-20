from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from crackvit.cli import main
from crackvit.config import from_dict
from crackvit.evaluation.runner import benchmark, evaluate, select_threshold
from crackvit.inference import infer, starts, tiled_probabilities
from crackvit.reporting import plot_run
from crackvit.storage import registry
from crackvit.training import train


def test_tiling_covers_edges_and_small_images():
    class Constant(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1, *x.shape[-2:])
    for size in [(17, 19), (73, 45)]:
        image = Image.new("RGB", size)
        result = tiled_probabilities(Constant(), image, {"mean": [0]*3, "std": [1]*3}, 32, 8)
        assert result.shape == (size[1], size[0])
        np.testing.assert_allclose(result, .5)
    with pytest.raises(ValueError):
        starts(100, 32, 32)


def test_end_to_end_train_reload_eval_threshold_infer_report(dataset_fixture, tmp_path):
    root, manifest, _ = dataset_fixture
    cfg = from_dict({"data": {"manifest": str(manifest), "root": str(root), "height": 32, "width": 40},
                     "model": {"embed_dim": 24, "depth": 1, "heads": 3, "patch_size": 8,
                               "decoder_channels": 16, "dropout": 0.},
                     "train": {"epochs": 2, "batch_size": 2, "threads": 2},
                     "output": str(tmp_path / "run"), "purpose": "smoke"})
    summary = train(cfg)
    run = Path(cfg.output)
    assert summary["epochs_executed"] == 2 and (run / "best.pt").exists()
    result = evaluate(run / "best.pt", manifest, root, run / "test", bootstrap=10)
    assert result["images"] == 1 and result["purpose"] == "smoke"
    threshold = select_threshold(run / "best.pt", manifest, root, run / "threshold.json")
    assert threshold["selection_split"] == "val"
    original_mask = (root / "2_mask.png").read_bytes()
    Image.new("L", (40, 32)).save(root / "2_mask.png")
    with pytest.raises(ValueError, match="Validation data changed"):
        select_threshold(run / "best.pt", manifest, root, run / "bad_threshold.json")
    (root / "2_mask.png").write_bytes(original_mask)
    prediction = infer(run / "best.pt", root / "3.png", run / "infer", tile=32, overlap=8)
    assert prediction["width"] == 40
    assert (run / "infer" / "overlay.png").exists()
    timing = benchmark(run / "best.pt", run / "timing.json", warmup=1, iterations=2)
    assert timing["fps_mean"] > 0
    plot_run(run, run / "test", run / "figures")
    assert (run / "figures" / "learning_curves.png").exists()
    with registry(run / "experiments.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM epochs").fetchone()[0] == 2
    with pytest.raises(FileExistsError):
        train(cfg)
    # A complete run can be restored without fabricating further epochs.
    assert train(cfg, run / "last.pt")["epochs_executed"] == 2
    cfg.train.learning_rate *= 2
    with pytest.raises(ValueError):
        train(cfg, run / "last.pt")


def test_cli_help_and_missing_config(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0 and "train" in capsys.readouterr().out
    assert main(["train", "--config", "does-not-exist.yaml"]) == 1


def test_interrupted_resume_matches_uninterrupted(dataset_fixture, tmp_path, monkeypatch):
    from copy import deepcopy
    import crackvit.training as training
    root, manifest, _ = dataset_fixture
    cfg = from_dict({"data": {"manifest": str(manifest), "root": str(root), "height": 32, "width": 40},
                     "model": {"embed_dim": 24, "depth": 1, "heads": 3, "patch_size": 8,
                               "decoder_channels": 16, "dropout": 0.1},
                     "train": {"epochs": 3, "batch_size": 2, "threads": 2},
                     "output": str(tmp_path / "complete"), "purpose": "smoke"})
    train(cfg)
    resumed = deepcopy(cfg)
    resumed.output = str(tmp_path / "interrupted")
    original = training.run_epoch
    calls = 0
    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("Simulated interruption after completed epoch")
        return original(*args, **kwargs)
    monkeypatch.setattr(training, "run_epoch", interrupt)
    with pytest.raises(RuntimeError, match="Simulated"):
        train(resumed)
    monkeypatch.setattr(training, "run_epoch", original)
    train(resumed, Path(resumed.output) / "last.pt")
    a = training.load_checkpoint(Path(cfg.output) / "last.pt")
    b = training.load_checkpoint(Path(resumed.output) / "last.pt")
    assert a["history"] == b["history"]
    assert all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
