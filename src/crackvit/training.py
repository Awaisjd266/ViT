"""Audited training, validation selection and resumable experiment artifacts."""
import logging
import os
import random
import uuid
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from crackvit.config import ExperimentConfig
from crackvit.data.dataset import CrackDataset, training_statistics
from crackvit.data.manifest import audit_manifest, read_manifest
from crackvit.evaluation.metrics import Confusion
from crackvit.losses import BCEDiceLoss
from crackvit.models import build_model
from crackvit.storage import finish_run, record_epoch, register_run
from crackvit.utils import environment, seed_everything, seed_worker, sha256_file, write_json

LOG = logging.getLogger(__name__)


def atomic_checkpoint(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device="cpu") -> dict:
    # PyTorch safe loader; only use trusted project checkpoints.
    return torch.load(path, map_location=device, weights_only=True)


def make_loader(samples, root, cfg, stats, train=False, generator=None):
    return DataLoader(CrackDataset(samples, root, cfg.data.height, cfg.data.width, stats,
                                  augment=train and cfg.data.augment),
                      batch_size=cfg.train.batch_size, shuffle=train,
                      num_workers=cfg.data.workers, pin_memory=cfg.train.device.startswith("cuda"),
                      worker_init_fn=seed_worker, generator=generator)


def run_epoch(model, loader, loss_fn, device, threshold, optimizer=None, scaler=None,
              clip_grad=1.0, amp=False):
    train = optimizer is not None
    model.train(train)
    total, images, confusion = 0.0, 0, Confusion()
    for batch in loader:
        x, y = batch["image"].to(device), batch["mask"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
            loss = loss_fn(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
        total += float(loss.detach()) * len(x)
        images += len(x)
        confusion.update(logits.detach().sigmoid() >= threshold, y > 0.5)
    return {"loss": total / images, **confusion.metrics()}


def train(cfg: ExperimentConfig, resume: Path | None = None) -> dict:
    cfg.validate()
    seed_everything(cfg.train.seed, cfg.train.threads)
    device = torch.device(cfg.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if cfg.train.amp and device.type != "cuda":
        raise ValueError("AMP is only enabled for CUDA experiments")
    out, root = Path(cfg.output), Path(cfg.data.root)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "last.pt").exists() and resume is None:
        raise FileExistsError("Run already exists; choose a new output or explicitly resume")
    audit = audit_manifest(Path(cfg.data.manifest), root, out / "data_audit.json")
    if not audit["passed"]:
        raise ValueError("Dataset audit failed; inspect data_audit.json")
    samples = read_manifest(Path(cfg.data.manifest))
    train_samples = [s for s in samples if s.split == "train"]
    val_samples = [s for s in samples if s.split == "val"]
    if cfg.data.max_train:
        train_samples = train_samples[:cfg.data.max_train]
    if cfg.data.max_val:
        val_samples = val_samples[:cfg.data.max_val]
    selected = train_samples + val_samples
    stats = training_statistics(selected, root, cfg.data.height, cfg.data.width)
    # IDs are not enough: bind the run to actual decoded/image-mask checksums.
    identity = [{k: r[k] for k in ("sample_id", "split", "image_sha256", "mask_sha256", "decoded_sha256")}
                for r in audit["records"]]
    stats["manifest_sha256"] = audit["manifest_sha256"]
    generator = torch.Generator().manual_seed(cfg.train.seed)
    model = build_model(cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate,
                                 weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    loss_fn = BCEDiceLoss(cfg.train.bce_weight, cfg.train.dice_weight, cfg.train.pos_weight).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp)
    start, best, bad_epochs, history = 0, -1.0, 0, []
    run_id, db = uuid.uuid4().hex, out / "experiments.sqlite3"
    if resume:
        ck = load_checkpoint(resume, device)
        if ck["config"] != cfg.as_dict() or ck["data_identity"] != identity:
            raise ValueError("Resume config or data identity differs from checkpoint")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start, best, bad_epochs = ck["epoch"] + 1, ck["best_f1"], ck["bad_epochs"]
        history, run_id = ck["history"], ck["run_id"]
        generator.set_state(ck["loader_rng"].cpu())
        torch.set_rng_state(ck["torch_rng"].cpu())
        random.setstate(ck["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
    else:
        register_run(db, run_id, cfg.as_dict())
    write_json(out / "config.json", cfg.as_dict())
    write_json(out / "environment.json", environment())
    write_json(out / "normalization.json", stats)
    write_json(out / "sample_selection.json", {"train": [s.sample_id for s in train_samples],
                                               "val": [s.sample_id for s in val_samples]})
    train_loader = make_loader(train_samples, root, cfg, stats, True, generator)
    val_loader = make_loader(val_samples, root, cfg, stats)
    status = "failed"
    try:
        for epoch in range(start, cfg.train.epochs):
            tr = run_epoch(model, train_loader, loss_fn, device, cfg.threshold, optimizer,
                           scaler, cfg.train.clip_grad, cfg.train.amp)
            va = run_epoch(model, val_loader, loss_fn, device, cfg.threshold)
            score = va["f1"] if va["f1"] is not None else -1.0
            improved = score > best
            if improved:
                best, bad_epochs = score, 0
            else:
                bad_epochs += 1
            row = {"epoch": epoch + 1, "train_loss": tr["loss"], "val_loss": va["loss"],
                   "val_f1": va["f1"], "val_miou": va["miou"],
                   "learning_rate": optimizer.param_groups[0]["lr"]}
            history.append(row)
            record_epoch(db, run_id, epoch + 1, tr["loss"], va["loss"], va["f1"])
            scheduler.step()
            payload = {"format_version": 1, "config": cfg.as_dict(), "stats": stats,
                       "data_identity": identity, "run_id": run_id, "epoch": epoch,
                       "best_f1": best, "bad_epochs": bad_epochs, "history": history,
                       "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                       "loader_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                       "python_rng": random.getstate(),
                       "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
            atomic_checkpoint(out / "last.pt", payload)
            if improved:
                atomic_checkpoint(out / "best.pt", payload)
            write_json(out / "history.json", history)
            LOG.info("epoch=%d train_loss=%.5f val_loss=%.5f val_f1=%s", epoch + 1,
                     tr["loss"], va["loss"], va["f1"])
            if bad_epochs >= cfg.train.patience:
                break
        status = "completed"
    finally:
        finish_run(db, run_id, status)
    summary = {"run_id": run_id, "status": status, "purpose": cfg.purpose,
               "epochs_executed": len(history), "best_validation_f1": best,
               "training_samples": len(train_samples), "validation_samples": len(val_samples),
               "parameters": sum(p.numel() for p in model.parameters()),
               "checkpoint_sha256": sha256_file(out / "best.pt") if (out / "best.pt").exists() else None,
               "benchmark_reproduction": False}
    write_json(out / "summary.json", summary)
    return summary
