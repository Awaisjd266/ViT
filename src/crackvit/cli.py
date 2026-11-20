"""One CLI connecting acquisition, audited training, inference and reporting."""
import argparse
import json
import logging
import sys
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crackvit", description="Auditable pavement crack segmentation")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Verify imported versions against dependency metadata and pins")
    d = commands.add_parser("download", help="Download original data (license acknowledgment required)")
    d.add_argument("dataset", choices=["deepcrack", "crack500"])
    d.add_argument("--destination", type=Path, default=Path("datasets"))
    d.add_argument("--accept-terms", action="store_true")
    d = commands.add_parser("extract")
    d.add_argument("archive", type=Path)
    d.add_argument("destination", type=Path)
    d = commands.add_parser("extract-crack500")
    d.add_argument("archive", type=Path)
    d.add_argument("destination", type=Path)
    d = commands.add_parser("prepare", help="Build original-image manifests")
    d.add_argument("dataset", choices=["deepcrack", "crack500", "crack500-patches"])
    d.add_argument("--root", type=Path, required=True)
    d.add_argument("--output", type=Path, required=True)
    d.add_argument("--seed", type=int, default=42)
    d = commands.add_parser("audit")
    d.add_argument("--manifest", type=Path, required=True)
    d.add_argument("--root", type=Path, required=True)
    d.add_argument("--output", type=Path, default=Path("reports/data_audit.json"))
    d = commands.add_parser("train")
    d.add_argument("--config", type=Path, required=True)
    d.add_argument("--resume", type=Path)
    for name in ["evaluate", "threshold"]:
        d = commands.add_parser(name)
        d.add_argument("--checkpoint", type=Path, required=True)
        d.add_argument("--manifest", type=Path, required=True)
        d.add_argument("--root", type=Path, required=True)
        d.add_argument("--output", type=Path, required=True)
        d.add_argument("--device", default="cpu")
        if name == "evaluate":
            d.add_argument("--split", choices=["val", "test"], default="test")
            d.add_argument("--threshold", type=float)
            d.add_argument("--bootstrap", type=int, default=1000)
            d.add_argument("--allow-cross-domain", action="store_true")
    d = commands.add_parser("infer")
    d.add_argument("--checkpoint", type=Path, required=True)
    d.add_argument("--image", type=Path, required=True)
    d.add_argument("--output", type=Path, required=True)
    d.add_argument("--device", default="cpu")
    d.add_argument("--tile", type=int, default=256)
    d.add_argument("--overlap", type=int, default=64)
    d.add_argument("--threshold", type=float)
    d = commands.add_parser("benchmark")
    d.add_argument("--checkpoint", type=Path, required=True)
    d.add_argument("--output", type=Path, required=True)
    d.add_argument("--device", default="cpu")
    d.add_argument("--warmup", type=int, default=5)
    d.add_argument("--iterations", type=int, default=30)
    d = commands.add_parser("report")
    d.add_argument("--run", type=Path, required=True)
    d.add_argument("--evaluation", type=Path, required=True)
    d.add_argument("--output", type=Path, required=True)
    d = commands.add_parser("compare")
    d.add_argument("a", type=Path, help="Evaluation per_image.json")
    d.add_argument("b", type=Path)
    d.add_argument("--output", type=Path, required=True)
    d.add_argument("--iterations", type=int, default=1000)
    d = commands.add_parser("tune")
    d.add_argument("--config", type=Path, required=True)
    d.add_argument("--output", type=Path, required=True)
    d.add_argument("--learning-rates", nargs="+", type=float, default=[0.0001, 0.0003, 0.001])
    d = commands.add_parser("suite")
    d.add_argument("--config", type=Path, required=True)
    d.add_argument("--output", type=Path, required=True)
    d.add_argument("--models", nargs="+", default=["vit", "cnn", "resnet", "fcn", "unet"])
    d.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    d.add_argument("--ablations", action="store_true")
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
        if result is not None:
            print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        logging.error("%s: %s", type(exc).__name__, exc)
        return 1


def dispatch(a):
    if a.command == "doctor":
        from crackvit.utils import check_environment
        return check_environment()
    if a.command == "download":
        from crackvit.data.acquire import CRACK500_DRIVE_ID, acquire_deepcrack
        if not a.accept_terms:
            raise ValueError("Review upstream terms and use --accept-terms for noncommercial research")
        if a.dataset == "deepcrack":
            return {"root": str(acquire_deepcrack(a.destination, True))}
        import gdown
        from crackvit.utils import sha256_file, write_json
        dest = a.destination / "downloads" / "pavement-crack-datasets.zip"
        dest.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=CRACK500_DRIVE_ID, output=str(dest), resume=True)
        result = {"source": CRACK500_DRIVE_ID, "sha256": sha256_file(dest),
                  "bytes": dest.stat().st_size,
                  "note": "Source bundle includes other datasets; extract original CRACK500 only"}
        write_json(dest.with_suffix(".provenance.json"), result)
        return result
    if a.command == "extract":
        from crackvit.data.acquire import safe_extract
        safe_extract(a.archive, a.destination)
        return {"extracted": str(a.destination)}
    if a.command == "extract-crack500":
        from crackvit.data.acquire import extract_crack500_bundle
        return extract_crack500_bundle(a.archive, a.destination)
    if a.command == "prepare":
        from crackvit.data.manifest import prepare_crack500, prepare_crack500_crops, prepare_deepcrack
        if a.dataset == "deepcrack":
            samples = prepare_deepcrack(a.root, a.output, a.seed)
        elif a.dataset == "crack500-patches":
            samples = prepare_crack500_crops(a.root, a.output)
        else:
            samples = prepare_crack500(a.root, a.output)
        return {"manifest": str(a.output), "samples": len(samples)}
    if a.command == "audit":
        from crackvit.data.manifest import audit_manifest
        result = audit_manifest(a.manifest, a.root, a.output)
        if not result["passed"]:
            raise ValueError(f"Audit failed; details: {a.output}")
        return {k: v for k, v in result.items() if k != "records"}
    if a.command == "train":
        from crackvit.config import load_config
        from crackvit.training import train
        return train(load_config(a.config), a.resume)
    if a.command == "evaluate":
        from crackvit.evaluation.runner import evaluate
        return evaluate(a.checkpoint, a.manifest, a.root, a.output, a.split, a.device,
                        a.threshold, a.bootstrap, a.allow_cross_domain)
    if a.command == "threshold":
        from crackvit.evaluation.runner import select_threshold
        return select_threshold(a.checkpoint, a.manifest, a.root, a.output, device=a.device)
    if a.command == "infer":
        from crackvit.inference import infer
        return infer(a.checkpoint, a.image, a.output, a.device, a.tile, a.overlap, a.threshold)
    if a.command == "benchmark":
        from crackvit.evaluation.runner import benchmark
        return benchmark(a.checkpoint, a.output, a.device, a.warmup, a.iterations)
    if a.command == "report":
        from crackvit.reporting import plot_run
        plot_run(a.run, a.evaluation, a.output)
        return {"report_directory": str(a.output)}
    if a.command == "compare":
        from crackvit.evaluation.metrics import paired_bootstrap
        from crackvit.utils import write_json
        result = paired_bootstrap(json.loads(a.a.read_text()), json.loads(a.b.read_text()), a.iterations)
        write_json(a.output, result)
        return result
    if a.command == "suite":
        from crackvit.config import load_config
        from crackvit.experiments import suite
        return suite(load_config(a.config), a.output, a.models, a.seeds, a.ablations)
    if a.command == "tune":
        from crackvit.config import load_config
        from crackvit.tuning import tune
        return tune(load_config(a.config), a.output, a.learning_rates)
    raise ValueError("Unknown command")


if __name__ == "__main__":
    sys.exit(main())
