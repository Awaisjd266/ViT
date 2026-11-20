"""Reproducibility, atomic artifacts and environment capture."""
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import PIL


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: str | Path, data: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def seed_everything(seed: int, threads: int = 4) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def environment() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
        "numpy": np.__version__, "pillow": PIL.__version__,
        "torch_threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda, "git_commit": commit,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def check_environment() -> dict:
    """Compare imported code with dependency metadata as well as declared pins."""
    import importlib.metadata as metadata
    import matplotlib
    import torchvision
    import yaml
    modules = {"torch": (torch.__version__, "2.6.0"),
               "torchvision": (torchvision.__version__, "0.21.0"),
               "numpy": (np.__version__, "2.2.6"), "Pillow": (PIL.__version__, "11.2.1"),
               "matplotlib": (matplotlib.__version__, "3.10.3"),
               "PyYAML": (yaml.__version__, "6.0.2")}
    checks = {name: {"imported": str(actual), "metadata": metadata.version(name), "expected": expected,
                     "passed": str(actual).split("+")[0] == expected and
                     str(actual) == metadata.version(name)} for name, (actual, expected) in modules.items()}
    result = {"packages": checks, "passed": all(x["passed"] for x in checks.values())}
    if not result["passed"]:
        raise RuntimeError(f"Dependency version or installed-code mismatch: {result}")
    return result
