"""Download authentic upstream archives; never invent benchmark images or labels."""
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from crackvit.utils import sha256_file, write_json

DEEPCRACK_URL = "https://raw.githubusercontent.com/yhlleo/DeepCrack/master/dataset/DeepCrack.zip"
CRACK500_DRIVE_ID = "13_vDYl54Mrd34dddX9w4ppAEiuWv4MlD"


def download(url: str, output: Path, expected_sha256: str | None = None) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "crackvit-research/0.1"})
    with urllib.request.urlopen(req, timeout=120) as response, tmp.open("wb") as f:
        shutil.copyfileobj(response, f)
    digest = sha256_file(tmp)
    if expected_sha256 and digest != expected_sha256:
        tmp.unlink()
        raise ValueError("Archive checksum mismatch")
    if not zipfile.is_zipfile(tmp):
        tmp.unlink()
        raise ValueError("Download is not a ZIP archive (possibly a login or quota page)")
    tmp.replace(output)
    result = {"url": url, "sha256": digest, "bytes": output.stat().st_size}
    write_json(output.with_suffix(".provenance.json"), result)
    return result


def safe_extract(archive: Path, destination: Path, max_bytes: int = 12_000_000_000) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        if sum(m.file_size for m in members) > max_bytes:
            raise ValueError("Archive exceeds extraction size limit")
        for info in members:
            name = info.filename.replace("\\", "/")
            p = PurePosixPath(name)
            mode = info.external_attr >> 16
            if p.is_absolute() or ".." in p.parts or ":" in name or mode & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe archive member: {name}")
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Archive member escapes destination")
        z.extractall(root)


def acquire_deepcrack(destination: Path, accept_terms: bool = False) -> Path:
    if not accept_terms:
        raise ValueError("Read the upstream noncommercial research/education terms; pass --accept-terms")
    archive = destination / "downloads" / "DeepCrack.zip"
    if not archive.exists():
        download(DEEPCRACK_URL, archive)
    raw = destination / "raw" / "deepcrack"
    safe_extract(archive, raw)
    return raw


def extract_crack500_bundle(archive: Path, destination: Path) -> dict:
    """Extract the actually published original subsets and complete crop benchmark.

    This release lacks traindata.zip. No mosaicking or invented originals is attempted.
    """
    destination.mkdir(parents=True, exist_ok=True)
    found = []
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if "/CRACK500/" not in name or not name.endswith(
                ("traincrop.zip", "valcrop.zip", "testcrop.zip", "valdata.zip", "testdata.zip")
            ):
                continue
            nested = destination / Path(name).name
            with z.open(name) as source, nested.open("wb") as target:
                shutil.copyfileobj(source, target)
            safe_extract(nested, destination)
            nested.unlink()
            found.append(Path(name).name)
    result = {"archive_sha256": sha256_file(archive), "extracted": found,
              "original_training_images": "not included in this upstream archive",
              "primary_500_image_protocol_ready": False}
    write_json(destination / "source_inventory.json", result)
    return result
