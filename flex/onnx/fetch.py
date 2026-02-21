"""
Download ONNX embedding model from GitHub release assets.

Called by `flex init`. Model stored at ~/.flex/models/ to persist across
pip upgrades. Uses urllib only — no extra dependencies.
"""
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

FLEX_HOME = Path(os.environ.get("FLEX_HOME", Path.home() / ".flex"))
MODEL_DIR = FLEX_HOME / "models"

BASE_URL = "https://github.com/axpsystems/flex/releases/download/v0.1.1"

FILES = [
    ("model.onnx", "30ff8ad63546f9efd85019f394445f566ea595c119b08aa6663058af9e18fa87"),
    ("model.onnx.data", "853ca16b709b09328d2d596c29e747163566139af3836fee64a358317e1c4268"),
]


def model_dir() -> Path:
    """Return model directory, creating if needed."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR


def model_ready() -> bool:
    """Check if all model files exist."""
    return all((MODEL_DIR / name).exists() for name, _ in FILES)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / (1 << 20)
        total_mb = total_size / (1 << 20)
        sys.stdout.write(f"\r  downloading: {mb:.1f}/{total_mb:.1f} MB ({pct}%)")
        sys.stdout.flush()


def download_model(force: bool = False) -> Path:
    """
    Download model files from GitHub release assets.

    Args:
        force: Re-download even if files exist.

    Returns:
        Path to model directory.

    Raises:
        RuntimeError: If download fails or checksum mismatch.
    """
    dest = model_dir()

    for name, expected_hash in FILES:
        target = dest / name
        if target.exists() and not force:
            continue

        url = f"{BASE_URL}/{name}"
        print(f"  {name}")
        try:
            urllib.request.urlretrieve(url, target, reporthook=_progress_hook)
            print()  # newline after progress
        except Exception as e:
            target.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download {name} from {url}: {e}\n"
                f"You can download manually and place in {dest}/"
            ) from e

        # Verify checksum
        actual = _sha256(target)
        if actual != expected_hash:
            target.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {name}.\n"
                f"  expected: {expected_hash}\n"
                f"  got:      {actual}\n"
                f"Re-run 'flex init' to retry."
            )

    return dest
