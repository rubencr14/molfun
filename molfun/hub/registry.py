"""
Pretrained model registry and weight auto-download.

Maps model names to their weight URLs, configs, and metadata.
Downloads are cached in ``~/.molfun/weights/<model_name>/``.

Usage::

    from molfun.hub.registry import download_weights, get_config

    weights_path = download_weights("openfold")
    config = get_config("openfold")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import hashlib
import logging
import shutil
import sys
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

_WEIGHTS_DIR = Path.home() / ".molfun" / "weights"

_OPENFOLD_WEIGHTS_URL = (
    "https://huggingface.co/nz/OpenFold/resolve/main/"
    "finetuning_ptm_2.pt"
)

_OPENFOLD_CONFIG_PRESET = "finetuning_ptm"


@dataclass
class PretrainedSpec:
    """Specification for a pretrained model."""
    name: str
    backend: str
    description: str
    weights_url: str
    weights_filename: str
    config_preset: str
    config_kwargs: dict = field(default_factory=dict)
    sha256: Optional[str] = None
    size_mb: Optional[int] = None


PRETRAINED_REGISTRY: dict[str, PretrainedSpec] = {}


def _register(spec: PretrainedSpec) -> None:
    PRETRAINED_REGISTRY[spec.name] = spec


_register(PretrainedSpec(
    name="openfold",
    backend="openfold",
    description="OpenFold AlphaFold2 (finetuning_ptm preset, full weights)",
    weights_url=_OPENFOLD_WEIGHTS_URL,
    weights_filename="finetuning_ptm_2.pt",
    config_preset=_OPENFOLD_CONFIG_PRESET,
    size_mb=700,
))

_register(PretrainedSpec(
    name="openfold_ptm",
    backend="openfold",
    description="OpenFold AlphaFold2 with pTM head (finetuning_ptm preset)",
    weights_url=_OPENFOLD_WEIGHTS_URL,
    weights_filename="finetuning_ptm_2.pt",
    config_preset=_OPENFOLD_CONFIG_PRESET,
    size_mb=700,
))


def list_pretrained() -> list[PretrainedSpec]:
    """Return all registered pretrained models."""
    return list(PRETRAINED_REGISTRY.values())


def download_weights(
    name: str,
    force: bool = False,
    cache_dir: Optional[str] = None,
    progress: bool = True,
) -> Path:
    """
    Download pretrained weights to local cache.

    Args:
        name: Pretrained model name (e.g. "openfold").
        force: Re-download even if cached.
        cache_dir: Override default cache directory.
        progress: Show download progress.

    Returns:
        Path to the downloaded weights file.
    """
    if name not in PRETRAINED_REGISTRY:
        available = ", ".join(sorted(PRETRAINED_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    spec = PRETRAINED_REGISTRY[name]
    base = Path(cache_dir) if cache_dir else _WEIGHTS_DIR
    model_dir = base / spec.name
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_path = model_dir / spec.weights_filename

    if weights_path.exists() and not force:
        logger.info("Using cached weights: %s", weights_path)
        return weights_path

    logger.info("Downloading %s weights (%s MB)...", spec.name, spec.size_mb or "?")
    _download_file(spec.weights_url, weights_path, progress=progress)

    if spec.sha256:
        _verify_sha256(weights_path, spec.sha256)

    logger.info("Weights saved to %s", weights_path)
    return weights_path


def get_config(name: str):
    """
    Build the config object for a pretrained model.

    Args:
        name: Pretrained model name.

    Returns:
        Config object (e.g. OpenFold ml_collections config).
    """
    if name not in PRETRAINED_REGISTRY:
        available = ", ".join(sorted(PRETRAINED_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    spec = PRETRAINED_REGISTRY[name]

    if spec.backend == "openfold":
        return _get_openfold_config(spec.config_preset, **spec.config_kwargs)
    else:
        raise ValueError(
            f"Config builder not implemented for backend '{spec.backend}'"
        )


def _get_openfold_config(preset: str, **kwargs):
    """Build OpenFold config from a preset name."""
    try:
        from openfold.config import model_config
        cfg = model_config(preset)
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg
    except ImportError:
        raise ImportError(
            "OpenFold is required for config generation. "
            "Install with: pip install openfold or from "
            "https://github.com/aqlaboratory/openfold"
        )


def _download_file(url: str, dest: Path, progress: bool = True) -> None:
    """Download a file with optional progress bar."""
    tmp = dest.with_suffix(".tmp")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "molfun/0.2"})
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 8 * 1024 * 1024

            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress and total > 0:
                        pct = downloaded / total * 100
                        bar_len = 30
                        filled = int(bar_len * downloaded / total)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        mb = downloaded / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        sys.stderr.write(
                            f"\r  [{bar}] {pct:.0f}% ({mb:.0f}/{total_mb:.0f} MB)"
                        )
                        sys.stderr.flush()

            if progress and total > 0:
                sys.stderr.write("\n")

        shutil.move(str(tmp), str(dest))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _verify_sha256(path: Path, expected: str) -> None:
    """Verify file integrity via SHA-256."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    actual = sha.hexdigest()
    if actual != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {path.name}: "
            f"expected {expected[:12]}..., got {actual[:12]}..."
        )
