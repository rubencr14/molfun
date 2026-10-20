"""Filesystem locations for the kinase pilot, read from the environment.

Keys (set them in the repo ``.env``; see ``.env.example``):

- ``MOLFUN_KINASE_DATA``: data folder of the KLIFS/KinCoRe kinase dataset.
- ``MOLFUN_WORK_DIR``: where pilot outputs (splits, records, runs) are written.
- ``MOLFUN_SIMPLEFOLD_CCD_RDB``: Boltz ``ccd.rdb`` used by SimpleFold processing.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_KEYS = ("MOLFUN_KINASE_DATA", "MOLFUN_WORK_DIR", "MOLFUN_SIMPLEFOLD_CCD_RDB")


@dataclass(frozen=True)
class PilotPaths:
    """Resolved pilot locations."""

    kinase_data: Path
    work_dir: Path
    ccd_rdb: Path


def pilot_paths(environ: Mapping[str, str] | None = None) -> PilotPaths:
    """Return the pilot locations.

    Args:
        environ: Mapping to read from. Defaults to ``os.environ`` after loading
            the nearest ``.env`` file (values already set are not overridden).

    Raises:
        RuntimeError: If a required key is unset; the message names the key.
    """
    if environ is None:
        try:
            from dotenv import load_dotenv

            load_dotenv(override=False)
        except ImportError:
            pass
        environ = os.environ

    missing = [key for key in _KEYS if not environ.get(key)]
    if missing:
        raise RuntimeError(f"Missing environment keys: {', '.join(missing)} (see .env.example)")

    def resolve(key: str) -> Path:
        return Path(environ[key]).expanduser()

    return PilotPaths(
        kinase_data=resolve("MOLFUN_KINASE_DATA"),
        work_dir=resolve("MOLFUN_WORK_DIR"),
        ccd_rdb=resolve("MOLFUN_SIMPLEFOLD_CCD_RDB"),
    )
