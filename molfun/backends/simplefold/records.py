"""SimpleFold training records: a group-balanced dataset over processed targets.

Records are produced by upstream's ``process_mmcif`` (targets) and
``process_structure`` (tokens). Each dataset item is one *group* (e.g. one
UniProt accession): a random member is drawn per access, so a protein with
hundreds of structures counts once per epoch. Cropping and featurization
reuse upstream's Boltz pipeline with the settings of its training configs.
"""

from __future__ import annotations

import json
import pickle
import random
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from molfun.backends.simplefold.imports import import_simplefold


class SimpleFoldRecordDataset(Dataset):
    """One item per group; each access featurizes a random member record."""

    def __init__(
        self,
        tokenized_dir: str | Path,
        target_dir: str | Path,
        groups: dict[str, list[str]],
        max_tokens: int = 256,
        max_atoms: int = 2304,
    ):
        """
        Args:
            tokenized_dir: ``process_structure`` output (``records/``, ``tokens/``).
            target_dir: ``process_mmcif`` output (``structures/``), used if a token file is missing.
            groups: Group name → record ids.
            max_tokens: Crop size in residues.
            max_atoms: Crop size in atoms.
        """
        if not groups or any(not ids for ids in groups.values()):
            raise ValueError("groups must be non-empty and every group needs at least one record")
        self._sf = import_simplefold()
        self.tokenized_dir = Path(tokenized_dir)
        self.target_dir = Path(target_dir)
        self.group_names = list(groups)
        self.group_members = [list(ids) for ids in groups.values()]
        self.max_tokens = max_tokens
        self.max_atoms = max_atoms
        self._tokenizer = self._sf.BoltzTokenizer()
        self._featurizer = self._sf.BoltzFeaturizer()
        self._cropper = self._sf.BoltzCropper(min_neighborhood=0, max_neighborhood=40)

    def __len__(self) -> int:
        return len(self.group_members)

    def __getitem__(self, index: int) -> dict:
        record_id = random.choice(self.group_members[index])
        return self.featurize(record_id)

    def featurize(self, record_id: str) -> dict:
        """Load, crop and featurize one record (as upstream's training dataset does)."""
        sf = self._sf
        with open(self.tokenized_dir / "records" / f"{record_id.lower()}.json") as f:
            record = sf.Record(**json.load(f))
        token_path = self.tokenized_dir / "tokens" / f"{record.id}.pkl"
        if token_path.is_file():
            with open(token_path, "rb") as f:
                tokenized = pickle.load(f)
        else:
            tokenized = self._tokenizer.tokenize(sf.load_input(record, self.target_dir))

        max_num_tokens = len(tokenized.tokens)
        if max_num_tokens == 0:
            raise ValueError(f"record {record.id} has no tokens")
        tokenized = self._cropper.crop(
            tokenized, max_atoms=self.max_atoms, max_tokens=self.max_tokens, random=_numpy_rng()
        )
        sequence = sf.extract_sequence_from_tokens(tokenized)
        features = self._featurizer.process(
            tokenized,
            max_atoms=None,
            max_tokens=None,
            symmetries={},
            atoms_per_window_queries=32,
            min_dist=2.0,
            max_dist=22.0,
            num_bins=64,
            compute_symmetries=False,
            rotation_augment_ref_pos=False,
            rotation_augment_coords=True,
        )
        features["aa_seq"] = sequence
        features["record"] = asdict(record)
        features["max_num_tokens"] = torch.tensor(max_num_tokens, dtype=torch.long)
        features["cropped_num_tokens"] = torch.tensor(len(tokenized.tokens), dtype=torch.long)
        return features


def _redis_server_binary() -> str | None:
    found = shutil.which("redis-server")
    if found:
        return found
    candidate = Path(sys.executable).parent / "redis-server"
    return str(candidate) if candidate.is_file() else None


@contextmanager
def _redis_ccd(ccd_rdb: Path, port: int, timeout: float = 300.0):
    """Serve the Boltz CCD dump on ``port`` for the duration of the block
    (read-only: persistence disabled, nothing is written back)."""
    import redis

    exe = _redis_server_binary()
    if exe is None:
        raise FileNotFoundError(
            "redis-server not found (conda install -c conda-forge redis-server)"
        )
    args = [exe, "--dbfilename", ccd_rdb.name, "--dir", str(ccd_rdb.parent), "--port", str(port)]
    args += ["--save", "", "--appendonly", "no", "--bind", "127.0.0.1"]
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        client = redis.Redis(host="127.0.0.1", port=port)
        deadline = time.monotonic() + timeout
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"redis-server exited: {proc.stderr.read()[-1000:]}")
            try:
                if client.ping() and client.dbsize() > 0:
                    break
            except (redis.exceptions.ConnectionError, redis.exceptions.BusyLoadingError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"redis-server did not load {ccd_rdb} within {timeout}s")
            time.sleep(0.5)
        yield
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def _run_upstream(script: str, *args) -> None:
    root = import_simplefold().root
    cmd = [sys.executable, str(root / script), *map(str, args)]
    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed: {(result.stderr or result.stdout)[-2000:]}")


def prepare_simplefold_records(
    cif_dir: str | Path,
    out_dir: str | Path,
    ccd_rdb: str | Path,
    port: int = 7777,
    num_processes: int = 16,
) -> Path:
    """Run upstream ``process_mmcif`` (no assemblies) and ``process_structure``.

    Writes ``out_dir/targets``, ``out_dir/tokens`` and
    ``out_dir/preprocess_report.json`` (inputs, tokenized count, ids that
    upstream failed to process — it reports per-file failures only on stdout
    and still exits 0). Returns the tokens dir.
    A temporary ``redis-server`` serves the CCD while ``process_mmcif`` runs.

    Raises:
        FileNotFoundError: If ``redis-server`` is not installed.
        RuntimeError: If no input was turned into a tokenized record.
    """
    cif_dir, out_dir, ccd_rdb = Path(cif_dir), Path(out_dir), Path(ccd_rdb)
    targets, tokens = out_dir / "targets", out_dir / "tokens"
    with _redis_ccd(ccd_rdb, port):
        _run_upstream(
            "process_mmcif.py",
            "--data_dir", cif_dir, "--out_dir", targets,
            "--num-processes", num_processes, "--redis-port", port,
        )  # fmt: skip
    _run_upstream("process_structure.py", "--target_dir", targets, "--token_dir", tokens)

    inputs = sorted(p.name.split(".")[0].lower() for p in cif_dir.glob("*.cif*"))
    done = {p.stem for p in (tokens / "tokens").glob("*.pkl")}
    report = {
        "n_inputs": len(inputs),
        "n_tokenized": sum(1 for i in inputs if i in done),
        "missing": [i for i in inputs if i not in done],
    }
    (out_dir / "preprocess_report.json").write_text(json.dumps(report, indent=1))
    if report["n_tokenized"] == 0:
        raise RuntimeError(f"no records were produced from {len(inputs)} inputs in {cif_dir}")
    return tokens


def collate_records(items: list[dict]) -> dict:
    """Collate featurized records with upstream's collate (use as ``collate_fn``)."""
    return import_simplefold().collate(items)


def _numpy_rng():
    """A legacy-API NumPy RNG (upstream croppers call ``randint``/``choice``),
    seeded from Python's ``random`` so DataLoader worker seeding applies."""
    import numpy as np

    return np.random.RandomState(random.getrandbits(32))
