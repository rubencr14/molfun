#!/usr/bin/env python
"""
Predict a protein structure from a fine-tuned checkpoint.

Takes a trained checkpoint (local or MinIO), picks a random structure
from the trained collection, extracts its sequence from the CIF file,
runs inference, and saves the predicted PDB.

This script shows how to use molfun for inference after fine-tuning:
  1. Load a fine-tuned checkpoint
  2. Pick a random protein from the trained collection
  3. Extract the sequence from CIF/PDB on disk or from MinIO
  4. Run forward pass → predicted 3D coordinates
  5. Write a PDB file you can open in PyMOL / ChimeraX

Usage:
    # Using local checkpoint and local data
    python scripts/predict_from_trained.py \\
        --checkpoint ./checkpoints/kinase_lora_v1/best

    # Using MinIO checkpoint and MinIO data
    python scripts/predict_from_trained.py \\
        --checkpoint s3://molfun-data/checkpoints/kinase_lora_v1/best \\
        --minio

    # Pick a specific PDB instead of random
    python scripts/predict_from_trained.py \\
        --checkpoint ./checkpoints/kinase_lora_v1/best \\
        --pdb-id 1yi3
"""

from __future__ import annotations

import logging
import random
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", message="Using a non-tuple sequence for multidimensional indexing")
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter")
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")
logging.getLogger("torch.fx._symbolic_trace").setLevel(logging.ERROR)

import argparse

from molfun.constants import THREE_TO_ONE as _THREE_TO_ONE
from molfun.models import MolfunStructureModel
from molfun.storage import MinioStorage


DEFAULT_COLLECTION = "kinases_human"
DEFAULT_CACHE_DIR = "./data/kinases"


# ── Sequence extraction ────────────────────────────────────────────────────────

def extract_sequence_from_cif(cif_path: str) -> str:
    """Extract the amino acid sequence from a mmCIF or PDB file using BioPython."""
    path = Path(cif_path)
    from Bio.PDB import MMCIFParser, PDBParser

    if path.suffix.lower() == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure("s", str(path))
    model = structure[0]
    chain = next(iter(model))

    seq_chars = []
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        aa = _THREE_TO_ONE.get(res.get_resname().strip())
        if aa:
            seq_chars.append(aa)

    return "".join(seq_chars)


# ── PDB writer ─────────────────────────────────────────────────────────────────

def coords_to_pdb(sequence: str, coords, plddt=None) -> str:
    """
    Write a PDB string from CA coordinates.

    coords: tensor [L, 37, 3] (all atoms) or [L, 3] (CA only).
    """
    import torch

    t = coords.detach().cpu()
    if t.dim() == 4:
        t = t[0, :, 1, :]   # batch, CA atom index = 1
    elif t.dim() == 3:
        t = t[:, 1, :]       # CA atom index = 1
    # t is now [L, 3]

    plddt_vals = None
    if plddt is not None:
        p = plddt.detach().cpu()
        if p.dim() >= 2:
            p = p[0]
        plddt_vals = p.tolist()

    from molfun.constants import ONE_TO_THREE as _ONE_TO_THREE

    lines = []
    for i, (aa, xyz) in enumerate(zip(sequence, t.tolist())):
        resname = _ONE_TO_THREE.get(aa.upper(), "UNK")
        x, y, z = xyz[0], xyz[1], xyz[2]
        bfactor = plddt_vals[i] * 100 if plddt_vals else 0.0
        line = (
            f"ATOM  {i+1:5d}  CA  {resname:3s} A{i+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfactor:6.2f}           C  "
        )
        lines.append(line)
    lines.append("END")
    return "\n".join(lines)


# ── Data helpers ───────────────────────────────────────────────────────────────

def pick_random_cif(cache_dir: str) -> str:
    """Pick a random .cif file from the local cache directory."""
    cache = Path(cache_dir)
    cif_files = sorted(cache.glob("*.cif"))
    if not cif_files:
        raise FileNotFoundError(f"No .cif files found in {cache_dir}")
    chosen = random.choice(cif_files)
    print(f"  Picked: {chosen.name}")
    return str(chosen)


def download_from_minio(storage: MinioStorage, pdb_id: str, collection: str, cache_dir: str) -> str:
    """Download a specific PDB from MinIO to local cache."""
    found, missing = storage.sync_ids_to_local([pdb_id], collection, cache_dir)
    if missing:
        raise FileNotFoundError(f"{pdb_id}.cif not found in MinIO under {collection}/")
    return str(Path(cache_dir) / f"{pdb_id}.cif")


def list_minio_ids(storage: MinioStorage, collection: str) -> list[str]:
    """List all PDB IDs available in MinIO for a collection."""
    client = storage._client()
    prefix = collection.lstrip("/")
    objects = client.list_objects(storage._bucket, prefix=prefix + "/", recursive=True)
    ids = []
    for obj in objects:
        name = obj.object_name.split("/")[-1]
        if name.endswith(".cif"):
            ids.append(name[:-4])
    return ids


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Predict structure from a fine-tuned checkpoint")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint dir (local or s3://...)")
    p.add_argument("--pdb-id", type=str, default=None,
                   help="Specific PDB ID to predict (default: random from collection)")
    p.add_argument("--collection", type=str, default=DEFAULT_COLLECTION)
    p.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    p.add_argument("--output", type=str, default="predicted.pdb",
                   help="Output PDB file path")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--minio", action="store_true",
                   help="Use MinIO for checkpoint and data")
    return p.parse_args()


def main():
    args = parse_args()

    storage: Optional[MinioStorage] = None
    if args.minio:
        storage = MinioStorage.from_env()

    # ── 1. Load model + checkpoint ────────────────────────────────────────────

    print(f"Loading checkpoint: {args.checkpoint}")

    ckpt_dir = args.checkpoint

    # If checkpoint is on MinIO, download it locally first
    if ckpt_dir.startswith("s3://"):
        if storage is None:
            storage = MinioStorage.from_env()
        local_ckpt = "./checkpoints/_predict_tmp"
        print(f"  Downloading checkpoint from MinIO → {local_ckpt}")
        _download_checkpoint_from_minio(storage, ckpt_dir, local_ckpt)
        ckpt_dir = local_ckpt

    model = MolfunStructureModel.from_pretrained("openfold", device=args.device)
    model.load(ckpt_dir)
    model.adapter.eval()
    print(f"  Model loaded on {args.device}")

    # ── 2. Pick a protein ─────────────────────────────────────────────────────

    print(f"\nSelecting protein from '{args.collection}'...")

    if args.pdb_id:
        pdb_id = args.pdb_id.lower()
    elif storage is not None:
        ids = list_minio_ids(storage, args.collection)
        if not ids:
            raise RuntimeError(f"No structures found in MinIO under {args.collection}/")
        pdb_id = random.choice(ids)
        print(f"  Picked from MinIO: {pdb_id}")
    else:
        cif_path = pick_random_cif(args.cache_dir)
        pdb_id = Path(cif_path).stem

    # ── 3. Ensure CIF is local ────────────────────────────────────────────────

    local_cif = Path(args.cache_dir) / f"{pdb_id}.cif"
    if not local_cif.exists():
        if storage is not None:
            print(f"  Downloading {pdb_id}.cif from MinIO...")
            download_from_minio(storage, pdb_id, args.collection, args.cache_dir)
        else:
            from molfun.data.sources.pdb import PDBFetcher
            print(f"  Downloading {pdb_id}.cif from RCSB...")
            PDBFetcher(cache_dir=args.cache_dir).fetch([pdb_id])

    # ── 4. Extract sequence ───────────────────────────────────────────────────

    print(f"\nExtracting sequence from {local_cif.name}...")
    sequence = extract_sequence_from_cif(str(local_cif))
    print(f"  Sequence ({len(sequence)} residues): {sequence[:60]}{'...' if len(sequence) > 60 else ''}")

    # ── 5. Predict structure ──────────────────────────────────────────────────

    print(f"\nRunning inference...")
    output = model.predict(sequence)

    coords = output.structure_coords
    plddt = output.confidence

    if coords is None:
        raise RuntimeError("Model did not return structure coordinates. Check that the model ran correctly.")

    mean_plddt = float(plddt.mean().item()) * 100 if plddt is not None else None
    print(f"  Done. Mean pLDDT: {mean_plddt:.1f}" if mean_plddt else "  Done.")

    # ── 6. Save PDB ───────────────────────────────────────────────────────────

    pdb_string = coords_to_pdb(sequence, coords, plddt)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(pdb_string)
    print(f"\nSaved → {out_path}")

    # ── 7. Upload predicted PDB to MinIO ──────────────────────────────────────

    if storage is not None:
        _upload_prediction_to_minio(storage, str(out_path), args.collection, pdb_id)


def _download_checkpoint_from_minio(storage: MinioStorage, s3_path: str, local_dir: str) -> None:
    """Download all files under an s3:// checkpoint path to local_dir."""
    from pathlib import Path as P

    # s3://bucket/prefix → prefix
    bucket_prefix = s3_path.replace(f"s3://{storage._bucket}/", "")
    client = storage._client()
    local = P(local_dir)
    local.mkdir(parents=True, exist_ok=True)

    objects = list(client.list_objects(storage._bucket, prefix=bucket_prefix, recursive=True))
    if not objects:
        raise FileNotFoundError(f"No files found at {s3_path}")

    for obj in objects:
        name = obj.object_name.split("/")[-1]
        dest = local / name
        if not dest.exists():
            client.fget_object(storage._bucket, obj.object_name, str(dest))
    print(f"  Downloaded {len(objects)} checkpoint files")


def _upload_prediction_to_minio(storage: MinioStorage, local_pdb: str, collection: str, pdb_id: str) -> None:
    """Upload the predicted PDB to MinIO under predictions/."""
    client = storage._client()
    object_name = f"predictions/{collection}/{pdb_id}_predicted.pdb"
    client.fput_object(storage._bucket, object_name, local_pdb)
    print(f"Remote → s3://{storage._bucket}/{object_name}")


if __name__ == "__main__":
    main()
