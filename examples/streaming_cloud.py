#!/usr/bin/env python
"""
Stream structural data from cloud storage (S3 / MinIO / GCS).

When your dataset is too large for local disk (or you want a shared data lake),
Molfun's streaming pipeline reads structures lazily from object storage.

No download step — data flows directly into training.

Requires:
  pip install molfun[streaming]
  # For S3: pip install s3fs
  # For GCS: pip install gcsfs

MinIO setup (local S3-compatible storage):
  docker run -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address :9001
  # Default creds: minioadmin / minioadmin
"""

import os

from molfun.data.storage import open_path, list_files
from molfun.data.datasets.streaming import StreamingStructureDataset

# ── 1. Using the storage abstraction ─────────────────────────────────

print("=== Storage Abstraction ===\n")

# The same API works for local paths, S3, GCS, MinIO:

# Local
local_files = list_files("data/structures/", "*.cif")
print(f"  Local CIF files: {len(local_files)}")

# S3 (uncomment with real bucket)
# s3_files = list_files("s3://my-bucket/structures/", "*.cif")

# MinIO (local S3-compatible)
# os.environ["AWS_ENDPOINT_URL"] = "http://localhost:9000"
# os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
# os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
# minio_files = list_files("s3://structures/", "*.cif")

# Read a file transparently from any backend:
# with open_path("s3://my-bucket/structures/7bv2.cif") as f:
#     content = f.read()

# ── 2. Streaming dataset ─────────────────────────────────────────────

print("\n=== Streaming Dataset ===\n")

# StreamingStructureDataset is a PyTorch IterableDataset that:
#   - Lists files lazily from the remote store
#   - Downloads and parses one structure at a time
#   - Never holds the full dataset in memory
#   - Supports multi-worker DataLoader

# From local directory:
streaming_ds = StreamingStructureDataset(
    root="data/structures/",
    glob_pattern="*.cif",
    max_seq_len=256,
)

print(f"  Dataset: StreamingStructureDataset")
print(f"  Root:    data/structures/")
print(f"  Pattern: *.cif")
print()

# Iterate (each item is a feature dict):
for i, features in enumerate(streaming_ds):
    print(f"  Structure {i}: {features.get('pdb_id', 'unknown')}, "
          f"seq_len={features['seq_length'].item()}")
    if i >= 2:
        print("  ...")
        break

# ── 3. Training with streaming data ──────────────────────────────────

print("\n=== Streaming + Training ===\n")

from torch.utils.data import DataLoader

# Multi-worker streaming: each worker gets a shard of the file list
loader = DataLoader(
    streaming_ds,
    batch_size=1,
    num_workers=2,
    collate_fn=lambda batch: batch[0],  # structures vary in size
)

print("  DataLoader with 2 workers, batch_size=1")
print("  Each worker streams its own shard from storage")
print()
print("  In practice, pass this loader to strategy.fit():")
print("    strategy.fit(model, loader, val_loader, epochs=20)")

# ── 4. Cloud storage examples ────────────────────────────────────────

print("\n=== Cloud Configuration Examples ===\n")

configs = {
    "AWS S3": {
        "root": "s3://my-lab-bucket/alphafold-structures/",
        "env": "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY",
    },
    "MinIO (self-hosted)": {
        "root": "s3://structures/",
        "env": "AWS_ENDPOINT_URL=http://minio:9000, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY",
    },
    "Google Cloud Storage": {
        "root": "gs://my-lab-bucket/structures/",
        "env": "GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json",
    },
    "Local NFS mount": {
        "root": "/mnt/shared/structures/",
        "env": "(none)",
    },
}

for name, cfg in configs.items():
    print(f"  {name}:")
    print(f"    root = \"{cfg['root']}\"")
    print(f"    env  = {cfg['env']}")
    print()

print("All use the same StreamingStructureDataset API — just change the root path.")
