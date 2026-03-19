"""
Data pipeline: sources, datasets, splits, storage, parsers, collections.
"""

from molfun.data.sources.pdb import PDBFetcher, StructureRecord, deduplicate_by_sequence
from molfun.data.sources.affinity import AffinityFetcher
from molfun.data.sources.msa import MSAProvider
from molfun.data.datasets.structure import StructureDataset
from molfun.data.datasets.affinity import AffinityDataset
from molfun.data.datasets.streaming import StreamingStructureDataset
from molfun.data.splits import DataSplitter
from molfun.data.storage import open_path, list_files, exists, ensure_dir, is_remote
from molfun.data.collections import (
    COLLECTIONS,
    CollectionSpec,
    list_collections,
    fetch_collection,
    count_collection,
    count_all_collections,
)

__all__ = [
    "PDBFetcher",
    "StructureRecord",
    "deduplicate_by_sequence",
    "AffinityFetcher",
    "MSAProvider",
    "StructureDataset",
    "AffinityDataset",
    "StreamingStructureDataset",
    "DataSplitter",
    "COLLECTIONS",
    "CollectionSpec",
    "list_collections",
    "fetch_collection",
    "count_collection",
    "count_all_collections",
    "open_path",
    "list_files",
    "exists",
    "ensure_dir",
    "is_remote",
]
