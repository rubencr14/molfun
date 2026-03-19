"""
Data pipeline: sources, datasets, splits, storage, parsers, collections.
"""

from molfun.data.collections import (
    COLLECTIONS,
    CollectionSpec,
    count_all_collections,
    count_collection,
    fetch_collection,
    list_collections,
)
from molfun.data.datasets.affinity import AffinityDataset
from molfun.data.datasets.streaming import StreamingStructureDataset
from molfun.data.datasets.structure import StructureDataset
from molfun.data.sources.affinity import AffinityFetcher
from molfun.data.sources.msa import MSAProvider
from molfun.data.sources.pdb import PDBFetcher, StructureRecord, deduplicate_by_sequence
from molfun.data.splits import DataSplitter
from molfun.data.storage import ensure_dir, exists, is_remote, list_files, open_path

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
