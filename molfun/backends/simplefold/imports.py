"""Import the upstream SimpleFold modules molfun uses, in one place.

SimpleFold's source tree is written to be run from ``src/simplefold``: its
modules import each other as top-level packages (``model``, ``utils``,
``processor``, ``boltz_data_pipeline``). We put that directory on
``sys.path`` and import only those packages. SimpleFold also ships a
top-level ``datasets`` package that collides with HuggingFace ``datasets``;
none of the modules below need it, and any pre-existing ``datasets`` entry in
``sys.modules`` is restored after importing.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from types import SimpleNamespace

_TOPLEVEL = ("model", "utils", "processor", "boltz_data_pipeline")


def _origin(module) -> str:
    """File of a module, or first search path of a (namespace) package."""
    origin = getattr(module, "__file__", None)
    if origin:
        return origin
    paths = list(getattr(module, "__path__", []) or [])
    return paths[0] if paths else ""


def _is_under(module, root: Path) -> bool:
    origin = _origin(module)
    return bool(origin) and Path(origin).resolve().is_relative_to(root)


@functools.lru_cache(maxsize=1)
def import_simplefold() -> SimpleNamespace:
    """Return the upstream SimpleFold symbols used by the molfun backend.

    Raises:
        ImportError: If the SimpleFold fork is not installed, or if another
            package already owns one of SimpleFold's top-level module names.
    """
    try:
        import simplefold
    except ImportError as exc:  # pragma: no cover - exercised by importorskip
        raise ImportError("SimpleFold is not installed: pip install -e <simplefold fork>") from exc

    root = Path(simplefold.__file__).resolve().parent
    for name in _TOPLEVEL:
        module = sys.modules.get(name)
        if module is not None and not _is_under(module, root):
            raise ImportError(
                f"top-level module {name!r} is already imported from {_origin(module)}"
            )

    added = str(root) not in sys.path
    if added:
        sys.path.insert(0, str(root))

    previous_datasets = sys.modules.get("datasets")
    try:
        from boltz_data_pipeline.crop.boltz import BoltzCropper
        from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
        from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
        from boltz_data_pipeline.types import Input, Record, Structure
        from boltz_data_pipeline.write.mmcif import to_mmcif
        from model.flow import LinearPath
        from model.simplefold import SimpleFold
        from model.torch.sampler import EMSampler
        from processor.protein_processor import ProteinDataProcessor
        from utils.boltz_utils import process_structure, save_structure
        from utils.datamodule_utils import (
            collate,
            extract_sequence_from_tokens,
            load_input,
            process_one_inference_structure,
        )
        from utils.esm_utils import _af2_to_esm, esm_registry
        from utils.fasta_utils import CCD_URL, process_fastas
    finally:
        # The imported packages stay cached in sys.modules (their submodules
        # resolve through the package __path__), so the source dir can leave
        # sys.path and later top-level imports (e.g. HF ``datasets``) are not shadowed.
        if added:
            sys.path.remove(str(root))
        current = sys.modules.get("datasets")
        if previous_datasets is not None:
            sys.modules["datasets"] = previous_datasets
        elif current is not None and _is_under(current, root):
            del sys.modules["datasets"]

    return SimpleNamespace(
        root=root,
        architecture_dir=root / "configs" / "model" / "architecture",
        BoltzCropper=BoltzCropper,
        BoltzFeaturizer=BoltzFeaturizer,
        BoltzTokenizer=BoltzTokenizer,
        EMSampler=EMSampler,
        Input=Input,
        LinearPath=LinearPath,
        ProteinDataProcessor=ProteinDataProcessor,
        Record=Record,
        SimpleFold=SimpleFold,
        Structure=Structure,
        af2_to_esm=_af2_to_esm,
        ccd_url=CCD_URL,
        collate=collate,
        esm_registry=esm_registry,
        extract_sequence_from_tokens=extract_sequence_from_tokens,
        load_input=load_input,
        process_fastas=process_fastas,
        process_one_inference_structure=process_one_inference_structure,
        process_structure=process_structure,
        save_structure=save_structure,
        to_mmcif=to_mmcif,
    )
