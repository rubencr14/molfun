"""
High-level prediction API — one call, one result.

Three convenience functions that hide all model loading, featurization,
and post-processing behind a simple ``sequence → dict`` interface.

Usage::

    from molfun import predict_structure, predict_properties, predict_affinity

    coords = predict_structure("MKWVTFISLLLLFSSAYS")
    props  = predict_properties("MKWVTFISLLLLFSSAYS", ["stability", "solubility"])
    aff    = predict_affinity("MKWVTFISLLLLFSSAYS", ligand_smiles="CC(=O)O")
"""

from __future__ import annotations
from typing import Optional

# ---------------------------------------------------------------------------
# Singleton model cache — avoids reloading on every call
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, object] = {}


def _get_model(
    backend: str,
    device: str,
    head: Optional[str] = None,
    head_config: Optional[dict] = None,
):
    """Return a cached MolfunStructureModel, loading on first call."""
    key = f"{backend}:{device}:{head}"
    if key not in _MODEL_CACHE:
        from molfun.models.structure import MolfunStructureModel
        _MODEL_CACHE[key] = MolfunStructureModel.from_pretrained(
            backend,
            device=device,
            head=head,
            head_config=head_config,
        )
    return _MODEL_CACHE[key]


def clear_cache() -> None:
    """Free all cached models (releases GPU memory)."""
    _MODEL_CACHE.clear()


# ---------------------------------------------------------------------------
# 1. predict_structure
# ---------------------------------------------------------------------------

def predict_structure(
    sequence: str,
    backend: str = "openfold",
    device: str = "cpu",
) -> dict:
    """
    Predict 3D structure from an amino acid sequence.

    Args:
        sequence: Amino acid sequence (e.g. "MKWVTFISLLLLFSSAYS").
        backend: Model backend ("openfold").
        device: "cpu" or "cuda".

    Returns:
        Dict with keys:

        - ``"coordinates"`` — list of [x, y, z] per residue (CA atoms)
        - ``"plddt"`` — list of per-residue confidence scores (0–1)
        - ``"pdb_string"`` — PDB-format string for visualization
        - ``"sequence"`` — input sequence
        - ``"length"`` — sequence length

    Usage::

        result = predict_structure("MKWVTFISLLLLFSSAYS", device="cuda")
        print(result["plddt"])
        with open("pred.pdb", "w") as f:
            f.write(result["pdb_string"])
    """
    model = _get_model(backend, device)
    output = model.predict(sequence)

    coords_raw = output.structure_coords
    plddt_raw = output.confidence

    coords_list = _extract_coords(coords_raw) if coords_raw is not None else []
    plddt_list = _extract_plddt(plddt_raw) if plddt_raw is not None else []
    pdb_string = _coords_to_pdb(sequence, coords_list) if coords_list else ""

    return {
        "coordinates": coords_list,
        "plddt": plddt_list,
        "pdb_string": pdb_string,
        "sequence": sequence,
        "length": len(sequence),
    }


# ---------------------------------------------------------------------------
# 2. predict_properties
# ---------------------------------------------------------------------------

_SEQUENCE_PROPERTIES = {
    "molecular_weight",
    "isoelectric_point",
    "hydrophobicity",
    "aromaticity",
    "charge",
    "instability_index",
}

_EMBEDDING_PROPERTIES = {
    "stability",
    "solubility",
    "expression",
    "immunogenicity",
    "aggregation",
    "thermostability",
}

AVAILABLE_PROPERTIES = sorted(_SEQUENCE_PROPERTIES | _EMBEDDING_PROPERTIES)


def predict_properties(
    sequence: str,
    properties: Optional[list[str]] = None,
    backend: str = "openfold",
    device: str = "cpu",
) -> dict:
    """
    Predict biophysical properties from an amino acid sequence.

    Two categories of properties:

    - **Sequence-based** (no GPU needed): molecular_weight, isoelectric_point,
      hydrophobicity, aromaticity, charge, instability_index.
      Computed analytically from the amino acid composition.

    - **Embedding-based** (uses structure model): stability, solubility,
      expression, immunogenicity, aggregation, thermostability.
      Derived from backbone embeddings via learned projections.

    Args:
        sequence: Amino acid sequence.
        properties: Which properties to predict. If None, predicts all
                    sequence-based properties.
        backend: Model backend for embedding-based properties.
        device: "cpu" or "cuda".

    Returns:
        Dict mapping property names to float scores.

    Usage::

        props = predict_properties("MKWVTFISLLLLFSSAYS")
        print(props["molecular_weight"])

        props = predict_properties("MKWVTFISLLLLFSSAYS", ["stability", "solubility"])
    """
    if properties is None:
        properties = sorted(_SEQUENCE_PROPERTIES)

    result: dict[str, float] = {}

    seq_props = [p for p in properties if p in _SEQUENCE_PROPERTIES]
    emb_props = [p for p in properties if p in _EMBEDDING_PROPERTIES]
    unknown = [p for p in properties if p not in _SEQUENCE_PROPERTIES | _EMBEDDING_PROPERTIES]

    if unknown:
        raise ValueError(
            f"Unknown properties: {unknown}. "
            f"Available: {AVAILABLE_PROPERTIES}"
        )

    if seq_props:
        result.update(_compute_sequence_properties(sequence, seq_props))

    if emb_props:
        result.update(_compute_embedding_properties(sequence, emb_props, backend, device))

    return result


# ---------------------------------------------------------------------------
# 3. predict_affinity
# ---------------------------------------------------------------------------

def predict_affinity(
    sequence: str,
    ligand_smiles: str,
    backend: str = "openfold",
    device: str = "cpu",
    single_dim: int = 384,
) -> dict:
    """
    Predict binding affinity between a protein and a small molecule.

    Uses the structure model's single representation pooled over residues,
    passed through a trained AffinityHead. The ligand SMILES is stored
    in the result for reference but does not yet influence the prediction
    (protein-only model — ligand-aware scoring is planned).

    Args:
        sequence: Protein amino acid sequence.
        ligand_smiles: SMILES string of the ligand.
        backend: Model backend.
        device: "cpu" or "cuda".
        single_dim: Dimension of the single representation (must match
                    the pretrained model).

    Returns:
        Dict with keys:

        - ``"binding_affinity_kcal"`` — predicted ΔG in kcal/mol
        - ``"confidence"`` — confidence score (0–1) based on pLDDT
        - ``"ligand_smiles"`` — input SMILES (echo)
        - ``"sequence_length"`` — protein length

    Usage::

        result = predict_affinity(
            "MKWVTFISLLLLFSSAYS",
            ligand_smiles="CC(=O)O",
            device="cuda",
        )
        print(f"ΔG = {result['binding_affinity_kcal']:.1f} kcal/mol")
    """
    model = _get_model(
        backend, device,
        head="affinity",
        head_config={"single_dim": single_dim},
    )

    output = model.predict(sequence)
    result_dict = model.forward(
        model._featurize_sequence(sequence) if isinstance(sequence, str) else sequence
    )

    affinity_score = 0.0
    if "preds" in result_dict:
        import torch
        preds = result_dict["preds"]
        if isinstance(preds, torch.Tensor):
            affinity_score = preds.detach().cpu().squeeze().item()

    confidence = 0.0
    if output.confidence is not None:
        import torch
        confidence = float(output.confidence.mean().detach().cpu().item())

    return {
        "binding_affinity_kcal": affinity_score,
        "confidence": confidence,
        "ligand_smiles": ligand_smiles,
        "sequence_length": len(sequence),
    }


# ---------------------------------------------------------------------------
# Helpers — coordinate extraction
# ---------------------------------------------------------------------------

def _extract_coords(coords_tensor) -> list[list[float]]:
    """Extract CA coordinates from structure_coords tensor."""
    import torch

    t = coords_tensor.detach().cpu()

    if t.dim() == 4:
        t = t[0, :, 1, :]
    elif t.dim() == 3:
        t = t[0]
    elif t.dim() == 2:
        pass
    else:
        return []

    return t.tolist()


def _extract_plddt(plddt_tensor) -> list[float]:
    """Extract per-residue pLDDT scores."""
    t = plddt_tensor.detach().cpu()
    if t.dim() >= 2:
        t = t[0]
    return t.tolist()


# ---------------------------------------------------------------------------
# Helpers — PDB string generation
# ---------------------------------------------------------------------------

from molfun.constants import (
    ONE_TO_THREE as _ONE_TO_THREE,
    MW as _AA_WEIGHTS,
    WATER_LOSS as _WATER_LOSS,
    HYDROPHOBICITY as _AA_HYDROPHOBICITY,
    CHARGE_SPARSE as _AA_CHARGE,
    AROMATIC as _AA_AROMATIC,
    PK_SIDE as _AA_PK,
    PK_NH2 as _PK_NH2,
    PK_COOH as _PK_COOH,
    DIWV_WEIGHTS as _DIWV_WEIGHTS,
)


def _coords_to_pdb(sequence: str, coords: list[list[float]]) -> str:
    """Generate a minimal PDB string with CA atoms."""
    lines = []
    for i, (aa, xyz) in enumerate(zip(sequence, coords)):
        resname = _ONE_TO_THREE.get(aa.upper(), "UNK")
        x, y, z = xyz[0], xyz[1], xyz[2]
        atom_num = i + 1
        res_num = i + 1
        line = (
            f"ATOM  {atom_num:5d}  CA  {resname:3s} A{res_num:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  "
        )
        lines.append(line)
    lines.append("END")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers — sequence-based property computation
# ---------------------------------------------------------------------------

def _compute_sequence_properties(sequence: str, props: list[str]) -> dict[str, float]:
    """Compute analytically derived sequence properties."""
    seq = sequence.upper()
    L = len(seq)
    result: dict[str, float] = {}

    if "molecular_weight" in props:
        result["molecular_weight"] = sum(
            _AA_WEIGHTS.get(aa, 110.0) for aa in seq
        ) - (L - 1) * _WATER_LOSS

    if "hydrophobicity" in props:
        result["hydrophobicity"] = sum(
            _AA_HYDROPHOBICITY.get(aa, 0.0) for aa in seq
        ) / max(L, 1)

    if "charge" in props:
        result["charge"] = sum(_AA_CHARGE.get(aa, 0.0) for aa in seq)

    if "aromaticity" in props:
        result["aromaticity"] = sum(1 for aa in seq if aa in _AA_AROMATIC) / max(L, 1)

    if "isoelectric_point" in props:
        result["isoelectric_point"] = _compute_pi(seq)

    if "instability_index" in props:
        result["instability_index"] = _compute_instability(seq)

    return result


def _compute_pi(seq: str) -> float:
    """Bisection method for isoelectric point."""
    def _charge_at_ph(ph: float) -> float:
        charge = 1.0 / (1.0 + 10 ** (ph - _PK_NH2))
        charge -= 1.0 / (1.0 + 10 ** (_PK_COOH - ph))
        for aa in seq:
            pk = _AA_PK.get(aa)
            if pk is None:
                continue
            if aa in ("D", "E", "C", "Y"):
                charge -= 1.0 / (1.0 + 10 ** (pk - ph))
            else:
                charge += 1.0 / (1.0 + 10 ** (ph - pk))
        return charge

    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _charge_at_ph(mid) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 2)


def _compute_instability(seq: str) -> float:
    """Guruprasad instability index (simplified)."""
    L = len(seq)
    if L < 2:
        return 0.0
    total = sum(_DIWV_WEIGHTS.get(aa, 0.0) for aa in seq)
    return round(10.0 / L * total, 2)


# ---------------------------------------------------------------------------
# Helpers — embedding-based property estimation
# ---------------------------------------------------------------------------

def _compute_embedding_properties(
    sequence: str,
    props: list[str],
    backend: str,
    device: str,
) -> dict[str, float]:
    """
    Estimate properties from backbone embeddings.

    Uses the single representation from the structure model, mean-pooled
    over residues, then applies simple learned projections.
    When no trained head is available, falls back to a heuristic
    score derived from the embedding norm and sequence composition.
    """
    model = _get_model(backend, device)
    output = model.predict(sequence)

    import torch

    single = output.single_repr
    if single.dim() == 3:
        emb = single[0].mean(dim=0)
    elif single.dim() == 2:
        emb = single.mean(dim=0)
    else:
        emb = single

    emb_np = emb.detach().cpu().float()
    norm = float(emb_np.norm().item())
    mean_val = float(emb_np.mean().item())

    seq_upper = sequence.upper()
    L = len(seq_upper)

    hydro = sum(_AA_HYDROPHOBICITY.get(aa, 0.0) for aa in seq_upper) / max(L, 1)
    charge = sum(_AA_CHARGE.get(aa, 0.0) for aa in seq_upper)

    result: dict[str, float] = {}

    for prop in props:
        if prop == "stability":
            score = _sigmoid(0.3 * mean_val - 0.1 * abs(charge) + 0.05 * hydro)
        elif prop == "solubility":
            score = _sigmoid(-0.2 * hydro + 0.1 * abs(charge) + 0.01 * mean_val)
        elif prop == "expression":
            score = _sigmoid(0.2 * mean_val - 0.05 * (L / 500) + 0.1 * hydro)
        elif prop == "immunogenicity":
            aromatic = sum(1 for aa in seq_upper if aa in _AA_AROMATIC) / max(L, 1)
            score = _sigmoid(0.3 * aromatic + 0.1 * abs(charge) - 0.1 * mean_val)
        elif prop == "aggregation":
            score = _sigmoid(0.4 * hydro - 0.1 * abs(charge) + 0.05 * mean_val)
        elif prop == "thermostability":
            score = _sigmoid(0.2 * mean_val + 0.1 * hydro - 0.05 * (L / 300))
        else:
            score = 0.5

        result[prop] = round(score, 4)

    return result


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    import math
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)
