"""Extract one protein chain from a PDB entry as a clean single-chain mmCIF.

The output keeps the entry's metadata (dates, method, resolution) and the
residue numbering ``label_seq_id`` of the source, so residue *i* of the chain's
SEQRES sequence is ``label_seq_id == i + 1`` everywhere downstream.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import gemmi

_ASSEMBLY_CATEGORIES = (
    "_pdbx_struct_assembly.",
    "_pdbx_struct_assembly_gen.",
    "_pdbx_struct_assembly_prop.",
    "_pdbx_struct_assembly_auth_evidence.",
    "_pdbx_struct_oper_list.",
)


@dataclass(frozen=True)
class ChainInfo:
    """What was extracted.

    Attributes:
        seqres: Full entity (SEQRES) sequence, one letter per residue.
        n_observed: Polymer residues with coordinates.
    """

    seqres: str
    n_observed: int


def seqres_one_letter(names: list[str]) -> str:
    """One-letter sequence; modified residues map to their parent (MSE→M,
    SEP→S, TPO→T, PTR→Y), unknown ones to X. Microheterogeneity (``"SER,THR"``)
    keeps the first residue."""
    letters = []
    for name in names:
        info = gemmi.find_tabulated_residue(name.split(",")[0])
        code = info.one_letter_code if info is not None and info.is_amino_acid() else " "
        letters.append(code.upper() if code.strip() else "X")
    return "".join(letters)


def _keep_one_conformer(chain: gemmi.Chain, altloc: str | None) -> None:
    """Keep ``altloc`` (plus atoms without altloc), or the highest-occupancy one."""
    for residue in chain:
        occupancy = defaultdict(list)
        for atom in residue:
            if atom.has_altloc():
                occupancy[atom.altloc].append(atom.occ)
        if not occupancy:
            continue
        if altloc is not None and altloc in occupancy:
            keep = altloc
        else:
            keep = max(occupancy, key=lambda a: (sum(occupancy[a]) / len(occupancy[a]), -ord(a)))
        for i in reversed(range(len(residue))):
            atom = residue[i]
            if atom.has_altloc() and atom.altloc != keep:
                del residue[i]
            elif atom.has_altloc():
                atom.altloc = "\0"


def extract_chain(
    cif_gz: str | Path, chain: str, out_path: str | Path, altloc: str | None = None
) -> ChainInfo:
    """Write ``chain`` of the entry to ``out_path`` (mmCIF) and describe it.

    First model only; other chains, ligands and waters removed; one
    conformer per residue (``altloc`` if given, else the highest occupancy).

    Raises:
        KeyError: If the entry has no such author chain.
    """
    doc = gemmi.cif.read(str(cif_gz))
    return extract_chain_from_doc(doc, chain, out_path, altloc)


def extract_chain_from_doc(
    doc: gemmi.cif.Document, chain: str, out_path: str | Path, altloc: str | None = None
) -> ChainInfo:
    """Like :func:`extract_chain` for an already-parsed document (read once,
    extract many chains)."""
    block = doc.sole_block()
    st = gemmi.make_structure_from_block(block)
    st.setup_entities()
    while len(st) > 1:
        del st[1]
    model = st[0]
    names = [c.name for c in model]
    if chain not in names:
        raise KeyError(f"chain {chain!r} not in {st.name} (has {names})")
    for name in names:
        if name != chain:
            model.remove_chain(name)
    st.remove_ligands_and_waters()
    _keep_one_conformer(model[chain], altloc)
    st.remove_alternative_conformations()
    st.remove_empty_chains()

    polymer = model[chain].get_polymer()
    entity = st.get_entity_of(polymer)
    if entity is not None and entity.full_sequence:
        seqres = seqres_one_letter(list(entity.full_sequence))
    else:
        seqres = seqres_one_letter([r.name for r in polymer])
    n_observed = sum(1 for r in polymer if r.label_seq)

    # The entry's biological assemblies reference chains removed above; the
    # Boltz parser expands assembly 1 and would fail ("no subchain B").
    st.assemblies.clear()

    out_doc = gemmi.cif.Document()
    out_block = out_doc.add_copied_block(block)
    st.update_mmcif_block(out_block)
    for category in _ASSEMBLY_CATEGORIES:
        out_block.find_mmcif_category(category).erase()
    out_doc.write_file(str(out_path))
    return ChainInfo(seqres=seqres, n_observed=n_observed)
