"""
SDF (Structure-Data File) parser for small molecules.

SDF is the standard multi-molecule format used in cheminformatics.
Each molecule has atoms with 3D coordinates, bonds, and optional
property fields. No external dependencies.
"""

from __future__ import annotations
from typing import Optional

from molfun.data.parsers.base import BaseLigandParser, ParsedMolecule, ParsedAtom, ParsedBond


_BOND_TYPE_MAP = {"1": 1, "2": 2, "3": 3, "4": 4}  # 4 = aromatic


class SDFParser(BaseLigandParser):
    """
    Parse SDF/MOL V2000 format files.

    Handles multi-molecule SDF files (separated by $$$$).
    Extracts atoms, bonds, 3D coordinates, and SD property fields.

    Usage::

        parser = SDFParser()
        molecules = parser.parse_file("ligands.sdf")
        for mol in molecules:
            print(mol.name, mol.num_atoms, mol.coords.shape)
    """

    def parse_text(self, text: str) -> list[ParsedMolecule]:
        molecules = []
        blocks = text.split("$$$$")

        for block in blocks:
            block = block.strip()
            if not block:
                continue
            mol = self._parse_mol_block(block)
            if mol is not None:
                molecules.append(mol)

        return molecules

    def _parse_mol_block(self, block: str) -> Optional[ParsedMolecule]:
        lines = block.splitlines()
        if len(lines) < 4:
            return None

        mol_name = lines[0].strip() or "unnamed"

        counts_line = lines[3].strip()
        parts = counts_line.split()
        if len(parts) < 2:
            return None

        try:
            n_atoms = int(parts[0])
            n_bonds = int(parts[1])
        except ValueError:
            return None

        atoms = []
        for i in range(n_atoms):
            line_idx = 4 + i
            if line_idx >= len(lines):
                break
            atom = self._parse_atom_line(lines[line_idx], i)
            if atom is not None:
                atoms.append(atom)

        bonds = []
        for i in range(n_bonds):
            line_idx = 4 + n_atoms + i
            if line_idx >= len(lines):
                break
            bond = self._parse_bond_line(lines[line_idx])
            if bond is not None:
                bonds.append(bond)

        properties = self._parse_properties(lines, 4 + n_atoms + n_bonds)

        return ParsedMolecule(
            name=mol_name,
            atoms=atoms,
            bonds=bonds,
            properties=properties,
        )

    @staticmethod
    def _parse_atom_line(line: str, idx: int) -> Optional[ParsedAtom]:
        parts = line.split()
        if len(parts) < 4:
            return None
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            element = parts[3].strip()
        except (ValueError, IndexError):
            return None

        charge = 0.0
        if len(parts) > 8:
            try:
                raw_chg = int(parts[8])
                if raw_chg > 0:
                    charge = 4.0 - raw_chg  # MOL file charge convention
            except ValueError:
                pass

        return ParsedAtom(
            index=idx, element=element,
            x=x, y=y, z=z,
            charge=charge,
        )

    @staticmethod
    def _parse_bond_line(line: str) -> Optional[ParsedBond]:
        parts = line.split()
        if len(parts) < 3:
            return None
        try:
            a1 = int(parts[0]) - 1  # 1-indexed → 0-indexed
            a2 = int(parts[1]) - 1
            order = int(parts[2])
        except ValueError:
            return None
        return ParsedBond(atom1=a1, atom2=a2, order=order)

    @staticmethod
    def _parse_properties(lines: list[str], start: int) -> dict:
        """Parse SD property fields (>  <PROPERTY_NAME>)."""
        props = {}
        i = start
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("> ") and "<" in line:
                prop_name = line.split("<")[1].split(">")[0]
                values = []
                i += 1
                while i < len(lines) and lines[i].strip():
                    values.append(lines[i].strip())
                    i += 1
                props[prop_name] = "\n".join(values)
            i += 1
        return props

    @staticmethod
    def extensions() -> list[str]:
        return [".sdf", ".sd", ".mol"]
