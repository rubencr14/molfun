"""
MOL2 (Tripos Sybyl) format parser for small molecules.

MOL2 files are widely used in molecular docking (DOCK, AutoDock)
and contain atom types, coordinates, bonds, and partial charges.
No external dependencies.
"""

from __future__ import annotations

from molfun.data.parsers.base import BaseLigandParser, ParsedMolecule, ParsedAtom, ParsedBond


_MOL2_BOND_MAP = {"1": 1, "2": 2, "3": 3, "ar": 4, "am": 1, "du": 0, "nc": 0, "un": 0}


class MOL2Parser(BaseLigandParser):
    """
    Parse Tripos MOL2 format files.

    Handles multi-molecule MOL2 files (multiple @<TRIPOS>MOLECULE sections).
    Extracts atom coordinates, Sybyl atom types, partial charges, and bonds.

    Usage::

        parser = MOL2Parser()
        molecules = parser.parse_file("ligand.mol2")
        mol = molecules[0]
        print(mol.name, mol.num_atoms, mol.coords.shape)
    """

    def parse_text(self, text: str) -> list[ParsedMolecule]:
        molecules = []
        mol_blocks = text.split("@<TRIPOS>MOLECULE")

        for block in mol_blocks[1:]:  # skip empty first split
            mol = self._parse_block("@<TRIPOS>MOLECULE" + block)
            if mol is not None:
                molecules.append(mol)

        return molecules

    def _parse_block(self, block: str) -> ParsedMolecule | None:
        sections = self._split_sections(block)

        mol_section = sections.get("MOLECULE", [])
        if not mol_section:
            return None

        mol_name = mol_section[0].strip() if mol_section else "unnamed"

        n_atoms = 0
        n_bonds = 0
        if len(mol_section) > 1:
            parts = mol_section[1].split()
            try:
                n_atoms = int(parts[0])
                n_bonds = int(parts[1]) if len(parts) > 1 else 0
            except (ValueError, IndexError):
                pass

        atoms = []
        for line in sections.get("ATOM", []):
            atom = self._parse_atom_line(line)
            if atom is not None:
                atoms.append(atom)

        bonds = []
        for line in sections.get("BOND", []):
            bond = self._parse_bond_line(line)
            if bond is not None:
                bonds.append(bond)

        if not atoms:
            return None

        return ParsedMolecule(name=mol_name, atoms=atoms, bonds=bonds)

    @staticmethod
    def _split_sections(block: str) -> dict[str, list[str]]:
        """Split MOL2 text into named sections."""
        sections: dict[str, list[str]] = {}
        current_section = None

        for line in block.splitlines():
            if line.startswith("@<TRIPOS>"):
                current_section = line.replace("@<TRIPOS>", "").strip()
                sections[current_section] = []
            elif current_section is not None:
                sections[current_section].append(line)

        return sections

    @staticmethod
    def _parse_atom_line(line: str) -> ParsedAtom | None:
        parts = line.split()
        if len(parts) < 6:
            return None
        try:
            atom_id = int(parts[0]) - 1  # 0-indexed
            atom_name = parts[1]
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            atom_type = parts[5]
        except (ValueError, IndexError):
            return None

        element = atom_type.split(".")[0]

        charge = 0.0
        if len(parts) > 8:
            try:
                charge = float(parts[8])
            except ValueError:
                pass

        residue = ""
        if len(parts) > 7:
            residue = parts[7]

        return ParsedAtom(
            index=atom_id,
            element=element,
            x=x, y=y, z=z,
            charge=charge,
            atom_type=atom_type,
            name=atom_name,
            residue=residue,
        )

    @staticmethod
    def _parse_bond_line(line: str) -> ParsedBond | None:
        parts = line.split()
        if len(parts) < 4:
            return None
        try:
            a1 = int(parts[1]) - 1  # 0-indexed
            a2 = int(parts[2]) - 1
            bond_type = parts[3].lower()
        except (ValueError, IndexError):
            return None

        order = _MOL2_BOND_MAP.get(bond_type, 1)
        return ParsedBond(atom1=a1, atom2=a2, order=order)

    @staticmethod
    def extensions() -> list[str]:
        return [".mol2"]
