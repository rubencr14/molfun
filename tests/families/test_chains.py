"""Single-chain extraction from PDB entries."""

import gzip

import gemmi
import pytest

from molfun.families.kinase.chains import extract_chain, seqres_one_letter


def _atom(name, element, x, altloc="\0", occ=1.0):
    atom = gemmi.Atom()
    atom.name = name
    atom.element = gemmi.Element(element)
    atom.pos = gemmi.Position(x, 0.0, 0.0)
    atom.altloc = altloc
    atom.occ = occ
    return atom


def _residue(name, num, atoms, het=False):
    res = gemmi.Residue()
    res.name = name
    res.seqid = gemmi.SeqId(num, " ")
    res.het_flag = "H" if het else "A"
    for atom in atoms:
        res.add_atom(atom)
    return res


def _two_chain_entry(path):
    """Chain A: ALA GLY SER (+LYS in SEQRES, unobserved), ligand ATP, water.
    GLY has two altlocs (A occ 0.4, B occ 0.6). Chain B: a different protein."""
    st = gemmi.Structure()
    st.name = "1TST"
    model = gemmi.Model("1")
    a = gemmi.Chain("A")
    a.add_residue(_residue("ALA", 1, [_atom("N", "N", 0.0), _atom("CA", "C", 1.0)]))
    a.add_residue(
        _residue(
            "GLY",
            2,
            [_atom("CA", "C", 2.0, "A", 0.4), _atom("CA", "C", 2.5, "B", 0.6)],
        )
    )
    a.add_residue(_residue("SER", 3, [_atom("CA", "C", 3.0)]))
    a.add_residue(_residue("ATP", 101, [_atom("PG", "P", 9.0)], het=True))
    a.add_residue(_residue("HOH", 201, [_atom("O", "O", 7.0)], het=True))
    b = gemmi.Chain("B")
    b.add_residue(_residue("TRP", 1, [_atom("CA", "C", 20.0)]))
    model.add_chain(a)
    model.add_chain(b)
    st.add_model(model)
    st.setup_entities()
    for entity in st.entities:
        if entity.entity_type == gemmi.EntityType.Polymer and "A" in entity.subchains[0]:
            entity.full_sequence = ["ALA", "GLY", "SER", "LYS"]
    st.assign_label_seq_id(force=True)
    doc = st.make_mmcif_document()
    block = doc.sole_block()
    subchains = ",".join(sorted({r.subchain for ch in st[0] for r in ch}))
    block.set_mmcif_category(
        "_pdbx_struct_assembly.",
        {
            "id": ["1"],
            "details": ["author_defined_assembly"],
            "method_details": ["PISA"],
            "oligomeric_details": ["dimeric"],
            "oligomeric_count": ["2"],
        },
    )
    block.set_mmcif_category(
        "_pdbx_struct_assembly_gen.",
        {"assembly_id": ["1"], "oper_expression": ["1"], "asym_id_list": [subchains]},
    )
    block.set_mmcif_category(
        "_pdbx_struct_oper_list.",
        {
            "id": ["1"],
            "type": ["identity operation"],
            "name": ["1_555"],
            "symmetry_operation": ["x,y,z"],
            **{f"matrix[{i}][{j}]": ["1" if i == j else "0"] for i in (1, 2, 3) for j in (1, 2, 3)},
            **{f"vector[{i}]": ["0"] for i in (1, 2, 3)},
        },
    )
    with gzip.open(path, "wt") as f:
        f.write(doc.as_string())


@pytest.fixture
def entry(tmp_path):
    path = tmp_path / "1tst.cif.gz"
    _two_chain_entry(path)
    return path


def _read(path):
    return gemmi.read_structure(str(path))


def test_extract_chain_single_chain_no_hetero(entry, tmp_path):
    info = extract_chain(entry, "A", tmp_path / "10.cif")

    st = _read(tmp_path / "10.cif")
    assert [ch.name for ch in st[0]] == ["A"]
    assert [r.name for r in st[0]["A"]] == ["ALA", "GLY", "SER"]
    assert info.seqres == "AGSK"
    assert info.n_observed == 3


def test_extract_chain_keeps_label_seq(entry, tmp_path):
    extract_chain(entry, "A", tmp_path / "10.cif")

    st = _read(tmp_path / "10.cif")
    assert [r.label_seq for r in st[0]["A"]] == [1, 2, 3]


def test_extract_chain_default_altloc_is_highest_occupancy(entry, tmp_path):
    extract_chain(entry, "A", tmp_path / "10.cif")

    gly = _read(tmp_path / "10.cif")[0]["A"][1]
    assert len(gly) == 1
    assert gly[0].pos.x == pytest.approx(2.5)


def test_extract_chain_requested_altloc(entry, tmp_path):
    extract_chain(entry, "A", tmp_path / "10.cif", altloc="A")

    gly = _read(tmp_path / "10.cif")[0]["A"][1]
    assert len(gly) == 1
    assert gly[0].pos.x == pytest.approx(2.0)


def test_fixture_has_an_assembly_over_both_chains(entry):
    assert len(gemmi.read_structure(str(entry)).assemblies) == 1


def test_extract_chain_drops_assemblies(entry, tmp_path):
    """The entry's biological assembly references chains we removed; a
    single-chain file must carry none (the Boltz parser expands assemblies)."""
    extract_chain(entry, "A", tmp_path / "10.cif")

    block = gemmi.cif.read(str(tmp_path / "10.cif")).sole_block()
    assert len(gemmi.read_structure(str(tmp_path / "10.cif")).assemblies) == 0
    assert block.find_mmcif_category("_pdbx_struct_assembly_gen.").width() == 0


def test_extract_chain_unknown_chain_raises(entry, tmp_path):
    with pytest.raises(KeyError, match="Z"):
        extract_chain(entry, "Z", tmp_path / "10.cif")


def test_seqres_maps_modified_residues_to_parent():
    assert seqres_one_letter(["ALA", "MSE", "SEP", "UNK", "SER,THR"]) == "AMSXS"
