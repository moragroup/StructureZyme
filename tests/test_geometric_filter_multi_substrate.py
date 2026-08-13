# tests/test_geometric_filter_multi_substrate.py
import pandas as pd

import structurezyme.steps.geometric_filtering_cofactor_MCS as geo_mod
from structurezyme.steps.geometric_filtering_cofactor_MCS import (
    GeneralGeometricFiltering,
    _suffix_keys,
)


def test_suffix_keys_appends_index():
    base = {"distance_ligand_to_closest_nuc": {"CYS": 3.1},
            "ligand_moiety_method": "mcs"}
    out = _suffix_keys(base, 1)
    assert out == {"distance_ligand_to_closest_nuc_s1": {"CYS": 3.1},
                   "ligand_moiety_method_s1": "mcs"}


def test_suffix_keys_noop_for_none_index():
    base = {"ligand_moiety_method": "mcs"}
    assert _suffix_keys(base, None) == {"ligand_moiety_method": "mcs"}


def _stub_common(monkeypatch):
    """Stub out all PDB/RDKit I/O so __execute's control flow can be tested
    without real structures. All substrate-dependent behavior is driven by
    moiety_centroid_with_fallbacks based on the SMILES it receives.
    """
    monkeypatch.setattr(geo_mod, "get_hetatm_chain_ids", lambda p: ["A"])
    monkeypatch.setattr(geo_mod, "extract_chain_as_rdkit_mol",
                        lambda p, cid, sanitize=False: object())
    monkeypatch.setattr(geo_mod, "closest_ligands_by_element_composition",
                        lambda ligands, smiles, top_k=1: [object()])
    monkeypatch.setattr(geo_mod, "as_mol", lambda x: x)
    monkeypatch.setattr(geo_mod, "assign_bond_orders_from_smiles",
                        lambda mol, smiles: mol)
    monkeypatch.setattr(geo_mod, "ensure_3d", lambda mol: mol)
    monkeypatch.setattr(geo_mod, "get_all_nucs_atom_coords", lambda p: {"CYS_10": []})
    monkeypatch.setattr(geo_mod, "find_min_distance",
                        lambda centroid, nucs: {"nuc_res": "CYS_10", "distance": 4.2})


def _base_row(smiles, moiety):
    return {
        "Entry": "P1",
        "docked_structure": "chai_0",
        "substrate_smiles": smiles,
        "cofactor_smiles": None,
        "substrate_moiety": moiety,
        "cofactor_moiety": None,
        "tool": "chai",
    }


def test_together_mode_emits_per_substrate_columns_without_hard_filtering(
    tmp_path, monkeypatch
):
    (tmp_path / "chai_0.pdb").write_text("dummy")
    _stub_common(monkeypatch)

    def _fake_centroid(mol, moiety_smiles, kind, grow_mcs_by_one_bond, use_chirality):
        if moiety_smiles == "c1ccncc1":
            # Simulate a failure for the second substrate's moiety matching.
            raise ValueError("boom")
        return [(0.0, 0.0, 0.0)], "mcs", [0]

    monkeypatch.setattr(geo_mod, "moiety_centroid_with_fallbacks", _fake_centroid)

    df = pd.DataFrame([_base_row("CCO.c1ccncc1", "CCO|c1ccncc1")])
    step = GeneralGeometricFiltering(
        preparedfiles_dir=str(tmp_path), output_dir=str(tmp_path / "out")
    )
    out = step.execute(df)

    # Row is NOT dropped even though one substrate's analysis raised.
    assert len(out) == 1
    cols = set(out.columns)
    assert "ligand_moiety_method_s0" in cols
    assert "ligand_moiety_method_s1" in cols
    assert "distance_ligand_to_closest_nuc_s0" in cols
    assert "distance_ligand_to_closest_nuc_s1" in cols
    # Substrate 0 succeeded.
    assert out.loc[0, "ligand_moiety_method_s0"] == "mcs"
    # Substrate 1 failed internally but the row still exists with None fallback.
    assert out.loc[0, "ligand_moiety_method_s1"] is None
    # Unsuffixed keys must not appear in together mode.
    assert "ligand_moiety_method" not in cols
    assert "distance_ligand_to_closest_nuc" not in cols


def test_single_substrate_row_keeps_unsuffixed_columns_unchanged(tmp_path, monkeypatch):
    (tmp_path / "chai_0.pdb").write_text("dummy")
    _stub_common(monkeypatch)

    def _fake_centroid(mol, moiety_smiles, kind, grow_mcs_by_one_bond, use_chirality):
        return [(0.0, 0.0, 0.0)], "mcs", [0]

    monkeypatch.setattr(geo_mod, "moiety_centroid_with_fallbacks", _fake_centroid)

    df = pd.DataFrame([_base_row("CCO", "CCO")])
    step = GeneralGeometricFiltering(
        preparedfiles_dir=str(tmp_path), output_dir=str(tmp_path / "out")
    )
    out = step.execute(df)

    cols = set(out.columns)
    assert "ligand_moiety_method" in cols
    assert "distance_ligand_to_closest_nuc" in cols
    assert out.loc[0, "ligand_moiety_method"] == "mcs"
    # No suffixed columns leak in for a single-substrate row.
    assert "ligand_moiety_method_s0" not in cols
