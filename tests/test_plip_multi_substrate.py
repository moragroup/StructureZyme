# tests/test_plip_multi_substrate.py
import pandas as pd
import structurezyme.steps.plip_step as plip_mod
from structurezyme.steps.plip_step import PLIP


class _FakeInteractions:
    hbonds_ldon = []; hbonds_pdon = []
    hydrophobic_contacts = []; saltbridge_pneg = []; saltbridge_lneg = []
    pistacking = []; pication_laro = []; pication_paro = []
    halogen_bonds = []; water_bridges = []; metal_complexes = []


def test_plip_together_emits_per_substrate_keys(tmp_path, monkeypatch):
    (tmp_path / "chai_0.pdb").write_text("dummy")
    df = pd.DataFrame({"Entry": ["P1"], "docked_structure": ["chai_0"],
                       "substrate_smiles": ["CCO.c1ccncc1"]})

    class _FakeComplex:
        interaction_sets = {"LIG:B:1": _FakeInteractions(),
                            "LIG:C:2": _FakeInteractions()}
        def load_pdb(self, p): pass
        def analyze(self): pass
    monkeypatch.setattr(plip_mod, "PDBComplex", _FakeComplex, raising=True)

    # return a different chain per substrate so both are analyzable
    calls = {"n": 0}
    def _fake_select(path, smiles):
        calls["n"] += 1
        return ("B", 1, "LIG") if calls["n"] == 1 else ("C", 2, "LIG")
    monkeypatch.setattr(plip_mod, "select_ligand_from_smiles_via_composition",
                        _fake_select, raising=True)

    step = PLIP(input_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
    out = step.execute(df)
    cols = set(out.columns)
    assert "plip_hydrogen_nbonds_s0" in cols
    assert "plip_hydrogen_nbonds_s1" in cols
    # single-substrate contract preserved: no unsuffixed key in together mode
    assert "plip_hydrogen_nbonds" not in cols


def test_plip_single_substrate_row_keeps_unsuffixed_columns_unchanged(tmp_path, monkeypatch):
    (tmp_path / "chai_0.pdb").write_text("dummy")
    df = pd.DataFrame({"Entry": ["P1"], "docked_structure": ["chai_0"],
                       "substrate_smiles": ["CCO"]})

    class _FakeComplex:
        interaction_sets = {"LIG:B:1": _FakeInteractions()}
        def load_pdb(self, p): pass
        def analyze(self): pass
    monkeypatch.setattr(plip_mod, "PDBComplex", _FakeComplex, raising=True)
    monkeypatch.setattr(plip_mod, "select_ligand_from_smiles_via_composition",
                        lambda path, smiles: ("B", 1, "LIG"), raising=True)

    step = PLIP(input_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
    out = step.execute(df)
    cols = set(out.columns)
    assert "plip_hydrogen_nbonds" in cols
    assert "plip_hydrogen_nbonds_s0" not in cols


def test_plip_together_mode_does_not_hard_filter_row(tmp_path, monkeypatch):
    """Even if one substrate can't be resolved to a ligand, the row survives
    with a None-filled _s{i} block rather than being dropped."""
    (tmp_path / "chai_0.pdb").write_text("dummy")
    df = pd.DataFrame({"Entry": ["P1"], "docked_structure": ["chai_0"],
                       "substrate_smiles": ["CCO.c1ccncc1"]})

    class _FakeComplex:
        interaction_sets = {"LIG:B:1": _FakeInteractions()}
        def load_pdb(self, p): pass
        def analyze(self): pass
    monkeypatch.setattr(plip_mod, "PDBComplex", _FakeComplex, raising=True)

    calls = {"n": 0}
    def _fake_select(path, smiles):
        calls["n"] += 1
        if calls["n"] == 1:
            return ("B", 1, "LIG")
        return None  # second substrate unresolvable
    monkeypatch.setattr(plip_mod, "select_ligand_from_smiles_via_composition",
                        _fake_select, raising=True)

    step = PLIP(input_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
    out = step.execute(df)

    assert len(out) == 1
    assert out.loc[0, "plip_hydrogen_nbonds_s0"] == 0
    assert out.loc[0, "plip_hydrogen_nbonds_s1"] is None


def test_plip_together_unexpected_substrate_error_does_not_leak_unsuffixed_keys(tmp_path, monkeypatch):
    """If a substrate raises an UNEXPECTED error mid-loop (not the guarded
    'no ligand' path), its failure must stay isolated as a suffixed _s{i} None
    block -- it must NOT fall through to the outer handler and write UNSUFFIXED
    keys that mix with the good substrate's _s0 columns."""
    (tmp_path / "chai_0.pdb").write_text("dummy")
    df = pd.DataFrame({"Entry": ["P1"], "docked_structure": ["chai_0"],
                       "substrate_smiles": ["CCO.c1ccncc1"]})

    class _FakeComplex:
        interaction_sets = {"LIG:B:1": _FakeInteractions()}
        def load_pdb(self, p): pass
        def analyze(self): pass
    monkeypatch.setattr(plip_mod, "PDBComplex", _FakeComplex, raising=True)

    calls = {"n": 0}
    def _fake_select(path, smiles):
        calls["n"] += 1
        if calls["n"] == 1:
            return ("B", 1, "LIG")
        raise RuntimeError("boom: unexpected failure on substrate #1")
    monkeypatch.setattr(plip_mod, "select_ligand_from_smiles_via_composition",
                        _fake_select, raising=True)

    step = PLIP(input_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
    out = step.execute(df)
    cols = set(out.columns)

    assert len(out) == 1
    # good substrate keeps its suffixed value; failed substrate is a suffixed None
    assert out.loc[0, "plip_hydrogen_nbonds_s0"] == 0
    assert out.loc[0, "plip_hydrogen_nbonds_s1"] is None
    # NO unsuffixed key leaks into the together-mode row
    assert "plip_hydrogen_nbonds" not in cols
    assert "plip_salt_bridges" not in cols


def test_plip_together_pdb_load_failure_yields_suffixed_none_no_leak(tmp_path, monkeypatch):
    """A once-per-PDB load/analyze failure is per-ROW, but in together mode it
    must still yield correctly-SUFFIXED None blocks for every substrate -- it
    must NOT write unsuffixed default keys."""
    (tmp_path / "chai_0.pdb").write_text("dummy")
    df = pd.DataFrame({"Entry": ["P1"], "docked_structure": ["chai_0"],
                       "substrate_smiles": ["CCO.c1ccncc1"]})

    class _FakeComplex:
        def load_pdb(self, p): pass
        def analyze(self): raise RuntimeError("boom: analyze failed")
    monkeypatch.setattr(plip_mod, "PDBComplex", _FakeComplex, raising=True)
    monkeypatch.setattr(plip_mod, "select_ligand_from_smiles_via_composition",
                        lambda path, smiles: ("B", 1, "LIG"), raising=True)

    step = PLIP(input_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
    out = step.execute(df)
    cols = set(out.columns)

    assert len(out) == 1
    # both substrates get suffixed None blocks; no unsuffixed key leaks
    assert out.loc[0, "plip_hydrogen_nbonds_s0"] is None
    assert out.loc[0, "plip_hydrogen_nbonds_s1"] is None
    assert "plip_hydrogen_nbonds" not in cols
