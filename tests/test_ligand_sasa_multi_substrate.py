# tests/test_ligand_sasa_multi_substrate.py
import pandas as pd
import structurezyme.steps.ligandSASA_step as sasa_mod
from structurezyme.steps.ligandSASA_step import LigandSASA


def test_sasa_together_emits_per_substrate_keys(tmp_path, monkeypatch):
    (tmp_path / "chai_0.pdb").write_text("dummy")
    df = pd.DataFrame({"Entry": ["P1"], "docked_structure": ["chai_0"],
                       "substrate_smiles": ["CCO.c1ccncc1"]})

    calls = {"n": 0}
    def _fake_select(path, smiles):
        calls["n"] += 1
        return ("B", 1, "LIG") if calls["n"] == 1 else ("C", 2, "LIG")
    monkeypatch.setattr(sasa_mod, "select_ligand_from_smiles_via_composition",
                        _fake_select, raising=True)

    # stub the freesasa machinery so no real SASA runs
    class _FakeStruct: pass
    monkeypatch.setattr(sasa_mod.freesasa, "Structure",
                        lambda *a, **k: _FakeStruct(), raising=True)
    class _FakeResult:
        def totalArea(self): return 100.0
    monkeypatch.setattr(sasa_mod.freesasa, "calc",
                        lambda s: _FakeResult(), raising=True)
    monkeypatch.setattr(sasa_mod.freesasa, "selectArea",
                        lambda sel, s, r: {"ligand": 40.0}, raising=True)
    # stub the Bio.PDB save path
    monkeypatch.setattr(sasa_mod, "PDBParser",
                        lambda QUIET=True: type("P", (), {
                            "get_structure": lambda self, n, p: {0: object()}})(),
                        raising=True)
    monkeypatch.setattr(sasa_mod, "PDBIO",
                        lambda: type("IO", (), {
                            "set_structure": lambda self, s: None,
                            "save": lambda self, p, select=None: open(p, "w").close(),
                        })(), raising=True)

    step = LigandSASA(input_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
    out = step.execute(df)
    cols = set(out.columns)
    assert "buried_sasa_s0" in cols
    assert "buried_sasa_s1" in cols
    assert "buried_sasa" not in cols
