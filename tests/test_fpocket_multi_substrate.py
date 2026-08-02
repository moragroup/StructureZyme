# tests/test_fpocket_multi_substrate.py
import pandas as pd
import structurezyme.steps.fpocket_step as fp_mod
from structurezyme.steps.fpocket_step import Fpocket  # class name confirmed in fpocket_step.py


def test_fpocket_together_emits_per_substrate_keys(tmp_path, monkeypatch):
    prepared = tmp_path / "prep"; prepared.mkdir()
    (prepared / "chai_0.pdb").write_text("dummy")
    out = tmp_path / "out"; out.mkdir()
    df = pd.DataFrame({"Entry": ["P1"], "docked_structure": ["chai_0"],
                       "substrate_smiles": ["CCO.c1ccncc1"]})

    # stub the single-substrate worker to a deterministic feature Series so the
    # test targets ONLY the per-substrate suffixing/merge logic
    calls = {"smiles": []}
    def _fake_single(self, row, sub_smiles):
        calls["smiles"].append(sub_smiles)
        idx = len(calls["smiles"]) - 1
        return pd.Series({"ASvolume_dir": f"/x{idx}",
                          "pocket_volume": 123.0 + idx,
                          "pocket_sasa": 45.0 + idx})
    monkeypatch.setattr(fp_mod.Fpocket, "_fpocket_one_substrate",
                        _fake_single, raising=False)

    step = fp_mod.Fpocket(preparedfiles_dir=str(prepared), output_dir=str(out),
                          num_threads=1)
    res = step.execute(df)
    cols = set(res.columns)
    assert "pocket_volume_s0" in cols
    assert "pocket_volume_s1" in cols
    assert "pocket_volume" not in cols
    # each substrate's worker was actually invoked with its own SMILES
    assert calls["smiles"] == ["CCO", "c1ccncc1"]
    # distinct per-substrate values are preserved, not collapsed/overwritten
    assert res.loc[0, "pocket_volume_s0"] == 123.0
    assert res.loc[0, "pocket_volume_s1"] == 124.0


def test_fpocket_single_substrate_unchanged(monkeypatch, tmp_path):
    """Single-substrate rows keep unsuffixed columns and call the worker with
    the same substrate SMILES as before the refactor (legacy behavior)."""
    prepared = tmp_path / "prep"; prepared.mkdir()
    (prepared / "chai_0.pdb").write_text("dummy")
    out = tmp_path / "out"; out.mkdir()
    df = pd.DataFrame({"Entry": ["P1"], "docked_structure": ["chai_0"],
                       "substrate_smiles": ["CCO"]})

    calls = {"smiles": []}
    def _fake_single(self, row, sub_smiles):
        calls["smiles"].append(sub_smiles)
        return pd.Series({"ASvolume_dir": "/x", "pocket_volume": 99.0})
    monkeypatch.setattr(fp_mod.Fpocket, "_fpocket_one_substrate",
                        _fake_single, raising=False)

    step = fp_mod.Fpocket(preparedfiles_dir=str(prepared), output_dir=str(out),
                          num_threads=1)
    res = step.execute(df)
    cols = set(res.columns)
    assert "pocket_volume" in cols
    assert "pocket_volume_s0" not in cols
    assert calls["smiles"] == ["CCO"]
