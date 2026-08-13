# tests/test_fpocket_multi_substrate.py
from pathlib import Path
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
    def _fake_single(self, row, sub_smiles, tag=None):
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
    def _fake_single(self, row, sub_smiles, tag=None):
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


def _stub_fpocket_env(monkeypatch, out_dir):
    """Stub subprocess+parsing so `_fpocket_one_substrate` runs without a real
    fpocket binary. Returns nothing; simulates a successful run whose output dir
    the code will move into `self.output_dir`."""
    monkeypatch.setattr(fp_mod, "fpocket_r_from_smiles_via_composition",
                        lambda p, s: None, raising=True)

    class _Ok:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(cmd, cwd=None, capture_output=True, text=True):
        # fpocket writes `<stem>_out/` next to the copied pdb in `cwd`
        stem = None
        for i, a in enumerate(cmd):
            if a == "-f":
                stem = Path(cmd[i + 1]).stem
        (Path(cwd) / f"{stem}_out" / "pockets").mkdir(parents=True, exist_ok=True)
        (Path(cwd) / f"{stem}_out" / "info.txt").write_text("SASA info")
        return _Ok()

    monkeypatch.setattr(fp_mod.subprocess, "run", _fake_run, raising=True)
    monkeypatch.setattr(fp_mod, "extract_fpocket_features",
                        lambda p: {"pocket_volume": 7.0}, raising=True)
    monkeypatch.setattr(fp_mod, "extract_SASA",
                        lambda p: {"pocket_sasa": 3.0}, raising=True)


def test_fpocket_single_substrate_asvolume_dir_is_legacy_name(monkeypatch, tmp_path):
    """REGRESSION: single-substrate/off rows must keep the legacy fpocket output
    directory name `<stem>_fpocket_output` (no substrate-hash tag) so the
    surfaced `ASvolume_dir` DataFrame column value is byte-for-byte unchanged."""
    prepared = tmp_path / "prep"; prepared.mkdir()
    (prepared / "chai_0.pdb").write_text("dummy")
    out = tmp_path / "out"; out.mkdir()
    _stub_fpocket_env(monkeypatch, out)

    step = fp_mod.Fpocket(preparedfiles_dir=str(prepared), output_dir=str(out),
                          num_threads=1)
    row = pd.Series({"Entry": "P1", "docked_structure": "chai_0",
                     "substrate_smiles": "CCO"})
    # legacy path: dispatcher passes tag=None for single substrate
    res = step._process_single_row_with_fpocket(row)
    assert res["ASvolume_dir"] == str(out / "chai_0_fpocket_output")
    assert "_s0" not in "".join(res.index)


def test_fpocket_together_output_dirs_are_distinct(monkeypatch, tmp_path):
    """Together mode must give each substrate a DISTINCT tagged output dir so the
    per-substrate fpocket runs do not collide/overwrite each other."""
    prepared = tmp_path / "prep"; prepared.mkdir()
    (prepared / "chai_0.pdb").write_text("dummy")
    out = tmp_path / "out"; out.mkdir()
    _stub_fpocket_env(monkeypatch, out)

    step = fp_mod.Fpocket(preparedfiles_dir=str(prepared), output_dir=str(out),
                          num_threads=1)
    row = pd.Series({"Entry": "P1", "docked_structure": "chai_0",
                     "substrate_smiles": "CCO.c1ccncc1"})
    res = step._process_single_row_with_fpocket(row)
    d0, d1 = res["ASvolume_dir_s0"], res["ASvolume_dir_s1"]
    assert d0 != d1
    # both are tagged (not the legacy bare name) and live under output_dir
    assert d0.startswith(str(out / "chai_0_")) and d0.endswith("_fpocket_output")
    assert d1.startswith(str(out / "chai_0_")) and d1.endswith("_fpocket_output")
    assert d0 != str(out / "chai_0_fpocket_output")
