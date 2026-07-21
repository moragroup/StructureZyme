"""run_vina must dock only rows that have catalytic residues.

Rows with an empty ``catalytic_residues`` (enzymes where squidly found no
catalytic triad and no vina_residues were given) cannot be docked by vina --
there is no box to place. Instead of crashing or dropping them, run_vina skips
vina for those rows and passes them through unchanged (vina_dir = NaN), keeping
their chai_dir/boltz_dir so downstream metrics fall back to the co-folded poses.
"""
from __future__ import annotations

import types

import pandas as pd


class _FakeLayout:
    def __init__(self, root):
        self.root = root


class _FakeRuntime:
    num_threads = 1


class _FakeConfig:
    def __init__(self, opts):
        self._opts = opts
        self.runtime = _FakeRuntime()

    def step_options(self, name):
        return self._opts


class _FakeCtx:
    def __init__(self, tmp_path, opts, frame):
        self.layout = _FakeLayout(tmp_path)
        self.config = _FakeConfig(opts)
        self._frame = frame

    def input_frames(self, spec):
        return [self._frame]

    def checkpoint_path(self, name):
        return self.layout.root / f"{name}.pkl"


class _FakeVina:
    """Records which Entries it was asked to dock; returns an output_dir path."""

    docked_entries: list = []

    def __init__(self, *args, **kwargs):
        pass

    def __rlshift__(self, df):  # df << Vina(...)
        return self.__execute(df)

    def __execute(self, df):
        out = df.copy()
        _FakeVina.docked_entries = out["Entry"].tolist()
        out["output_dir"] = ["/vina/" + e for e in out["Entry"]]
        return out


def _run(monkeypatch, tmp_path):
    import structurezyme.step_runners as sr
    from structurezyme.steps import dock_vina_step

    _FakeVina.docked_entries = []
    monkeypatch.setattr(dock_vina_step, "Vina", _FakeVina)

    frame = pd.DataFrame({
        "Entry": ["E1", "E2", "E3"],
        "Sequence": ["MKAT", "MKAA", "MKAG"],
        "substrate_smiles": ["CCO", "CCO", "CCO"],
        "substrate_name": ["s", "s", "s"],
        "catalytic_residues": ["10|20|30", "", ""],
        "chai_dir": ["/chai/E1", "/chai/E2", "/chai/E3"],
        "boltz_dir": ["/boltz/E1", "/boltz/E2", "/boltz/E3"],
    })

    ctx = _FakeCtx(tmp_path, {"enabled": True}, frame)
    spec = types.SimpleNamespace(inputs=[])
    return sr.run_vina(ctx, spec)


def test_vina_only_docks_rows_with_residues(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path)
    # Only E1 has catalytic residues, so only E1 is sent to vina.
    assert _FakeVina.docked_entries == ["E1"]


def test_all_rows_survive_with_dirs_preserved(monkeypatch, tmp_path):
    df = _run(monkeypatch, tmp_path)
    assert sorted(df["Entry"].tolist()) == ["E1", "E2", "E3"]
    by = df.set_index("Entry")
    # Residue-less rows keep their co-folded dirs and have no vina_dir.
    assert by.loc["E2", "chai_dir"] == "/chai/E2"
    assert by.loc["E2", "boltz_dir"] == "/boltz/E2"
    assert pd.isna(by.loc["E2", "vina_dir"])
    # Docked row gets a vina_dir.
    assert by.loc["E1", "vina_dir"] == "/vina/E1"
