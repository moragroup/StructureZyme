"""run_squidly must NOT drop rows that lack catalytic residues.

Previously run_squidly deleted any row where both Squidly_CR_Position and
vina_residues were empty. That silently discarded enzymes without a canonical
catalytic triad (e.g. flavin monooxygenases), so they never reached chai/boltz
co-folding or any downstream analysis.

New contract: keep every row. Rows without residues get an empty
``catalytic_residues`` (so vina can skip them per-row) but still carry their
sequence/substrate into chai + boltz + the rest of the pipeline. The user is
notified which entries had no residues.
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
    def __init__(self, tmp_path, opts):
        self.layout = _FakeLayout(tmp_path)
        self.config = _FakeConfig(opts)

    def checkpoint_path(self, name):
        return self.layout.root / f"{name}.pkl"


class _PartialResidueSquidly:
    """Finds residues for E1 only; E2/E3 get none (empty string)."""

    def __init__(self, **kwargs):
        pass

    def execute(self, df):
        out = df.copy()
        out["Squidly_CR_Position"] = [
            "10|20|30" if e == "E1" else "" for e in out["Entry"]
        ]
        return out


def _run(monkeypatch, tmp_path, opts):
    import structurezyme.step_runners as sr
    from structurezyme.steps import squidly_step

    monkeypatch.setattr(squidly_step, "Squidly", _PartialResidueSquidly)

    seed = pd.DataFrame({
        "Entry": ["E1", "E2", "E3"],
        "Sequence": ["MKAT", "MKAA", "MKAG"],
        "substrate_smiles": ["CCO", "CCO", "CCO"],
        "vina_residues": ["", "", ""],
    })
    (tmp_path).mkdir(parents=True, exist_ok=True)
    seed.to_pickle(tmp_path / "_input.pkl")

    ctx = _FakeCtx(tmp_path, opts)
    spec = types.SimpleNamespace(inputs=[])
    return sr.run_squidly(ctx, spec)


def test_run_squidly_keeps_all_rows(monkeypatch, tmp_path):
    df = _run(monkeypatch, tmp_path, {"enabled": True})
    # All three enzymes survive, even the two without catalytic residues.
    assert sorted(df["Entry"].tolist()) == ["E1", "E2", "E3"]


def test_residueless_rows_have_empty_catalytic_residues(monkeypatch, tmp_path):
    df = _run(monkeypatch, tmp_path, {"enabled": True})
    by_entry = df.set_index("Entry")["catalytic_residues"]
    assert by_entry["E1"] == "10|20|30"
    assert by_entry["E2"] == ""
    assert by_entry["E3"] == ""
