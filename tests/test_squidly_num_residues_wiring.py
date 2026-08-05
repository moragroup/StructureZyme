"""run_squidly forwards squidly_num_residues into Squidly()."""
from __future__ import annotations

import types

import pandas as pd


class _FakeSquidly:
    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        _FakeSquidly.last_kwargs = kwargs

    def execute(self, df):
        out = df.copy()
        out["Squidly_CR_Position"] = "10|20|30"
        return out


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
    def __init__(self, tmp_path, opts, seed_df):
        self.layout = _FakeLayout(tmp_path)
        self.config = _FakeConfig(opts)
        self._seed = seed_df

    def checkpoint_path(self, name):
        return self.layout.root / f"{name}.pkl"


def _run(monkeypatch, tmp_path, opts):
    import structurezyme.step_runners as sr
    from structurezyme.steps import squidly_step

    monkeypatch.setattr(squidly_step, "Squidly", _FakeSquidly)
    seed = pd.DataFrame({
        "Entry": ["E1"],
        "Sequence": ["MKAT"],
        "substrate_smiles": ["CCO"],
    })
    seed.to_pickle(tmp_path / "_input.pkl")
    ctx = _FakeCtx(tmp_path, opts, seed)
    spec = types.SimpleNamespace(inputs=[])
    sr.run_squidly(ctx, spec)
    return _FakeSquidly.last_kwargs


def test_forwards_num_residues(monkeypatch, tmp_path):
    kwargs = _run(monkeypatch, tmp_path,
                  {"enabled": True, "squidly_num_residues": 3})
    assert kwargs.get("num_residues") == 3


def test_num_residues_defaults_none(monkeypatch, tmp_path):
    kwargs = _run(monkeypatch, tmp_path, {"enabled": True})
    assert kwargs.get("num_residues") is None
