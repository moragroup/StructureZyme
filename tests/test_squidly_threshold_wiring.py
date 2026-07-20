"""run_squidly must forward the squidly threshold options into Squidly().

The squidly `run` CLI selects catalytic residues where mean_prob > --mean-prob
AND variance < --mean-var (defaults 0.6 / 0.225). Lowering mean-prob / raising
mean-var makes squidly predict lower-confidence residues -- needed for enzymes
without a canonical catalytic triad (e.g. flavin monooxygenases). run_squidly
previously only wired squidly_as_threshold, so mean_prob/mean_var set in the
run config were silently ignored. These tests pin the wiring.
"""
from __future__ import annotations

import types

import pandas as pd
import pytest


class _FakeSquidly:
    """Captures constructor kwargs and returns the input df unchanged."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        _FakeSquidly.last_kwargs = kwargs

    def execute(self, df):
        # Emulate squidly finding one residue so the row is not dropped.
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


def test_run_squidly_forwards_mean_prob_and_mean_var(monkeypatch, tmp_path):
    kwargs = _run(
        monkeypatch,
        tmp_path,
        {"enabled": True, "squidly_mean_prob": 0.3, "squidly_mean_var": 0.5},
    )
    assert kwargs.get("mean_prob") == 0.3
    assert kwargs.get("mean_var") == 0.5


def test_run_squidly_mean_prob_defaults_none(monkeypatch, tmp_path):
    kwargs = _run(monkeypatch, tmp_path, {"enabled": True})
    assert kwargs.get("mean_prob") is None
    assert kwargs.get("mean_var") is None
