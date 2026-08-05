"""Squidly.execute selects top-N by mean when num_residues is set."""
from __future__ import annotations

import types

import numpy as np
import pandas as pd


class _FakeActiveSitePred:
    def __init__(self, **kwargs):
        self._args = kwargs.get("args") or []

    def execute(self, df):
        out = df.copy()
        means, variances = [], []
        for _entry, seq in df[["Entry", "Sequence"]].values:
            n = len(seq)
            m = np.full(n, 0.001, dtype=float)
            v = np.full(n, 1e-6, dtype=float)
            # Four candidate residues with distinct means; high variance on 200.
            m[10] = 0.30
            m[78] = 0.50
            m[120] = 0.40
            m[200] = 0.99
            v[200] = 5.0  # would fail any variance gate; top-N must ignore this.
            means.append(m.tolist())
            variances.append(v.tolist())
        out["Squidly_Ensemble_Residues"] = [""] * len(out)
        out["mean"] = means
        out["variance"] = variances
        out["entropy"] = [[0.0] * len(s) for s in out["Sequence"]]
        out = out.rename(columns={"Entry": "label"})
        return out


def _make_df():
    seq = "M" + "A" * 400
    return pd.DataFrame({"Entry": ["E1"], "Sequence": [seq]})


def _patch(monkeypatch):
    import structurezyme.steps.squidly_step as sqmod
    monkeypatch.setattr(sqmod.shutil, "which", lambda name: "/fake/squidly")
    fake_module = types.ModuleType("enzymetk.predict_catalyticsite_step")
    fake_module.ActiveSitePred = _FakeActiveSitePred
    monkeypatch.setitem(
        __import__("sys").modules,
        "enzymetk.predict_catalyticsite_step",
        fake_module,
    )


def test_top_n_ignores_variance_and_returns_n(monkeypatch):
    from structurezyme.steps.squidly_step import Squidly

    _patch(monkeypatch)
    step = Squidly(num_residues=3)
    out = step.execute(_make_df())
    # highest means: 200(0.99),78(0.50),120(0.40); output ascending index.
    assert list(out["Squidly_CR_Position"]) == ["78|120|200"]


def test_num_residues_overrides_thresholds_with_warning(monkeypatch, caplog):
    import logging
    from structurezyme.steps.squidly_step import Squidly

    _patch(monkeypatch)
    step = Squidly(num_residues=1, mean_prob=0.03, mean_var=0.5)
    with caplog.at_level(logging.WARNING):
        out = step.execute(_make_df())
    # top-1 by mean is index 200 (0.99), even though its variance fails a gate.
    assert list(out["Squidly_CR_Position"]) == ["200"]
    assert any("num_residues" in r.message for r in caplog.records)


def test_num_residues_none_uses_threshold_path(monkeypatch):
    from structurezyme.steps.squidly_step import Squidly

    _patch(monkeypatch)
    # No num_residues; permissive thresholds -> threshold path selects by gate.
    step = Squidly(mean_prob=0.03, mean_var=0.1)
    out = step.execute(_make_df())
    # index 200 has variance 5.0 (> 0.1) so it is excluded; others pass.
    assert list(out["Squidly_CR_Position"]) == ["10|78|120"]
