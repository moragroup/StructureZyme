"""Squidly wrapper must apply mean_prob / mean_var thresholds locally.

Root cause pinned by this test:
The upstream `squidly` CLI accepts --mean-prob / --mean-var at its outer layer
but does not forward them to the inner `squidly.py` worker (see squidly
package's __main__.py around lines 308-336: the cmd list omits the flags).
The worker therefore always filters residues at its argparse defaults
(mean_prob=0.6, mean_var=0.225). For enzymes without a canonical catalytic
triad (e.g. flavin monooxygenases), those defaults produce an empty
Squidly_Ensemble_Residues, and any user-supplied lower threshold is silently
ignored.

Fix: `Squidly.execute()` must recompute `Squidly_CR_Position` locally from the
per-residue `mean` and `variance` arrays whenever the user has supplied
non-None mean_prob / mean_var. This bypasses the upstream bug.

These tests simulate the buggy upstream behaviour by having a fake
ActiveSitePred that returns *empty* Squidly_Ensemble_Residues but populated
mean / variance arrays -- exactly what the real bug produces.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest


class _FakeActiveSitePred:
    """Simulates upstream squidly: returns ensemble arrays but ignores flags."""

    def __init__(self, **kwargs):
        # Capture args that would have been passed to the (broken) upstream CLI
        self._args = kwargs.get("args") or []

    def execute(self, df):
        out = df.copy()
        # Simulate two enzymes with meaningful mean/variance signals but where
        # the upstream (buggy) selection at defaults 0.6/0.225 produced NO
        # residues -- Squidly_Ensemble_Residues is empty.
        means = []
        variances = []
        for _entry, seq in df[["Entry", "Sequence"]].values:
            n = len(seq)
            m = np.full(n, 0.001, dtype=float)
            v = np.full(n, 1e-6, dtype=float)
            # Put two moderate-confidence residues at positions 10 and 78
            # (0-indexed) with variance well below any sane threshold.
            m[10] = 0.088
            m[78] = 0.118
            v[10] = 0.028
            v[78] = 0.019
            means.append(m.tolist())
            variances.append(v.tolist())
        out["Squidly_Ensemble_Residues"] = [""] * len(out)  # upstream bug: empty
        out["mean"] = means
        out["variance"] = variances
        out["entropy"] = [[0.0] * len(s) for s in out["Sequence"]]
        out = out.rename(columns={"Entry": "label"})
        return out


def _make_df():
    seq = "M" + "A" * 400  # 401 aa
    return pd.DataFrame({
        "Entry": ["F5SYD3", "A0A410H4M7"],
        "Sequence": [seq, seq[:-1]],  # 401 and 400 aa
    })


def _patch_squidly_which(monkeypatch):
    import structurezyme.steps.squidly_step as sqmod
    monkeypatch.setattr(sqmod.shutil, "which", lambda name: "/fake/squidly")


def _patch_upstream(monkeypatch):
    """Redirect the lazy `from enzymetk...` import to our fake."""
    fake_module = types.ModuleType("enzymetk.predict_catalyticsite_step")
    fake_module.ActiveSitePred = _FakeActiveSitePred
    monkeypatch.setitem(
        __import__("sys").modules,
        "enzymetk.predict_catalyticsite_step",
        fake_module,
    )


def test_local_filter_at_permissive_thresholds(monkeypatch):
    from structurezyme.steps.squidly_step import Squidly

    _patch_squidly_which(monkeypatch)
    _patch_upstream(monkeypatch)

    step = Squidly(mean_prob=0.03, mean_var=0.5)
    out = step.execute(_make_df())

    # At mean_prob>0.03 and var<0.5, positions 10 and 78 (both 0-indexed) qualify.
    assert set(out["Squidly_CR_Position"]) == {"10|78"}


def test_local_filter_at_high_mean_prob_returns_empty(monkeypatch):
    from structurezyme.steps.squidly_step import Squidly

    _patch_squidly_which(monkeypatch)
    _patch_upstream(monkeypatch)

    step = Squidly(mean_prob=0.6, mean_var=0.5)
    out = step.execute(_make_df())

    # At mean_prob>0.6 no residue qualifies (max is 0.118).
    assert list(out["Squidly_CR_Position"]) == ["", ""]


def test_local_filter_at_high_var_still_permissive(monkeypatch):
    from structurezyme.steps.squidly_step import Squidly

    _patch_squidly_which(monkeypatch)
    _patch_upstream(monkeypatch)

    # Very permissive: catches positions with variance below 1.0 (all of them).
    step = Squidly(mean_prob=0.03, mean_var=1.0)
    out = step.execute(_make_df())

    assert set(out["Squidly_CR_Position"]) == {"10|78"}


def test_no_local_filter_when_thresholds_are_none(monkeypatch):
    """If user didn't supply mean_prob/mean_var, don't second-guess upstream.

    Even though upstream is buggy for our permissive-threshold use case, when
    the user leaves both at None they want squidly's default 0.6/0.225 behaviour
    (which is what the buggy upstream also produces, coincidentally). We keep
    Squidly_CR_Position exactly as upstream returned it (empty in this case,
    because our _FakeActiveSitePred returns empty Squidly_Ensemble_Residues).
    """
    from structurezyme.steps.squidly_step import Squidly

    _patch_squidly_which(monkeypatch)
    _patch_upstream(monkeypatch)

    step = Squidly()  # both mean_prob and mean_var default to None
    out = step.execute(_make_df())

    # Upstream returned empty; we didn't override.
    assert list(out["Squidly_CR_Position"]) == ["", ""]
