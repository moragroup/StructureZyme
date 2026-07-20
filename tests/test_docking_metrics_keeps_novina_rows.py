"""run_docking_metrics must keep rows that have a co-folded pose but no vina.

When vina is enabled but a row was skipped by the per-row vina logic (no
catalytic residues), that row has vina_dir = NaN yet a valid boltz_dir/chai_dir.
The old filter kept only vina_dir.notna() rows when vina ran, silently dropping
those enzymes. A row should survive if it has *either* a vina pose or a boltz
co-folded pose.
"""
from __future__ import annotations

import types

import pandas as pd


class _FakeLayout:
    def __init__(self, root):
        self.root = root


class _FakeConfig:
    def __init__(self, vina_enabled):
        self._vina = vina_enabled

    def is_enabled(self, name):
        return self._vina if name == "vina" else True


class _FakeCtx:
    def __init__(self, tmp_path, frame, vina_enabled):
        self.layout = _FakeLayout(tmp_path)
        self.config = _FakeConfig(vina_enabled)
        self._frame = frame

    def input_frames(self, spec):
        return [self._frame]

    def checkpoint_path(self, name):
        return self.layout.root / f"{name}.pkl"


class _FakeDockingMetrics:
    def __init__(self, *args, **kwargs):
        pass

    def __rrshift__(self, other):  # >> chaining target
        return self

    def __rlshift__(self, df):  # df << (DockingMetrics >> Save)
        _FakeDockingMetrics.seen_entries = df["Entry"].tolist()
        return df


def _run(monkeypatch, tmp_path):
    import structurezyme.step_runners as sr
    from structurezyme.steps import extract_docking_metrics_step, save_step

    # Compose DockingMetrics(...) >> Save(...) into a single fake that records
    # which rows reached the metrics step.
    fake = _FakeDockingMetrics()

    class _FakeDM:
        def __init__(self, *a, **k):
            pass

        def __rshift__(self, other):
            return fake

    class _FakeSave:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(extract_docking_metrics_step, "DockingMetrics", _FakeDM)
    monkeypatch.setattr(save_step, "Save", _FakeSave)

    frame = pd.DataFrame({
        "Entry": ["E1", "E2"],
        "vina_dir": ["/vina/E1", pd.NA],   # E2 skipped vina
        "boltz_dir": ["/boltz/E1", "/boltz/E2"],
    })
    ctx = _FakeCtx(tmp_path, frame, vina_enabled=True)
    spec = types.SimpleNamespace(inputs=[])
    sr.run_docking_metrics(ctx, spec)
    return _FakeDockingMetrics.seen_entries


def test_metrics_keeps_row_with_boltz_but_no_vina(monkeypatch, tmp_path):
    seen = _run(monkeypatch, tmp_path)
    assert sorted(seen) == ["E1", "E2"]
