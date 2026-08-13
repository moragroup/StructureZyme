"""REGRESSION (final integration review, Critical): in ``together`` mode
``run_boltz`` hands Boltz a COLLAPSED frame (substrate #0 in ``substrate_smiles``,
substrate #1 in ``cofactor_smiles``) because docko can only route one extra
ligand SMILES -- but the frame it RETURNS (and persists to boltz.pkl, which
becomes the downstream analysis frame) must restore the PACKED
``substrate_smiles`` so the per-substrate analysis steps can re-expand it via
``iter_substrates`` and emit their ``_s{i}`` columns.

Without the fix, Boltz's collapsed ``substrate_smiles`` ("CCO") flows downstream,
``iter_substrates`` sees a single fragment, every analysis step takes the
single-substrate legacy path, and substrate #1 is silently dropped from all
analysis.
"""
from __future__ import annotations

import types

import pandas as pd


class _FakeLayout:
    def __init__(self, root):
        self.root = root


class _FakeRuntime:
    num_threads = 1


class _FakePaths:
    boltz_cache_dir = "/tmp/boltz-cache"


class _FakeConfig:
    def __init__(self, mode):
        self.multi_substrate_mode = mode
        self.runtime = _FakeRuntime()
        self.paths = _FakePaths()

    def step_options(self, name):
        return {"enabled": True, "use_msa_server": False}


class _FakeCtx:
    def __init__(self, tmp_path, mode, frame):
        self.layout = _FakeLayout(tmp_path)
        self.config = _FakeConfig(mode)
        self._frame = frame

    def input_frames(self, spec):
        return [self._frame]

    def checkpoint_path(self, name):
        return self.layout.root / f"{name}.pkl"


class _RecordingBoltz:
    """Records the substrate/cofactor columns Boltz actually receives, then
    behaves like the real enzymetk Boltz (mutates df in place, adds output_dir,
    preserves row order)."""

    received = None

    def __init__(self, *args, **kwargs):
        pass

    def __rlshift__(self, df):  # df << Boltz(...)
        _RecordingBoltz.received = {
            "substrate_smiles": list(df["substrate_smiles"]),
            "cofactor_smiles": list(df["cofactor_smiles"]),
        }
        df["output_dir"] = ["/boltz/" + e for e in df["Entry"]]
        return df


def _run(monkeypatch, tmp_path, mode, frame):
    import structurezyme.step_runners as sr
    from enzymetk import dock_boltz_step

    _RecordingBoltz.received = None
    monkeypatch.setattr(dock_boltz_step, "Boltz", _RecordingBoltz)

    ctx = _FakeCtx(tmp_path, mode, frame)
    spec = types.SimpleNamespace(inputs=[])
    return sr.run_boltz(ctx, spec)


def test_together_restores_packed_substrate_downstream(monkeypatch, tmp_path):
    frame = pd.DataFrame({
        "Entry": ["P1", "P2"],
        "Sequence": ["MKAT", "MKAA"],
        "substrate_smiles": ["CCO.c1ccncc1", "CCC.O"],
    })
    out = _run(monkeypatch, tmp_path, "together", frame)

    # Boltz itself must have received the COLLAPSED frame (one substrate each,
    # 2nd routed to the cofactor slot).
    assert _RecordingBoltz.received["substrate_smiles"] == ["CCO", "CCC"]
    assert _RecordingBoltz.received["cofactor_smiles"] == ["c1ccncc1", "O"]

    # But the RETURNED frame (== boltz.pkl == downstream analysis frame) must
    # carry the PACKED substrate_smiles so iter_substrates re-expands both.
    assert list(out["substrate_smiles"]) == ["CCO.c1ccncc1", "CCC.O"]
    # row order preserved (positional restore is correct)
    assert list(out["Entry"]) == ["P1", "P2"]


def test_off_mode_leaves_substrate_smiles_untouched(monkeypatch, tmp_path):
    frame = pd.DataFrame({
        "Entry": ["P1"],
        "Sequence": ["MKAT"],
        "substrate_smiles": ["CCO"],
        "cofactor_smiles": ["[Fe]"],
    })
    out = _run(monkeypatch, tmp_path, "off", frame)
    # off mode: no collapse, no restore, Boltz sees exactly what came in.
    assert _RecordingBoltz.received["substrate_smiles"] == ["CCO"]
    assert _RecordingBoltz.received["cofactor_smiles"] == ["[Fe]"]
    assert list(out["substrate_smiles"]) == ["CCO"]
