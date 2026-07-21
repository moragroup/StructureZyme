"""run_superimpose must not drop rows whose vina files are missing per-row.

Per-row vina skip (run_vina) can leave a row with valid chai/boltz files but
no vina files, even when vina is globally enabled. The legacy branch dropped
those rows entirely (df[valid_file_list("vina_files_for_superimposition")]),
so downstream steps only saw the vina-docked rows.

New contract:
- Rows with valid vina + chai (+ boltz) files run the full 3-way superimpose
  (vina<->chai, vina<->boltz, chai<->boltz).
- Rows with valid chai + boltz but no vina files run only chai<->boltz.
- Every surviving row (i.e. every row with at least valid chai+boltz) is
  present in the returned frame.
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
    def __init__(self, vina, fastrelax):
        self._flags = {"vina": vina, "fastrelax": fastrelax}
        self.runtime = _FakeRuntime()

    def is_enabled(self, name):
        return self._flags.get(name, True)

    def step_options(self, name):
        return {"enabled": self._flags.get(name, True)}


class _FakeCtx:
    def __init__(self, tmp_path, vina, frame):
        self.layout = _FakeLayout(tmp_path)
        self.config = _FakeConfig(vina=vina, fastrelax=False)
        self._frame = frame

    def input_frames(self, spec):
        return [self._frame]

    def checkpoint_path(self, name):
        p = self.layout.root / f"{name}.pkl"
        # Seed the prepare_files checkpoint so _superimpose_input reads it.
        if name == "prepare_files" and not p.exists():
            self._frame.to_pickle(p)
        return p


class _RecordingSuperimpose:
    """Records (name1, name2, entries) for every call; is a no-op pipeline step."""

    calls: list = []

    def __init__(self, structure_1, structure_2, output_dir, name1, name2, num_threads):
        self.name1 = name1
        self.name2 = name2
        self.s1 = structure_1
        self.s2 = structure_2

    def __rshift__(self, other):  # allow >> chaining
        return _Chain([self, other])

    def __rlshift__(self, df):  # df << single-step
        self._record(df)
        return df

    def _record(self, df):
        # Emulate the class-level filter: only rows where both structure cols
        # are non-empty list-of-files reach the actual work loop.
        mask = df[self.s1].apply(lambda v: isinstance(v, (list, tuple)) and len(v) > 0) & \
               df[self.s2].apply(lambda v: isinstance(v, (list, tuple)) and len(v) > 0)
        _RecordingSuperimpose.calls.append({
            "pair": (self.name1, self.name2),
            "entries": sorted(df.loc[mask, "Entry"].astype(str).tolist()),
        })


class _Chain:
    def __init__(self, steps):
        self.steps = steps

    def __rshift__(self, other):
        return _Chain(self.steps + [other])

    def __rlshift__(self, df):
        for s in self.steps:
            if isinstance(s, _RecordingSuperimpose):
                s._record(df)
            # _FakeSave: no-op
        return df


class _FakeSave:
    def __init__(self, *a, **k):
        pass

    def __rshift__(self, other):
        return _Chain([self, other])

    def __rlshift__(self, df):
        return df


def _make_frame(tmp_path):
    """Two rows have vina+chai+boltz files; one row has only chai+boltz."""
    d = tmp_path / "poses"
    d.mkdir()
    files = {}
    for name in ["v1", "v2", "c1", "c2", "b1", "b2"]:
        p = d / f"{name}.pdb"
        p.write_text("REMARK  fake\n")
        files[name] = str(p)
    return pd.DataFrame({
        "Entry": ["P41365", "F5SYD3", "A0A410H4M7"],
        "vina_files_for_superimposition": [
            [files["v1"], files["v2"]], None, None,
        ],
        "chai_files_for_superimposition": [
            [files["c1"], files["c2"]],
            [files["c1"], files["c2"]],
            [files["c1"], files["c2"]],
        ],
        "boltz_files_for_superimposition": [
            [files["b1"], files["b2"]],
            [files["b1"], files["b2"]],
            [files["b1"], files["b2"]],
        ],
    })


def _run(monkeypatch, tmp_path):
    import structurezyme.step_runners as sr
    from structurezyme.steps import superimposestructures_step, save_step

    _RecordingSuperimpose.calls = []
    monkeypatch.setattr(superimposestructures_step, "SuperimposeStructures",
                        _RecordingSuperimpose)
    monkeypatch.setattr(save_step, "Save", _FakeSave)

    frame = _make_frame(tmp_path)
    ctx = _FakeCtx(tmp_path, vina=True, frame=frame)
    spec = types.SimpleNamespace(inputs=[])
    df_out = sr.run_superimpose(ctx, spec)
    return df_out, _RecordingSuperimpose.calls


def test_run_superimpose_keeps_rows_without_vina(monkeypatch, tmp_path):
    df_out, _ = _run(monkeypatch, tmp_path)
    assert sorted(df_out["Entry"].astype(str).tolist()) == \
        ["A0A410H4M7", "F5SYD3", "P41365"]


def test_chai_boltz_superimpose_runs_for_every_row(monkeypatch, tmp_path):
    _, calls = _run(monkeypatch, tmp_path)
    chai_boltz = [c for c in calls if c["pair"] == ("chai", "boltz")]
    assert chai_boltz, "chai<->boltz superimpose was not called at all"
    entries = set().union(*(set(c["entries"]) for c in chai_boltz))
    assert entries == {"P41365", "F5SYD3", "A0A410H4M7"}


def test_vina_superimpose_runs_only_for_rows_with_vina(monkeypatch, tmp_path):
    _, calls = _run(monkeypatch, tmp_path)
    for pair_name in [("vina", "chai"), ("vina", "boltz")]:
        pair_calls = [c for c in calls if c["pair"] == pair_name]
        assert pair_calls, f"{pair_name} superimpose was not called at all"
        entries = set().union(*(set(c["entries"]) for c in pair_calls))
        # Only P41365 had valid vina files.
        assert entries == {"P41365"}, f"{pair_name} saw entries {entries}"
