"""run_ligand_rmsd must not drop entries whose pose comparisons all failed.

LigandRMSD iterates the on-disk superimposed PDBs and skips pose pairs whose
rdkit sanitize/align raises (Valence error, "No sub-structure match", etc.).
When *every* pair for a given Entry fails, the returned structures_df has no
rows for that Entry, and downstream steps lose the enzyme entirely.

Contract:
- run_ligand_rmsd returns a frame containing at least one row per Entry that
  was present in the input frame.
- Missing entries are reintroduced with NaN metric columns and their original
  identifier / path columns carried over from the input frame.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd


class _FakeLayout:
    def __init__(self, root):
        self.root = root


class _FakeRuntime:
    num_threads = 1


class _FakeConfig:
    runtime = _FakeRuntime()

    def is_enabled(self, name):
        return True

    def step_options(self, name):
        return {}


class _FakeCtx:
    def __init__(self, tmp_path, frame):
        self.layout = _FakeLayout(tmp_path)
        self.config = _FakeConfig()
        self._frame = frame

    def input_frames(self, spec):
        return [self._frame]

    def checkpoint_path(self, name):
        return self.layout.root / f"{name}.pkl"


class _PartialLigandRMSD:
    """Fake LigandRMSD that only produces rows for a subset of Entries.

    Emulates the real class: it returns (pairwise_df, structures_df) via
    ``df << LigandRMSD(...)``. Rows for Entries whose PDB pairs all failed
    rdkit processing are simply absent from both frames.
    """

    keep_entries = {"P41365"}

    def __init__(self, *a, **k):
        pass

    def __rlshift__(self, df):  # df << LigandRMSD(...)
        entries = [e for e in df["Entry"].tolist() if e in self.keep_entries]
        pairwise = pd.DataFrame({
            "Entry": entries,
            "docked_structure1": [f"{e}_a" for e in entries],
            "docked_structure2": [f"{e}_b" for e in entries],
            "ligand_rmsd": [0.5 for _ in entries],
        })
        structures = pd.DataFrame({
            "Entry": entries,
            "docked_structure": [f"{e}_a" for e in entries],
            "tool": ["chai" for _ in entries],
            "is_best": [True for _ in entries],
        })
        return pairwise, structures


def _run(monkeypatch, tmp_path):
    import structurezyme.step_runners as sr
    from structurezyme.steps import computeligandRMSD_step

    monkeypatch.setattr(computeligandRMSD_step, "LigandRMSD", _PartialLigandRMSD)

    # extract_docking_metrics is a pass-through in this test; monkeypatch to
    # avoid depending on real columns.
    from structurezyme.utils import helpers
    monkeypatch.setattr(helpers, "extract_docking_metrics", lambda d: d,
                        raising=False)

    frame = pd.DataFrame({
        "Entry": ["P41365", "F5SYD3", "A0A410H4M7"],
        "substrate_smiles": ["CCO", "C1=CC=CC=C1", "CCN"],
        "boltz_dir": ["/b/P", "/b/F", "/b/A"],
    })

    ctx = _FakeCtx(tmp_path, frame)
    spec = types.SimpleNamespace(inputs=[])
    return sr.run_ligand_rmsd(ctx, spec)


def test_run_ligand_rmsd_keeps_entries_with_all_failed_pairs(monkeypatch, tmp_path):
    (tmp_path / "superimposition").mkdir()
    df_out = _run(monkeypatch, tmp_path)
    assert sorted(df_out["Entry"].astype(str).tolist()) == \
        sorted(["A0A410H4M7", "F5SYD3", "P41365"])


def test_run_ligand_rmsd_missing_entries_have_nan_metrics(monkeypatch, tmp_path):
    (tmp_path / "superimposition").mkdir()
    df_out = _run(monkeypatch, tmp_path)
    # For entries that had no successful pair, ligand_rmsd metric columns
    # should be NaN (not absent) so downstream steps can still see the row.
    missing = df_out[df_out["Entry"].isin(["F5SYD3", "A0A410H4M7"])]
    # docked_structure is a metric-side column that only exists for entries
    # with at least one successful pair.
    if "docked_structure" in missing.columns:
        assert missing["docked_structure"].isna().all()
    # substrate_smiles is carried over from the source frame.
    assert missing["substrate_smiles"].notna().all()
