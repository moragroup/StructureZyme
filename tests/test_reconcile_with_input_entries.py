"""Unit tests for the _reconcile_with_input_entries helper.

The helper is the single point of defense against step implementations that
drop rows for entries whose per-artifact processing failed. Downstream
runners (ligand_rmsd, and potentially others) call it to guarantee that
every entry from the input frame survives with at least a NaN-metric row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def test_reconcile_no_missing_entries_is_identity():
    from structurezyme.step_runners import _reconcile_with_input_entries

    returned = pd.DataFrame({
        "Entry": ["E1", "E2"],
        "metric": [1.0, 2.0],
    })
    source = pd.DataFrame({
        "Entry": ["E1", "E2"],
        "substrate_smiles": ["CCO", "CCN"],
    })
    out = _reconcile_with_input_entries(returned, source)

    assert sorted(out["Entry"].tolist()) == ["E1", "E2"]
    assert out.loc[out["Entry"] == "E1", "metric"].iloc[0] == 1.0
    assert out.loc[out["Entry"] == "E2", "metric"].iloc[0] == 2.0


def test_reconcile_adds_nan_row_for_missing_entry():
    from structurezyme.step_runners import _reconcile_with_input_entries

    returned = pd.DataFrame({
        "Entry": ["E1"],
        "metric": [1.0],
    })
    source = pd.DataFrame({
        "Entry": ["E1", "E2", "E3"],
        "substrate_smiles": ["CCO", "CCN", "CN"],
    })
    out = _reconcile_with_input_entries(returned, source)

    assert sorted(out["Entry"].tolist()) == ["E1", "E2", "E3"]
    # Metric is NaN for the reintroduced entries.
    e2 = out.loc[out["Entry"] == "E2"].iloc[0]
    e3 = out.loc[out["Entry"] == "E3"].iloc[0]
    assert pd.isna(e2["metric"])
    assert pd.isna(e3["metric"])
    # Source columns are carried over.
    assert e2["substrate_smiles"] == "CCN"
    assert e3["substrate_smiles"] == "CN"


def test_reconcile_preserves_returned_column_order():
    from structurezyme.step_runners import _reconcile_with_input_entries

    returned = pd.DataFrame({
        "Entry": ["E1"],
        "docked_structure": ["s1"],
        "ligand_rmsd": [0.5],
    })
    source = pd.DataFrame({
        "Entry": ["E1", "E2"],
        "substrate_smiles": ["CCO", "CCN"],
    })
    out = _reconcile_with_input_entries(returned, source)

    # Returned's columns come first, in order; source-only columns get appended.
    assert list(out.columns[:3]) == ["Entry", "docked_structure", "ligand_rmsd"]
    assert "substrate_smiles" in out.columns


def test_reconcile_missing_entry_carries_source_paths():
    """When source has boltz_dir / vina_dir / etc. and the returned frame
    dropped an entry, the reintroduced row must carry over source-side paths
    (they are what downstream steps use to resolve files on disk)."""
    from structurezyme.step_runners import _reconcile_with_input_entries

    returned = pd.DataFrame({
        "Entry": ["E1"],
        "docked_structure": ["E1_a"],
        "ligand_rmsd": [0.42],
    })
    source = pd.DataFrame({
        "Entry": ["E1", "E2"],
        "boltz_dir": ["/b/E1", "/b/E2"],
        "vina_dir": ["/v/E1", np.nan],
        "substrate_smiles": ["CCO", "CCN"],
    })
    out = _reconcile_with_input_entries(returned, source)

    e2 = out.loc[out["Entry"] == "E2"].iloc[0]
    assert e2["boltz_dir"] == "/b/E2"
    # NaN in the source stays NaN.
    assert pd.isna(e2["vina_dir"])
    assert e2["substrate_smiles"] == "CCN"
    # Metric-side columns from returned are NaN for this entry.
    assert pd.isna(e2["docked_structure"])
    assert pd.isna(e2["ligand_rmsd"])


def test_reconcile_empty_source_is_identity():
    from structurezyme.step_runners import _reconcile_with_input_entries

    returned = pd.DataFrame({
        "Entry": ["E1"],
        "metric": [1.0],
    })
    source = pd.DataFrame({"Entry": [], "substrate_smiles": []})
    out = _reconcile_with_input_entries(returned, source)
    assert out.equals(returned)


def test_reconcile_source_without_entry_col_is_identity():
    from structurezyme.step_runners import _reconcile_with_input_entries

    returned = pd.DataFrame({"Entry": ["E1"], "metric": [1.0]})
    source = pd.DataFrame({"other": [1, 2, 3]})
    out = _reconcile_with_input_entries(returned, source)
    assert out.equals(returned)
