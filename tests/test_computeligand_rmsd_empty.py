"""compute_normalized_ligand_rmsd_stats must handle empty DataFrames.

When every PDB pair for a run fails RDKit sanitization (e.g. the previous
fmo18 run's 100% valence-error mode caused by the fastrelax ligand-template
corruption bug fixed in 3df8a1c), `LigandRMSD.__execute` reaches line 378
with an empty `rmsd_values` list, so `pd.DataFrame(rmsd_values)` produces a
0-column DataFrame. Passing that into `compute_normalized_ligand_rmsd_stats`
then raises `KeyError: 'tool1'` on line 105 because the column literally
does not exist.

The step should degrade gracefully -- returning an empty enriched DataFrame
-- so that downstream steps see "no rmsd data for this run" rather than
crashing the whole pipeline.
"""

from __future__ import annotations

import pandas as pd

from structurezyme.steps.computeligandRMSD_step import (
    compute_normalized_ligand_rmsd_stats,
)


def test_compute_normalized_ligand_rmsd_stats_handles_empty_dataframe():
    """0-row / 0-column DataFrame (all pose pairs failed RDKit processing)."""
    empty = pd.DataFrame()

    # Must not raise KeyError.
    result = compute_normalized_ligand_rmsd_stats(empty)

    # Contract: result is a DataFrame. It's empty (no per-entry stats to
    # compute) but the pipeline can continue.
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_compute_normalized_ligand_rmsd_stats_handles_zero_rows_but_expected_columns():
    """Empty rows but the expected columns exist (defensive DataFrame init).

    Some upstream call sites might pre-populate the frame with the expected
    schema even when there are no pairs. Should still not crash.
    """
    df = pd.DataFrame(
        columns=["Entry", "tool1", "tool2", "ligand_rmsd",
                 "docked_structure1", "docked_structure2"]
    )
    result = compute_normalized_ligand_rmsd_stats(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
