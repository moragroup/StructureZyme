"""Tests for `structurezyme.utils.helpers.extract_docking_metrics`.

This module reduces per-Entry dict-valued docking-quality metrics
(`chai_ptm`, `boltz2_confidence_score`, ...) to per-pose scalars using
the `docked_structure` column to key into the dicts.

Regression context: on the fmo-fad-01 SLURM run, the reducer failed to
match keys for every chai pose because it stripped only the trailing
`_relaxed` segment (leaving `<Entry>_<idx>_chai`) while the dict keys
are `<Entry>_<idx>` (no tool suffix). All 72 rows ended up with
`chai_ptm=NaN` in `checkpoints/ligand_rmsd.pkl`. The boltz side
accidentally worked because its dicts have exactly one entry, so a
single-item fallback rescued the lookup.

These tests lock the pose-id derivation and confirm the reducer
populates scalar columns for the correct `tool` masks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from structurezyme.utils.helpers import extract_docking_metrics


CHAI_PTM_DICT = {
    "E1_0": 0.91,
    "E1_1": 0.92,
    "E1_2": 0.93,
    "E1_3": 0.94,
    "E1_4": 0.95,
}
BOLTZ_CONF_DICT = {"E1_model_0": 0.88}


def _make_row_frame():
    """A minimal frame with two chai poses + two boltz poses (matching the
    production layout after fastrelax + protein_rmsd row explosion), where
    both `_chai` / `_boltz` and `_chai_relaxed` / `_boltz_relaxed` variants
    are present for the same underlying pose id."""
    return pd.DataFrame({
        "Entry": ["E1"] * 4,
        "tool": ["chai", "chai", "boltz", "boltz"],
        "docked_structure": [
            "E1_1_chai",
            "E1_2_chai_relaxed",
            "E1_model_0_boltz",
            "E1_model_0_boltz_relaxed",
        ],
        "chai_ptm": [dict(CHAI_PTM_DICT)] * 4,
        "boltz2_confidence_score": [dict(BOLTZ_CONF_DICT)] * 4,
    })


def test_chai_pose_lookup_populates_scalars_from_relaxed_names():
    """Regression: `<Entry>_<idx>_chai_relaxed` must resolve to the dict key
    `<Entry>_<idx>` (both `_relaxed` and `_chai` must be stripped)."""
    df = _make_row_frame()

    out = extract_docking_metrics(df)

    # Chai rows: both unrelaxed and relaxed variants must find their keys.
    chai_rows = out[out["tool"] == "chai"].sort_values("docked_structure")
    assert chai_rows["chai_ptm"].tolist() == [0.92, 0.93], (
        f"expected chai_ptm=[0.92, 0.93], got {chai_rows['chai_ptm'].tolist()}"
    )


def test_boltz_pose_lookup_populates_scalars_from_relaxed_names():
    """Boltz rows: `<Entry>_model_0_boltz_relaxed` must resolve to
    `<Entry>_model_0` (both `_relaxed` and `_boltz` must be stripped)."""
    df = _make_row_frame()

    out = extract_docking_metrics(df)

    boltz_rows = out[out["tool"] == "boltz"]
    # Both boltz rows point at the same underlying pose id, so both scalars
    # should equal the sole confidence value.
    assert boltz_rows["boltz2_confidence_score"].tolist() == [0.88, 0.88]


def test_chai_metrics_are_nan_on_non_chai_rows():
    """chai_* columns must be NaN on non-chai rows (unchanged behavior)."""
    df = _make_row_frame()

    out = extract_docking_metrics(df)

    non_chai = out[out["tool"] != "chai"]
    assert non_chai["chai_ptm"].isna().all()


def test_boltz_metrics_are_nan_on_non_boltz_rows():
    """boltz_* columns must be NaN on non-boltz rows (unchanged behavior)."""
    df = _make_row_frame()

    out = extract_docking_metrics(df)

    non_boltz = out[out["tool"] != "boltz"]
    assert non_boltz["boltz2_confidence_score"].isna().all()


def test_reducer_regression_reproduces_fmo_fad_01_shapes():
    """Recreate the exact `docked_structure` naming from the fmo-fad-01
    SLURM run: after fastrelax + protein_rmsd, chai rows are named
    `<Entry>_<0..4>_chai_relaxed` and boltz rows are named
    `<Entry>_model_0_boltz` / `<Entry>_model_0_boltz_relaxed`. Prior to
    the fix, chai_ptm was NaN on every row. Assert the post-fix state:
    every chai row resolves to its dict entry."""
    df = pd.DataFrame({
        "Entry": ["F5SYD3"] * 4 + ["A0A063Y6V3"] * 4,
        "tool": ["chai", "chai", "boltz", "boltz"] * 2,
        "docked_structure": [
            "F5SYD3_1_chai_relaxed",
            "F5SYD3_2_chai_relaxed",
            "F5SYD3_model_0_boltz",
            "F5SYD3_model_0_boltz_relaxed",
            "A0A063Y6V3_1_chai_relaxed",
            "A0A063Y6V3_4_chai_relaxed",
            "A0A063Y6V3_model_0_boltz",
            "A0A063Y6V3_model_0_boltz_relaxed",
        ],
        "chai_ptm": (
            [{"F5SYD3_0": 0.923, "F5SYD3_1": 0.924, "F5SYD3_2": 0.925,
              "F5SYD3_3": 0.923, "F5SYD3_4": 0.924}] * 4 +
            [{"A0A063Y6V3_0": 0.911, "A0A063Y6V3_1": 0.912, "A0A063Y6V3_2": 0.913,
              "A0A063Y6V3_3": 0.914, "A0A063Y6V3_4": 0.915}] * 4
        ),
        "boltz2_confidence_score": (
            [{"F5SYD3_model_0": 0.967}] * 4 +
            [{"A0A063Y6V3_model_0": 0.973}] * 4
        ),
    })

    out = extract_docking_metrics(df)

    # Every chai row must have its ptm populated (this is the exact bug).
    chai = out[out["tool"] == "chai"]
    assert chai["chai_ptm"].isna().sum() == 0, (
        "Regression: all chai_ptm entries should be populated. "
        f"Got NaN on {chai['chai_ptm'].isna().sum()} of {len(chai)} chai rows."
    )
    # Specific values.
    expected_by_docked = {
        "F5SYD3_1_chai_relaxed": 0.924,
        "F5SYD3_2_chai_relaxed": 0.925,
        "A0A063Y6V3_1_chai_relaxed": 0.912,
        "A0A063Y6V3_4_chai_relaxed": 0.915,
    }
    for ds, expected in expected_by_docked.items():
        actual = chai.loc[chai["docked_structure"] == ds, "chai_ptm"].iloc[0]
        assert actual == pytest.approx(expected), f"{ds}: {actual} != {expected}"

    # Every boltz row must have its confidence populated.
    boltz = out[out["tool"] == "boltz"]
    assert boltz["boltz2_confidence_score"].isna().sum() == 0
