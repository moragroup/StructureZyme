"""Tests for the energy-aware best-pose selector in computeligandRMSD_step."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from structurezyme.steps.computeligandRMSD_step import (
    pose_energy,
    select_best_docked_structures,
)


def test_pose_energy_chai_lookup():
    scores = {"chai": {"P41365_0": -12.5}, "boltz": {}, "vina": {}}
    assert pose_energy("P41365_0_chai", scores) == -12.5


def test_pose_energy_strips_relaxed_suffix():
    scores = {"chai": {"P41365_0": -12.5}, "boltz": {}, "vina": {}}
    assert pose_energy("P41365_0_chai_relaxed", scores) == -12.5


def test_pose_energy_boltz_model_key():
    scores = {"chai": {}, "boltz": {"P41365_model_0": -9.0}, "vina": {}}
    assert pose_energy("P41365_model_0_boltz_relaxed", scores) == -9.0


def test_pose_energy_vina_integer_key():
    scores = {"chai": {}, "boltz": {}, "vina": {3: -7.2}}
    assert pose_energy("P41365_3_vina_relaxed", scores) == -7.2


def test_pose_energy_missing_returns_none():
    scores = {"chai": {"P41365_0": -12.5}, "boltz": {}, "vina": {}}
    assert pose_energy("P41365_9_chai", scores) is None


def test_pose_energy_none_scores_returns_none():
    assert pose_energy("P41365_0_chai", None) is None
    assert pose_energy("P41365_0_chai", {}) is None


def _pairwise(entry, structures, rmsd_lookup):
    """Build a pairwise rmsd_df for one entry.

    structures: dict {name: tool}. rmsd_lookup: dict {frozenset({a,b}): rmsd}.
    """
    rows = []
    names = list(structures)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append({
                "Entry": entry,
                "docked_structure1": a,
                "docked_structure2": b,
                "tool1": structures[a],
                "tool2": structures[b],
                "ligand_rmsd": rmsd_lookup[frozenset({a, b})],
            })
    return pd.DataFrame(rows)


def test_fused_rank_energy_overrides_biased_geometry():
    # 3 chai poses clustered together + 1 boltz pose off to the side.
    # Geometry (min-per-tool) favors the chai pose nearest boltz, but the
    # boltz pose has clearly better (lower) energy and must win fused_rank.
    structs = {
        "E_0_chai": "chai", "E_1_chai": "chai", "E_2_chai": "chai",
        "E_0_boltz": "boltz",
    }
    rmsd = {
        frozenset({"E_0_chai", "E_1_chai"}): 0.5,
        frozenset({"E_0_chai", "E_2_chai"}): 0.5,
        frozenset({"E_1_chai", "E_2_chai"}): 0.5,
        frozenset({"E_0_chai", "E_0_boltz"}): 2.0,  # nearest chai to boltz
        frozenset({"E_1_chai", "E_0_boltz"}): 5.0,
        frozenset({"E_2_chai", "E_0_boltz"}): 5.0,
    }
    df = _pairwise("E", structs, rmsd)
    scores = {"E": {"chai": {"E_0": -5.0, "E_1": -5.0, "E_2": -5.0},
                    "boltz": {"E_0": -20.0}, "vina": {}}}
    out = select_best_docked_structures(df, scores)
    fused = out[out["method"] == "fused_rank"]
    assert len(fused) == 1
    assert fused.iloc[0]["best_structure"] == "E_0_boltz"
    assert fused.iloc[0]["tool"] == "boltz"


def test_fused_rank_no_energy_matches_geometry():
    # No energy passed -> fused_rank winner == inter_tool_min_per_tool winner.
    structs = {"E_0_chai": "chai", "E_1_chai": "chai", "E_0_boltz": "boltz"}
    rmsd = {
        frozenset({"E_0_chai", "E_1_chai"}): 0.5,
        frozenset({"E_0_chai", "E_0_boltz"}): 1.0,
        frozenset({"E_1_chai", "E_0_boltz"}): 3.0,
    }
    df = _pairwise("E", structs, rmsd)
    out = select_best_docked_structures(df, None)
    geo = out[out["method"] == "inter_tool_min_per_tool"].iloc[0]["best_structure"]
    fused = out[out["method"] == "fused_rank"].iloc[0]["best_structure"]
    assert fused == geo


def test_fused_rank_partial_energy_does_not_crash():
    # 2 chai + 2 boltz (normal case), only some poses relaxed.
    structs = {
        "E_0_chai": "chai", "E_1_chai": "chai",
        "E_0_boltz": "boltz", "E_1_boltz": "boltz",
    }
    rmsd = {
        frozenset({"E_0_chai", "E_1_chai"}): 0.5,
        frozenset({"E_0_boltz", "E_1_boltz"}): 0.5,
        frozenset({"E_0_chai", "E_0_boltz"}): 1.0,
        frozenset({"E_0_chai", "E_1_boltz"}): 2.0,
        frozenset({"E_1_chai", "E_0_boltz"}): 2.0,
        frozenset({"E_1_chai", "E_1_boltz"}): 2.0,
    }
    df = _pairwise("E", structs, rmsd)
    # only 2 of 4 poses have energy
    scores = {"E": {"chai": {"E_0": -30.0}, "boltz": {"E_0": -10.0}, "vina": {}}}
    out = select_best_docked_structures(df, scores)
    fused = out[out["method"] == "fused_rank"]
    assert len(fused) == 1  # never crashes, always emits


def test_fused_rank_vina_direction():
    # chai/boltz/vina in one entry; vina energy lower = better, same as others.
    structs = {"E_0_chai": "chai", "E_0_boltz": "boltz",
               "E_0_vina": "vina", "E_1_vina": "vina"}
    rmsd = {
        frozenset({"E_0_chai", "E_0_boltz"}): 1.0,
        frozenset({"E_0_chai", "E_0_vina"}): 1.0,
        frozenset({"E_0_chai", "E_1_vina"}): 1.0,
        frozenset({"E_0_boltz", "E_0_vina"}): 1.0,
        frozenset({"E_0_boltz", "E_1_vina"}): 1.0,
        frozenset({"E_0_vina", "E_1_vina"}): 0.5,
    }
    df = _pairwise("E", structs, rmsd)
    # vina keyed by int pose index; E_1_vina has the best (lowest) energy
    scores = {"E": {"chai": {"E_0": -5.0}, "boltz": {"E_0": -5.0},
                    "vina": {0: -6.0, 1: -50.0}}}
    out = select_best_docked_structures(df, scores)
    fused = out[out["method"] == "fused_rank"].iloc[0]
    assert fused["best_structure"] == "E_1_vina"


def test_existing_methods_schema_unchanged():
    structs = {"E_0_chai": "chai", "E_1_chai": "chai", "E_0_boltz": "boltz"}
    rmsd = {
        frozenset({"E_0_chai", "E_1_chai"}): 0.5,
        frozenset({"E_0_chai", "E_0_boltz"}): 1.0,
        frozenset({"E_1_chai", "E_0_boltz"}): 3.0,
    }
    df = _pairwise("E", structs, rmsd)
    out = select_best_docked_structures(df, None)
    assert set(out.columns) == {
        "Entry", "tool", "best_structure", "avg_ligandRMSD", "method"
    }
    assert {"inter_tool_weighted_avg", "inter_tool_min_per_tool"}.issubset(
        set(out["method"])
    )


def test_fused_rank_single_tool_entry_behavior():
    structs = {"E_0_chai": "chai", "E_1_chai": "chai"}
    rmsd = {frozenset({"E_0_chai", "E_1_chai"}): 0.5}
    df = _pairwise("E", structs, rmsd)

    out = select_best_docked_structures(df, None)
    assert out.empty is True

    scores = {
        "E": {
            "chai": {"E_0": -5.0, "E_1": -9.0},
            "boltz": {},
            "vina": {},
        }
    }
    out = select_best_docked_structures(df, scores)
    fused = out[out["method"] == "fused_rank"]
    assert len(fused) == 1
    assert fused.iloc[0]["best_structure"] == "E_1_chai"


def test_fastrelax_scores_by_entry_extraction():
    from structurezyme.steps.computeligandRMSD_step import (
        _fastrelax_scores_by_entry,
    )
    df = pd.DataFrame({
        "Entry": ["A", "A", "B"],
        "fastrelax_score": [
            {"chai": {"A_0": -1.0}, "boltz": {}, "vina": {}},
            {"chai": {"A_0": -1.0}, "boltz": {}, "vina": {}},
            {"chai": {"B_0": -2.0}, "boltz": {}, "vina": {}},
        ],
    })
    out = _fastrelax_scores_by_entry(df)
    assert out["A"]["chai"]["A_0"] == -1.0
    assert out["B"]["chai"]["B_0"] == -2.0


def test_fastrelax_scores_by_entry_missing_column():
    from structurezyme.steps.computeligandRMSD_step import (
        _fastrelax_scores_by_entry,
    )
    df = pd.DataFrame({"Entry": ["A"]})
    assert _fastrelax_scores_by_entry(df) == {}
