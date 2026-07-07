"""Tests for filterzyme.steps.PLACER_step.

Covers the module-level `_count_ligands` helper (ported from the deleted
`PLACER_forChai_step._count_ligands` method) and the `PLACER` step
constructor's `placer_script_path` validation.

No pyrosetta, no subprocess, no actual PLACER invocation — those are
covered in later tasks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from filterzyme.steps.PLACER_step import PLACER, _count_ligands, _select_one_per_entry


def _write_pdb(path, lines):
    """Write PDB-formatted lines (each already newline-terminated or not) to `path`."""
    with open(path, "w") as f:
        for line in lines:
            if not line.endswith("\n"):
                line = line + "\n"
            f.write(line)


# Column layout of a PDB ATOM/HETATM record (1-indexed columns from the
# PDB spec; 0-indexed slice ranges shown for clarity):
#   cols 1-6    (0:6)    record name  ("HETATM" or "ATOM  ")
#   cols 7-11   (6:11)   atom serial number
#   cols 13-16  (12:16)  atom name
#   col  17     (16)     alt loc
#   cols 18-20  (17:20)  residue name       <- _count_ligands reads line[17:20]
#   col  22     (21)     chain identifier   <- _count_ligands reads line[21]
#   cols 23-26  (22:26)  residue sequence   <- _count_ligands reads line[22:26]
#   cols 31-54  (30:54)  x, y, z coords


def test_count_ligands_single(tmp_path):
    pdb = tmp_path / "single.pdb"
    # One LIG instance: chain A, resseq 300, two atoms.
    _write_pdb(
        pdb,
        [
            "HETATM    1  C1  LIG A 300      10.000  10.000  10.000  1.00  0.00           C",
            "HETATM    2  C2  LIG A 300      11.000  10.000  10.000  1.00  0.00           C",
        ],
    )
    assert _count_ligands(pdb, "LIG") == 1


def test_count_ligands_multi(tmp_path):
    pdb = tmp_path / "multi.pdb"
    # Two distinct LIG instances: (A, 300) and (B, 400). Plus ALA ATOM records
    # that must NOT be counted.
    _write_pdb(
        pdb,
        [
            "ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00  0.00           N",
            "ATOM      2  CA  ALA A   1      1.000   0.000   0.000  1.00  0.00           C",
            "HETATM    3  C1  LIG A 300      10.000  10.000  10.000  1.00  0.00           C",
            "HETATM    4  C2  LIG A 300      11.000  10.000  10.000  1.00  0.00           C",
            "HETATM    5  C1  LIG B 400      20.000  20.000  20.000  1.00  0.00           C",
            "HETATM    6  C2  LIG B 400      21.000  20.000  20.000  1.00  0.00           C",
        ],
    )
    assert _count_ligands(pdb, "LIG") == 2


def test_count_ligands_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.pdb"
    assert _count_ligands(missing, "LIG") == 0


def test_placer_init_raises_on_missing_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="PLACER script not found"):
        PLACER(
            preparedfiles_dir=tmp_path,
            output_dir=tmp_path,
            predict_ligand="LIG",
            placer_script_path="/nonexistent/run_PLACER.py",
        )


def test_placer_init_succeeds_with_existing_script(tmp_path):
    fake_script = tmp_path / "fake_run_PLACER.py"
    fake_script.touch()
    step = PLACER(
        preparedfiles_dir=tmp_path,
        output_dir=tmp_path / "placer_out",
        predict_ligand="LIG",
        placer_script_path=str(fake_script),
    )
    assert step.placer_script_path == fake_script
    assert step.predict_ligand == "LIG"
    assert step.nsamples == 50  # default
    assert step.rerank == "prmsd"  # default
    assert (tmp_path / "placer_out").is_dir()  # output_dir was created


# ---------------------------------------------------------------------------
# Task 6: _select_one_per_entry (row-reduction for PLACER)
# ---------------------------------------------------------------------------


def test_select_one_per_entry_prefers_is_best():
    df = pd.DataFrame({
        "Entry": ["Q1", "Q1", "Q1", "Q2", "Q2", "Q2"],
        "docked_structure": ["Q1_0_chai", "Q1_1_chai", "Q1_2_chai", "Q2_0_chai", "Q2_1_chai", "Q2_2_chai"],
        "is_best": [False, True, False, False, False, True],
        "best_method": [np.nan, "inter_tool_min_per_tool", np.nan, np.nan, np.nan, "inter_tool_min_per_tool"],
    })
    result = _select_one_per_entry(df, entry_col="Entry")
    assert len(result) == 2
    # Q1's picked row is "Q1_1_chai" (the is_best=True row)
    assert set(result["docked_structure"]) == {"Q1_1_chai", "Q2_2_chai"}


def test_select_one_per_entry_method_count_tiebreak():
    df = pd.DataFrame({
        "Entry": ["Q1", "Q1"],
        "docked_structure": ["Q1_a_chai", "Q1_b_chai"],
        "is_best": [True, True],
        "best_method": ["inter_tool_min_per_tool", "inter_tool_min_per_tool,inter_tool_weighted_avg"],
    })
    result = _select_one_per_entry(df, entry_col="Entry")
    assert len(result) == 1
    assert result.iloc[0]["docked_structure"] == "Q1_b_chai"


def test_select_one_per_entry_alphabetical_final_tiebreak():
    df = pd.DataFrame({
        "Entry": ["Q1", "Q1"],
        "docked_structure": ["Q1_1_chai", "Q1_0_chai"],
        "is_best": [True, True],
        "best_method": ["inter_tool_min_per_tool", "inter_tool_weighted_avg"],
    })
    result = _select_one_per_entry(df, entry_col="Entry")
    assert len(result) == 1
    assert result.iloc[0]["docked_structure"] == "Q1_0_chai"


def test_select_one_per_entry_fallback_when_no_is_best():
    df = pd.DataFrame({
        "Entry": ["Q1", "Q1", "Q1"],
        "docked_structure": ["Q1_2_chai", "Q1_0_chai", "Q1_1_chai"],
        "is_best": [False, False, False],
        "best_method": [np.nan, np.nan, np.nan],
    })
    result = _select_one_per_entry(df, entry_col="Entry")
    assert len(result) == 1
    # Falls back to full pool; all have method_count=0 (NaN), tie-break is alphabetical
    assert result.iloc[0]["docked_structure"] == "Q1_0_chai"
