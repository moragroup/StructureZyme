"""Tests for `filterzyme.steps.fastrelax_step`.

Covers only the PyRosetta-independent logic of the `FastRelax` step:
dict-key extraction from file paths, per-engine top-K pose ranking
(descending for chai/boltz confidence scores, ascending for vina
affinities), column validation, per-row rank-and-select, and rewriting
file-path lists after relaxation. The actual PyRosetta `.apply()` call
is implemented in a later task and is not exercised here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from filterzyme.steps.fastrelax_step import (
    FastRelax,
    _apply_relaxed_paths,
    _confidence_key_from_path,
    _select_top_k,
)


# --- _confidence_key_from_path (always run) --------------------------------


def test_chai_key_from_path():
    assert _confidence_key_from_path("/x/Q97WW0_0_chai.pdb", "chai") == "Q97WW0_0"


def test_boltz_key_from_path():
    assert (
        _confidence_key_from_path("/x/Q97WW0_model_0_boltz.pdb", "boltz")
        == "Q97WW0_model_0"
    )


def test_vina_key_from_path():
    assert _confidence_key_from_path("/x/Q97WW0_1_vina.pdb", "vina") == 1


# --- _select_top_k ----------------------------------------------------------


def test_select_top_k_descending():
    paths = ["/x/E_0_chai.pdb", "/x/E_1_chai.pdb", "/x/E_2_chai.pdb"]
    confidence = {"E_0": 0.5, "E_1": 0.9, "E_2": 0.7}
    result = _select_top_k(paths, confidence, "chai", top_k=2)
    assert set(result) == {"/x/E_1_chai.pdb", "/x/E_2_chai.pdb"}
    assert len(result) == 2


def test_select_top_k_ascending():
    paths = ["/x/E_1_vina.pdb", "/x/E_2_vina.pdb", "/x/E_3_vina.pdb"]
    confidence = {1: -4.0, 2: -8.0, 3: -6.0}
    result = _select_top_k(paths, confidence, "vina", top_k=2)
    assert set(result) == {"/x/E_2_vina.pdb", "/x/E_3_vina.pdb"}
    assert len(result) == 2


def test_select_top_k_missing_key_dropped():
    paths = ["/x/E_0_chai.pdb", "/x/E_1_chai.pdb"]
    confidence = {"E_0": 0.5}  # E_1 has no confidence entry
    result = _select_top_k(paths, confidence, "chai", top_k=2)
    assert "/x/E_1_chai.pdb" not in result
    assert result == ["/x/E_0_chai.pdb"]


# --- FastRelax._validate_input -----------------------------------------------


def test_validate_columns_raises_on_missing():
    fr = FastRelax(output_dir="/tmp/fastrelax_out", ligand_resname="LIG")
    df = pd.DataFrame(
        {
            "Entry": ["a"],
            "boltz_files_for_superimposition": [["/x/a_model_0_boltz.pdb"]],
            "vina_files_for_superimposition": [["/x/a_1_vina.pdb"]],
        }
    )
    with pytest.raises(ValueError, match="missing required columns"):
        fr._validate_input(df)


# --- FastRelax._rank_and_select ----------------------------------------------


def test_rank_and_select_all_engines_per_row(tmp_path):
    fr = FastRelax(output_dir="/tmp/fastrelax_out", ligand_resname="LIG", top_k=1)

    chai_files = [
        str(tmp_path / "E_0_chai.pdb"),
        str(tmp_path / "E_1_chai.pdb"),
    ]
    boltz_files = [
        str(tmp_path / "E_model_0_boltz.pdb"),
        str(tmp_path / "E_model_1_boltz.pdb"),
    ]
    vina_files = [
        str(tmp_path / "E_1_vina.pdb"),
        str(tmp_path / "E_2_vina.pdb"),
    ]
    for f in chai_files + boltz_files + vina_files:
        Path(f).touch()

    row = pd.Series(
        {
            "Entry": "E",
            "chai_files_for_superimposition": chai_files,
            "boltz_files_for_superimposition": boltz_files,
            "vina_files_for_superimposition": vina_files,
            "chai_ptm": {"E_0": 0.5, "E_1": 0.9},
            "boltz2_confidence_score": {"E_model_0": 0.8, "E_model_1": 0.3},
            "vina_affinities": {1: -4.0, 2: -8.0},
        }
    )
    result = fr._rank_and_select(row)
    assert result == {
        "chai": [chai_files[1]],
        "boltz": [boltz_files[0]],
        "vina": [vina_files[1]],
    }


# --- _apply_relaxed_paths -----------------------------------------------------


def test_drop_unrelaxed_true():
    files = ["/x/E_0_chai.pdb", "/x/E_1_chai.pdb", "/x/E_2_chai.pdb"]
    selected = ["/x/E_1_chai.pdb"]
    relaxed_map = {"/x/E_1_chai.pdb": "/out/E_1_chai_relaxed.pdb"}
    result = _apply_relaxed_paths(files, selected, relaxed_map, drop_unrelaxed=True)
    assert result == ["/out/E_1_chai_relaxed.pdb"]


def test_drop_unrelaxed_false():
    files = ["/x/E_0_chai.pdb", "/x/E_1_chai.pdb", "/x/E_2_chai.pdb"]
    selected = ["/x/E_1_chai.pdb"]
    relaxed_map = {"/x/E_1_chai.pdb": "/out/E_1_chai_relaxed.pdb"}
    result = _apply_relaxed_paths(files, selected, relaxed_map, drop_unrelaxed=False)
    assert set(result) == {
        "/x/E_0_chai.pdb",
        "/out/E_1_chai_relaxed.pdb",
        "/x/E_2_chai.pdb",
    }
    assert len(result) == 3
