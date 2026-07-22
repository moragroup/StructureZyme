"""Tests for `structurezyme.steps.fastrelax_step`.

Covers the ranking/selection logic of the `FastRelax` step (dict-key
extraction from file paths, per-engine top-K pose ranking, column
validation, per-row rank-and-select, file-path list rewriting), plus
the wiring of `execute()`. A single pyrosetta-gated smoke test also
exercises `_relax_one` end-to-end against a real PDB fixture; it skips
cleanly when `pyrosetta` isn't importable.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from structurezyme.steps.fastrelax_step import (
    FastRelax,
    _apply_relaxed_paths,
    _confidence_key_from_path,
    _pyrosetta_available,
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
    fr = FastRelax(output_dir="/tmp/fastrelax_out")
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
    fr = FastRelax(output_dir="/tmp/fastrelax_out", top_k=1)

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


# --- FastRelax.execute (missing column) --------------------------------------


def test_execute_missing_column_raises(tmp_path):
    """`execute()` must raise ValueError when a required column is absent,
    reusing `_validate_input`. This exercises only the top of `execute()`
    and needs no pyrosetta.
    """
    fr = FastRelax(output_dir=str(tmp_path))
    df = pd.DataFrame(
        {
            "Entry": ["a"],
            # chai_files_for_superimposition intentionally missing
            "boltz_files_for_superimposition": [["/x/a_model_0_boltz.pdb"]],
            "vina_files_for_superimposition": [["/x/a_1_vina.pdb"]],
        }
    )
    with pytest.raises(ValueError, match="missing required columns"):
        fr.execute(df)


# --- FastRelax.execute (per-pose failure tolerance) --------------------------

# NOTE: The previous HEM-based `_relax_one` smoke test (removed 2026-07)
# assumed the OLD API where `_relax_one` accepted a single ligand
# resname and Rosetta's built-in HEM params handled the chemistry.
# The new API is DataFrame-driven -- see
# `test_fastrelax_preserves_ligand_atoms.py` for the end-to-end
# pyrosetta-gated regression test that covers `_relax_one` with real
# 2-ligand (indole + FAD) input.


def test_execute_relax_failure_keeps_original_path(tmp_path, monkeypatch):
    """When `_relax_one` raises for a pose, `execute()` must log and
    continue -- never abort the run -- and leave the pose's original
    path in the files-column (see spec Behavior step 6).
    """
    fr = FastRelax(
        output_dir=str(tmp_path),
        top_k=1,
        drop_unrelaxed=False,  # so unselected paths are preserved too
    )

    # Stub `_register_ligands` -- the real one does RDKit + subprocess
    # work per unique SMILES, which is unnecessary here (we never let
    # `_relax_one` actually run). This test targets the per-pose
    # failure-tolerance contract of `execute()`, not params generation.
    monkeypatch.setattr(FastRelax, "_register_ligands", lambda self, df: None)

    # Track that `_relax_one` was actually invoked (and always failed),
    # so we know we exercised `execute()`'s per-pose try/except and not
    # just some upstream filter.
    calls: list[str] = []

    def _always_fail(self, pdb_path, substrate_smiles=None, cofactor_smiles=None):
        calls.append(str(pdb_path))
        raise RuntimeError("simulated relax failure")

    monkeypatch.setattr(FastRelax, "_relax_one", _always_fail)

    # Use real (touched) files -- `_rank_and_select` guards via
    # `valid_file_list`, which requires paths to actually exist on disk.
    chai_files = [str(tmp_path / "E_0_chai.pdb"), str(tmp_path / "E_1_chai.pdb")]
    boltz_files = [str(tmp_path / "E_model_0_boltz.pdb")]
    vina_files = [str(tmp_path / "E_1_vina.pdb")]
    for f in chai_files + boltz_files + vina_files:
        Path(f).touch()

    df = pd.DataFrame(
        {
            "Entry": ["E"],
            "chai_files_for_superimposition": [chai_files],
            "boltz_files_for_superimposition": [boltz_files],
            "vina_files_for_superimposition": [vina_files],
            "chai_ptm": [{"E_0": 0.5, "E_1": 0.9}],
            "boltz2_confidence_score": [{"E_model_0": 0.8}],
            "vina_affinities": [{1: -4.0}],
        }
    )

    # Must not raise -- per-pose failures are logged and swallowed.
    out = fr.execute(df)

    # Sanity: _relax_one was actually invoked (top-1 per engine = 3 calls).
    assert len(calls) == 3

    # Original paths preserved (relaxation failed for the selected pose;
    # unselected preserved because drop_unrelaxed=False).
    assert set(out.iloc[0]["chai_files_for_superimposition"]) == set(chai_files)
    assert set(out.iloc[0]["boltz_files_for_superimposition"]) == set(boltz_files)
    assert set(out.iloc[0]["vina_files_for_superimposition"]) == set(vina_files)

    # fastrelax_score column exists but is empty for all engines (all
    # relax calls failed).
    score = out.iloc[0]["fastrelax_score"]
    assert score == {"chai": {}, "boltz": {}, "vina": {}}
