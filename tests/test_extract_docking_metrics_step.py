"""Tests for `structurezyme.steps.extract_docking_metrics_step`.

Covers the pure-logic Vina log path helper (`_vina_log_path`), the
pre-existing but previously untested `parse_vina_output`, and the
`DockingMetrics` wiring that surfaces parsed Vina affinities as a
`vina_affinities` column on `DockingMetrics.execute()`'s output.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from structurezyme.steps.extract_docking_metrics_step import (
    DockingMetrics,
    _vina_log_path,
    parse_vina_output,
)


_VINA_LOG_TABLE = """mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1       -6.5          0          0
   2       -6.1        1.2        2.4
"""


def test_vina_log_path_builds_expected_pattern():
    """`vina_dir` on the DataFrame is the receptor PDB *file path*
    (`<label_dir>/<Entry>.pdb`), not the containing directory — see
    `Vina._execute` which appends `str(pdb_path)` to `output_filenames`,
    and `PrepareVina` which relies on `vina_path.parent` /
    `vina_path.stem`. The log file lives next to that receptor PDB, so
    `_vina_log_path` must resolve to the receptor's parent directory."""
    receptor_pdb = "/some/label_dir/EntryA.pdb"
    result = _vina_log_path(receptor_pdb, "EntryA", "substrateB")
    assert result == Path("/some/label_dir") / "EntryA-substrateB_log.txt"


def test_vina_log_path_regression_matches_production_layout(tmp_path):
    """Regression: on the fmo-fad-01 SLURM run, `vina_dir` held
    `/.../docking/vina/<Entry>/<Entry>.pdb` and the resolver treated it
    as a directory, producing a nonexistent
    `/.../<Entry>.pdb/<Entry>-<lig>_log.txt`, silently yielding
    `vina_affinities = {}` on every row. Recreate that exact on-disk
    layout and assert the resolver finds the real log file."""
    label_dir = tmp_path / "docking" / "vina" / "F5SYD3"
    label_dir.mkdir(parents=True)
    receptor_pdb = label_dir / "F5SYD3.pdb"
    receptor_pdb.write_text("HEADER placeholder\n")
    real_log = label_dir / "F5SYD3-indole_log.txt"
    real_log.write_text(_VINA_LOG_TABLE)

    resolved = _vina_log_path(str(receptor_pdb), "F5SYD3", "indole")

    assert resolved == real_log
    assert resolved.exists(), (
        f"Resolver produced {resolved!r}, which does not exist. This is "
        "the pre-fix failure mode that silently caused vina_affinities={} "
        "on the fmo-fad-01 run."
    )


def test_parse_vina_output_parses_mode_affinity_table(tmp_path):
    log_file = tmp_path / "log.txt"
    log_file.write_text(_VINA_LOG_TABLE)

    result = parse_vina_output(log_file)

    assert result == {1: -6.5, 2: -6.1}


def test_docking_metrics_execute_populates_vina_affinities(tmp_path):
    """`vina_dir` mirrors production: it is the receptor PDB file path
    written by `Vina._execute`, not the containing directory."""
    label_dir = tmp_path / "label_dir"
    label_dir.mkdir()
    receptor_pdb = label_dir / "EntryA.pdb"
    receptor_pdb.write_text("HEADER placeholder\n")
    log_file = label_dir / "EntryA-substrateB_log.txt"
    log_file.write_text(_VINA_LOG_TABLE)

    chai_dir = tmp_path / "chai_root"
    (chai_dir / "chai").mkdir(parents=True)
    boltz_dir = tmp_path / "boltz_root"
    boltz_dir.mkdir()

    df = pd.DataFrame({
        "Entry": ["EntryA"],
        "substrate_name": ["substrateB"],
        "vina_dir": [str(receptor_pdb)],
        "chai_dir": [str(chai_dir)],
        "boltz_dir": [str(boltz_dir)],
    })

    step = DockingMetrics(input_dir=tmp_path, output_dir=tmp_path)
    output_df = step.execute(df)

    assert output_df.loc[0, "vina_affinities"] == {1: -6.5, 2: -6.1}


def test_docking_metrics_execute_vina_affinities_empty_when_not_run(tmp_path):
    """`vina_dir` is absent from rows when `run_vina=False`; DockingMetrics
    must not crash and must still produce the `vina_affinities` column."""
    chai_dir = tmp_path / "chai_root"
    (chai_dir / "chai").mkdir(parents=True)
    boltz_dir = tmp_path / "boltz_root"
    boltz_dir.mkdir()

    df = pd.DataFrame({
        "Entry": ["EntryA"],
        "substrate_name": ["substrateB"],
        "chai_dir": [str(chai_dir)],
        "boltz_dir": [str(boltz_dir)],
    })

    step = DockingMetrics(input_dir=tmp_path, output_dir=tmp_path)
    output_df = step.execute(df)

    assert output_df.loc[0, "vina_affinities"] == {}
