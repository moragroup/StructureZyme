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

from filterzyme.steps.PLACER_step import (
    PLACER,
    _count_ligands,
    _parse_placer_csv,
    _placer_available,
    _select_one_per_entry,
)


def _write_pdb(path, lines):
    """Write PDB-formatted lines (each already newline-terminated or not) to `path`."""
    with open(path, "w") as f:
        for line in lines:
            if not line.endswith("\n"):
                line = line + "\n"
            f.write(line)


def _make_fake_env(tmp_path):
    """Create a fake PLACER env layout (env/bin/python) under tmp_path.

    Returns the env root path. Use with ``placer_env_path=str(env_root)``.
    """
    from pathlib import Path as _P
    env_root = _P(tmp_path) / "fake_env"
    (env_root / "bin").mkdir(parents=True, exist_ok=True)
    (env_root / "bin" / "python").touch()
    return env_root


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
    env_root = _make_fake_env(tmp_path)
    with pytest.raises(FileNotFoundError, match="PLACER script not found"):
        PLACER(
            preparedfiles_dir=tmp_path,
            output_dir=tmp_path,
            predict_ligand="LIG",
            placer_script_path="/nonexistent/run_PLACER.py",
            placer_env_path=str(env_root),
        )


def test_placer_init_raises_on_missing_env(tmp_path):
    fake_script = tmp_path / "fake_run_PLACER.py"
    fake_script.touch()
    with pytest.raises(FileNotFoundError, match="PLACER env python not found"):
        PLACER(
            preparedfiles_dir=tmp_path,
            output_dir=tmp_path,
            predict_ligand="LIG",
            placer_script_path=str(fake_script),
            placer_env_path="/nonexistent/env",
        )


def test_placer_init_succeeds_with_existing_script(tmp_path):
    fake_script = tmp_path / "fake_run_PLACER.py"
    fake_script.touch()
    env_root = _make_fake_env(tmp_path)
    step = PLACER(
        preparedfiles_dir=tmp_path,
        output_dir=tmp_path / "placer_out",
        predict_ligand="LIG",
        placer_script_path=str(fake_script),
        placer_env_path=str(env_root),
    )
    assert step.placer_script_path == fake_script
    assert step.placer_env_python == env_root / "bin" / "python"
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


# ---------------------------------------------------------------------------
# Task 7.1: _build_pdb_path
# ---------------------------------------------------------------------------

from pathlib import Path


def test_build_pdb_path(tmp_path):
    fake_script = tmp_path / "fake_run_PLACER.py"
    fake_script.touch()
    env_root = _make_fake_env(tmp_path)
    step = PLACER(
        preparedfiles_dir="/x/preparedfiles",
        output_dir=tmp_path,
        predict_ligand="LIG",
        placer_script_path=str(fake_script),
        placer_env_path=str(env_root),
    )
    row = pd.Series({"docked_structure": "Q97WW0_1_vina"})
    assert step._build_pdb_path(row) == Path("/x/preparedfiles/Q97WW0_1_vina.pdb")


# ---------------------------------------------------------------------------
# Task 7.3: _build_cmd
# ---------------------------------------------------------------------------


def test_build_placer_cmd_single_ligand(tmp_path):
    fake_script = tmp_path / "fake_run_PLACER.py"
    fake_script.touch()
    env_root = _make_fake_env(tmp_path)
    step = PLACER(
        preparedfiles_dir=tmp_path,
        output_dir=tmp_path / "out",
        predict_ligand="LIG",
        placer_script_path=str(fake_script),
        placer_env_path=str(env_root),
        nsamples=50,
        rerank="prmsd",
    )
    cmd = step._build_cmd(Path("/x/foo.pdb"), n_ligands=1)
    assert cmd == [
        str(env_root / "bin" / "python"),
        str(fake_script),
        "--ifile", "/x/foo.pdb",
        "--odir", str(tmp_path / "out"),
        "--rerank", "prmsd",
        "-n", "50",
        "--predict_ligand", "LIG",
    ]


def test_build_placer_cmd_multi_ligand(tmp_path):
    fake_script = tmp_path / "fake_run_PLACER.py"
    fake_script.touch()
    env_root = _make_fake_env(tmp_path)
    step = PLACER(
        preparedfiles_dir=tmp_path,
        output_dir=tmp_path / "out",
        predict_ligand="LIG",
        placer_script_path=str(fake_script),
        placer_env_path=str(env_root),
    )
    cmd = step._build_cmd(Path("/x/foo.pdb"), n_ligands=3)
    # multi-ligand adds --predict_multi at the end
    assert cmd[-1] == "--predict_multi"
    # and everything before is the single-ligand form
    assert cmd[:-1] == step._build_cmd(Path("/x/foo.pdb"), n_ligands=1)


# ---------------------------------------------------------------------------
# Task 7.5: _parse_placer_csv
# ---------------------------------------------------------------------------


def test_parse_placer_csv_success(tmp_path):
    csv_path = tmp_path / "test.csv"
    csv_path.write_text(
        "label,model_idx,fape,lddt,rmsd,kabsch,prmsd,plddt,plddt_pde\n"
        "s,1,1.54,0.956,0.365,0.196,0.681,0.975,0.927\n"
        "s,2,1.87,0.941,0.539,0.161,1.621,0.952,0.801\n"
        "s,3,2.16,0.904,4.469,0.428,3.753,0.564,0.685\n"
    )
    result = _parse_placer_csv(csv_path)
    assert result == {
        "placer_fape":      1.54,
        "placer_lddt":      0.956,
        "placer_rmsd":      0.365,
        "placer_kabsch":    0.196,
        "placer_prmsd":     0.681,
        "placer_plddt":     0.975,
        "placer_plddt_pde": 0.927,
    }


def test_parse_placer_csv_missing_file(tmp_path):
    result = _parse_placer_csv(tmp_path / "does_not_exist.csv")
    assert result == {
        "placer_fape": None, "placer_lddt": None, "placer_rmsd": None,
        "placer_kabsch": None, "placer_prmsd": None, "placer_plddt": None,
        "placer_plddt_pde": None,
    }


def test_parse_placer_csv_malformed(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("this,is,not,valid,placer,output\nfoo,bar,baz,qux,quux,corge\n")
    result = _parse_placer_csv(csv_path)
    # Missing columns → all None
    assert all(v is None for v in result.values())


# ---------------------------------------------------------------------------
# Task 7.7: opt-in end-to-end smoke test (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _placer_available(),
    reason="Real PLACER run requires PLACER_SMOKE=1 + GPU + installed PLACER",
)
def test_execute_smoke(tmp_path):
    """End-to-end PLACER invocation via subprocess. Opt-in only (PLACER_SMOKE=1).

    Uses ``dnHEM1.pdb`` from the PLACER examples (HEM ligand in chain B, resseq 213),
    which is PLACER's own reference command-line example (see
    ``examples/commandline_examples.sh`` line 13). The ligand chain B is
    distinct from the protein chain A, avoiding the "ligand chains already exist
    in parsed protein chains" collision. Override the input file / ligand via
    ``PLACER_SMOKE_PDB`` and ``PLACER_SMOKE_LIGAND`` env vars if needed.
    """
    import os
    import shutil as _sh

    prep = tmp_path / "preparedfiles"
    prep.mkdir()
    pdb_name = os.environ.get(
        "PLACER_SMOKE_PDB",
        "/mnt/labs/data/mora/software/PLACER/examples/inputs/dnHEM1.pdb",
    )
    src = Path(pdb_name)
    dest = prep / "smoke_0_chai.pdb"
    _sh.copyfile(src, dest)

    ligand = os.environ.get("PLACER_SMOKE_LIGAND", "B-HEM-213")

    df = pd.DataFrame({
        "Entry": ["smoke"],
        "docked_structure": ["smoke_0_chai"],
        "is_best": [True],
        "best_method": ["inter_tool_min_per_tool"],
    })

    step = PLACER(
        preparedfiles_dir=prep,
        output_dir=tmp_path / "placer_out",
        predict_ligand=ligand,
        nsamples=3,  # keep fast
        rerank="prmsd",
        # placer_script_path defaults to /mnt/labs/.../PLACER/run_PLACER.py
        # placer_env_path defaults to   /mnt/labs/.../PLACER/env
    )
    result_df = step.execute(df)
    assert len(result_df) == 1
    row = result_df.iloc[0]
    assert row["Entry"] == "smoke"
    # At minimum, prmsd should be set (PLACER's main output metric)
    assert row["placer_prmsd"] is not None and row["placer_prmsd"] > 0
    assert row["placer_dir"] == str(tmp_path / "placer_out")


# ---------------------------------------------------------------------------
# Task 7.8: _validate_input + execute
# ---------------------------------------------------------------------------


def test_execute_missing_columns_raises(tmp_path):
    fake_script = tmp_path / "fake_run_PLACER.py"
    fake_script.touch()
    env_root = _make_fake_env(tmp_path)
    step = PLACER(
        preparedfiles_dir=tmp_path,
        output_dir=tmp_path / "out",
        predict_ligand="LIG",
        placer_script_path=str(fake_script),
        placer_env_path=str(env_root),
    )
    # DataFrame missing `is_best` and `best_method`
    df = pd.DataFrame({"Entry": ["Q1"], "docked_structure": ["Q1_0_chai"]})
    with pytest.raises(ValueError, match="missing required columns"):
        step.execute(df)


# ---------------------------------------------------------------------------
# Task 7.9: mocked-subprocess behavior tests for execute()
# ---------------------------------------------------------------------------


def test_execute_per_entry_success_merges_csv(tmp_path, monkeypatch):
    """execute() calls subprocess for each reduced entry and merges the CSV."""
    import subprocess

    fake_script = tmp_path / "fake_run_PLACER.py"
    fake_script.touch()
    prep = tmp_path / "preparedfiles"
    prep.mkdir()
    outdir = tmp_path / "placer_out"
    # Real PDB file for _count_ligands to read (empty is fine, function returns 0)
    (prep / "Q1_0_chai.pdb").touch()

    env_root = _make_fake_env(tmp_path)
    step = PLACER(
        preparedfiles_dir=prep,
        output_dir=outdir,
        predict_ligand="LIG",
        placer_script_path=str(fake_script),
        placer_env_path=str(env_root),
    )

    def fake_run(cmd, capture_output, text, check):
        # Write a fake CSV where PLACER would
        csv_path = outdir / "Q1_0_chai.csv"
        csv_path.write_text(
            "label,model_idx,fape,lddt,rmsd,kabsch,prmsd,plddt,plddt_pde\n"
            "Q1_0_chai,1,1.5,0.95,0.4,0.2,0.7,0.98,0.93\n"
        )
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    df = pd.DataFrame({
        "Entry": ["Q1"],
        "docked_structure": ["Q1_0_chai"],
        "is_best": [True],
        "best_method": ["inter_tool_min_per_tool"],
    })
    result = step.execute(df)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["placer_prmsd"] == 0.7
    assert row["placer_plddt"] == 0.98
    assert row["placer_dir"] == str(outdir)


def test_execute_subprocess_failure_sets_none(tmp_path, monkeypatch):
    """A failed subprocess → all 8 placer_* columns None; no exception raised."""
    import subprocess

    fake_script = tmp_path / "fake_run_PLACER.py"
    fake_script.touch()
    prep = tmp_path / "preparedfiles"
    prep.mkdir()
    (prep / "Q1_0_chai.pdb").touch()

    env_root = _make_fake_env(tmp_path)
    step = PLACER(
        preparedfiles_dir=prep,
        output_dir=tmp_path / "placer_out",
        predict_ligand="LIG",
        placer_script_path=str(fake_script),
        placer_env_path=str(env_root),
    )

    def fake_run_fail(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="some error")

    monkeypatch.setattr(subprocess, "run", fake_run_fail)

    df = pd.DataFrame({
        "Entry": ["Q1"],
        "docked_structure": ["Q1_0_chai"],
        "is_best": [True],
        "best_method": ["inter_tool_min_per_tool"],
    })
    result = step.execute(df)  # must not raise
    row = result.iloc[0]
    assert row["placer_prmsd"] is None
    assert row["placer_plddt"] is None
    assert row["placer_dir"] is None


def test_execute_missing_pdb_skips_entry(tmp_path, monkeypatch):
    """If the docked PDB doesn't exist in preparedfiles_dir, log+skip that entry."""
    import subprocess

    fake_script = tmp_path / "fake_run_PLACER.py"
    fake_script.touch()
    prep = tmp_path / "preparedfiles"
    prep.mkdir()
    # Note: NOT creating any .pdb file

    env_root = _make_fake_env(tmp_path)
    step = PLACER(
        preparedfiles_dir=prep,
        output_dir=tmp_path / "placer_out",
        predict_ligand="LIG",
        placer_script_path=str(fake_script),
        placer_env_path=str(env_root),
    )

    called = []

    def fake_run_should_not_be_called(*a, **kw):
        called.append(True)
        raise RuntimeError("subprocess.run should not have been called")

    monkeypatch.setattr(subprocess, "run", fake_run_should_not_be_called)

    df = pd.DataFrame({
        "Entry": ["Q1"],
        "docked_structure": ["MISSING_0_chai"],
        "is_best": [True],
        "best_method": ["inter_tool_min_per_tool"],
    })
    result = step.execute(df)
    assert len(called) == 0  # subprocess not called
    row = result.iloc[0]
    assert row["placer_prmsd"] is None
    assert row["placer_dir"] is None
