"""A missing prepared PDB must fail loudly, not silently produce empty results.

Regression guard for the silent-skip anti-pattern: the five analysis steps
(fpocket, ligand_sasa, plip, geometric_filter [general + esterase], PLACER)
resolve each pose as ``preparedfiles_dir/<docked_structure>.pdb``. When that
file is absent it means an upstream pipeline step (docking / prepare_files /
fastrelax) failed to produce the structure -- a genuine pipeline break, not a
legitimate "no result" outcome. These steps used to swallow the missing file
into a row of ``None`` values and still report success, which hid a real bug.

Each step must instead raise ``FileNotFoundError`` immediately when an expected
prepared PDB is missing. Legitimate per-row analysis failures on a *present*
PDB (no ligand chain, no pocket, etc.) are intentionally NOT covered here --
those remain valid empty results.
"""
from __future__ import annotations

import pandas as pd
import pytest


def _empty_prepared_dir(tmp_path):
    """A prepared-files dir that exists but contains no PDB for our row."""
    d = tmp_path / "preparedfiles_for_superimposition"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _base_row():
    """Minimal row referencing a docked_structure whose .pdb does not exist."""
    return {
        "Entry": "E1",
        "docked_structure": "does_not_exist_pose",
        "substrate_smiles": "CCOC(C)=O",
    }


def test_fpocket_missing_pdb_raises(tmp_path):
    from structurezyme.steps.fpocket_step import Fpocket

    prepared = _empty_prepared_dir(tmp_path)
    df = pd.DataFrame([_base_row()])
    step = Fpocket(preparedfiles_dir=prepared, output_dir=tmp_path / "asv")
    with pytest.raises(FileNotFoundError):
        step.execute(df)


def test_ligand_sasa_missing_pdb_raises(tmp_path):
    from structurezyme.steps.ligandSASA_step import LigandSASA

    prepared = _empty_prepared_dir(tmp_path)
    df = pd.DataFrame([_base_row()])
    step = LigandSASA(input_dir=prepared, output_dir=tmp_path / "sasa")
    with pytest.raises(FileNotFoundError):
        step.execute(df)


def test_plip_missing_pdb_raises(tmp_path):
    from structurezyme.steps.plip_step import PLIP

    prepared = _empty_prepared_dir(tmp_path)
    df = pd.DataFrame([_base_row()])
    step = PLIP(input_dir=prepared, output_dir=tmp_path / "plip")
    with pytest.raises(FileNotFoundError):
        step.execute(df)


def test_general_geometric_filter_missing_pdb_raises(tmp_path):
    from structurezyme.steps.geometric_filtering_cofactor_MCS import (
        GeneralGeometricFiltering,
    )

    prepared = _empty_prepared_dir(tmp_path)
    row = _base_row()
    row.update(
        {
            "cofactor_smiles": None,
            "substrate_moiety": None,
            "cofactor_moiety": None,
            "tool": "chai",
        }
    )
    df = pd.DataFrame([row])
    step = GeneralGeometricFiltering(preparedfiles_dir=prepared, output_dir=tmp_path / "geo")
    with pytest.raises(FileNotFoundError):
        step.execute(df)


def test_esterase_geometric_filter_missing_pdb_raises(tmp_path):
    from structurezyme.steps.geometric_filtering_esterase import (
        EsteraseGeometricFiltering,
    )

    prepared = _empty_prepared_dir(tmp_path)
    row = _base_row()
    row.update(
        {
            "cofactor_smiles": None,
            "substrate_moiety": None,
            "cofactor_moiety": None,
            "tool": "chai",
        }
    )
    df = pd.DataFrame([row])
    step = EsteraseGeometricFiltering(preparedfiles_dir=prepared, output_dir=tmp_path / "geo")
    with pytest.raises(FileNotFoundError):
        step.execute(df)


def _fake_placer_env(tmp_path):
    """Create a fake PLACER env + script so the PLACER ctor validation passes."""
    env_root = tmp_path / "fake_env"
    (env_root / "bin").mkdir(parents=True, exist_ok=True)
    (env_root / "bin" / "python").touch()
    script = tmp_path / "run_PLACER.py"
    script.touch()
    return env_root, script


def test_placer_missing_pdb_raises(tmp_path):
    from structurezyme.steps.PLACER_step import PLACER

    prepared = _empty_prepared_dir(tmp_path)
    env_root, script = _fake_placer_env(tmp_path)
    row = _base_row()
    row.update({"is_best": True, "best_method": "chai"})
    df = pd.DataFrame([row])
    step = PLACER(
        preparedfiles_dir=prepared,
        output_dir=tmp_path / "placer_out",
        predict_ligand="A-LIG-300",
        placer_script_path=str(script),
        placer_env_path=str(env_root),
    )
    with pytest.raises(FileNotFoundError):
        step.execute(df)
