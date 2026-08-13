"""Unit tests for PLACER's ``auto`` ligand-resname resolution.

These need no PLACER binary: they exercise the pure PDB-parsing resolver and
the execute() ``auto`` wiring with subprocess.run mocked.
"""
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from structurezyme.steps import PLACER_step
from structurezyme.steps.PLACER_step import PLACER, _resolve_predict_ligand


def _write_pdb(path: Path, resnames_by_line):
    """Write a minimal PDB with one HETATM per (resname, chain, resseq) tuple."""
    lines = []
    serial = 1
    for resname, chain, resseq in resnames_by_line:
        # PDB columns: HETATM(1-6) serial(7-11) name(13-16) resName(18-20)
        # chainID(22) resSeq(23-26)
        lines.append(
            f"HETATM{serial:>5}  C1  {resname:>3} {chain}{resseq:>4}"
            f"      0.000   0.000   0.000  1.00  0.00           C\n"
        )
        serial += 1
    path.write_text("".join(lines))


def _write_protein_only_pdb(path: Path):
    """A PDB with only standard ATOM residues (no ligand HETATM)."""
    path.write_text(
        "ATOM      1  CA  ALA A   1"
        "      0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  GLY A   2"
        "      1.000   0.000   0.000  1.00  0.00           C\n"
    )


def test_resolve_single_x01(tmp_path):
    """A fastrelax-renamed substrate (X01) is selected."""
    pdb = tmp_path / "a.pdb"
    _write_pdb(pdb, [("X01", "C", 1)])
    assert _resolve_predict_ligand(pdb) == "X01"


def test_resolve_single_lig(tmp_path):
    """A pre-fastrelax substrate (LIG) is selected."""
    pdb = tmp_path / "a.pdb"
    _write_pdb(pdb, [("LIG", "C", 1)])
    assert _resolve_predict_ligand(pdb) == "LIG"


def test_resolve_substrate_over_cofactor(tmp_path):
    """With substrate X01 + cofactor Z01, the substrate X01 is predicted."""
    pdb = tmp_path / "a.pdb"
    _write_pdb(pdb, [("X01", "C", 1), ("Z01", "D", 2)])
    assert _resolve_predict_ligand(pdb) == "X01"


def test_resolve_lig_over_cofactor(tmp_path):
    """With substrate LIG + cofactor Z01, LIG is predicted."""
    pdb = tmp_path / "a.pdb"
    _write_pdb(pdb, [("LIG", "C", 1), ("Z01", "D", 2)])
    assert _resolve_predict_ligand(pdb) == "LIG"


def test_resolve_ambiguous_raises(tmp_path):
    """Two non-substrate, non-cofactor ligands -> raise, do not guess."""
    pdb = tmp_path / "a.pdb"
    _write_pdb(pdb, [("ABC", "C", 1), ("DEF", "D", 2)])
    with pytest.raises(ValueError, match="could not auto-resolve"):
        _resolve_predict_ligand(pdb)


def test_resolve_no_ligand_raises(tmp_path):
    """Protein-only PDB (no HETATM ligand) -> raise."""
    pdb = tmp_path / "a.pdb"
    _write_protein_only_pdb(pdb)
    with pytest.raises(ValueError, match="could not auto-resolve"):
        _resolve_predict_ligand(pdb)


def test_execute_auto_uses_resolved_resname(tmp_path, monkeypatch):
    """execute() with predict_ligand='auto' resolves X01 from the PDB and passes
    it to the PLACER subprocess command."""
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    out = tmp_path / "out"
    # Prepared PDB whose ligand is X01 (fastrelax-renamed).
    pdb = prepared / "P41365_0_chai_relaxed.pdb"
    _write_pdb(pdb, [("X01", "C", 1)])

    # Fake PLACER install so __init__ passes.
    fake_script = tmp_path / "run_PLACER.py"
    fake_script.touch()
    fake_env = tmp_path / "placerenv"
    (fake_env / "bin").mkdir(parents=True)
    (fake_env / "bin" / "python").touch()

    placer = PLACER(
        preparedfiles_dir=prepared,
        output_dir=out,
        predict_ligand="auto",
        placer_script_path=str(fake_script),
        placer_env_path=str(fake_env),
    )

    captured = {}

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        # Write a CSV named <stem>*.csv so the parser finds it.
        stem = Path(cmd[cmd.index("--ifile") + 1]).stem
        csv = out / f"{stem}.csv"
        csv.write_text("fape,lddt,rmsd,kabsch,prmsd,plddt,plddt_pde\n0,0,0,0,0,0,0\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    df = pd.DataFrame(
        {
            "Entry": ["P41365"],
            "docked_structure": ["P41365_0_chai_relaxed"],
            "is_best": [True],
            "best_method": ["fused_rank"],
        }
    )
    result = placer.execute(df)

    # The resolved resname X01 must be the --predict_ligand value.
    cmd = captured["cmd"]
    assert "--predict_ligand" in cmd
    assert cmd[cmd.index("--predict_ligand") + 1] == "X01"
    assert "LIG" not in cmd
    # And scores were merged (not the None failure row).
    assert result["placer_fape"].iloc[0] == 0.0
