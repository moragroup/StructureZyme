"""Regression test: FastRelax must preserve non-canonical ligand chemistry.

Root cause of the historical bug: `_relax_one` called
`pyrosetta.pose_from_pdb` without any `-extra_res_fa` params files, so
Rosetta silently mapped every `LIG` residue to a hardcoded 29-atom
generic template. That corrupted BOTH ligands in every prior run
(indole from 9 heavy atoms and FAD from 53 both collapsed to 29 atoms
with the same generic atom names).

The fix requires FastRelax to:
  1. Read per-row `substrate_smiles` and `cofactor_smiles` from the
     DataFrame at execute time.
  2. Generate `.params` files for each unique SMILES, assigning each a
     unique 3-letter code (S01, S02, ... for substrates; C01, C02, ...
     for cofactors).
  3. Init PyRosetta with `-extra_res_fa` pointing at those params.
  4. Rename incoming `LIG` residues to the assigned codes so Rosetta
     can tell substrate and cofactor apart.
  5. Emit a relaxed PDB where the substrate and cofactor keep their
     original heavy-atom counts.

Fixture: `F5SYD3_2_chai_min.pdb` -- a stripped FMO18 pose from the
2026-07 fmo-fad-01 run. Chain B holds the 9-heavy-atom indole
substrate, chain C holds the 53-heavy-atom FAD cofactor, and 52
protein residues within 6 A of either ligand are retained for
context.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from structurezyme.steps.fastrelax_step import FastRelax, _pyrosetta_available


FIXTURE = Path(__file__).parent / "fixtures" / "fastrelax" / "F5SYD3_2_chai_min.pdb"

INDOLE_SMILES = "C1=CC=C2C(=C1)C=CN2"
FAD_SMILES = (
    "CC1=CC2=C(C=C1C)N(C3=NC(=O)NC(=O)C3=N2)C[C@@H]([C@@H]([C@@H]"
    "(COP(=O)(O)OP(=O)(O)OC[C@@H]4[C@H]([C@H]([C@@H](O4)N5C=NC6="
    "C(N=CN=C65)N)O)O)O)O)O"
)


def _heavy_atom_count_by_resname(pdb_path: Path) -> dict[str, int]:
    """Count heavy (non-H) atoms per 3-letter residue name in a PDB.

    Element is read from cols 77-78 per the PDB spec; falls back to the
    first non-digit char of the atom name if that field is blank
    (Rosetta output usually populates it, but tolerant is better).
    """
    counts: dict[str, int] = {}
    for line in pdb_path.read_text().splitlines():
        if not (line.startswith("HETATM") or line.startswith("ATOM  ")):
            continue
        resname = line[17:20].strip()
        element = line[76:78].strip() if len(line) >= 78 else ""
        if not element:
            atom_name = line[12:16].strip()
            element = atom_name.lstrip("0123456789")[:1].upper()
        if element == "H":
            continue
        counts[resname] = counts.get(resname, 0) + 1
    return counts


@pytest.mark.skipif(
    not _pyrosetta_available(),
    reason="pyrosetta is not importable in this environment",
)
def test_fastrelax_preserves_ligand_heavy_atom_counts(tmp_path):
    """FastRelax must NOT collapse indole (9 atoms) and FAD (53 atoms)
    into the 29-atom generic LIG template.

    Uses the new per-row SMILES API: FastRelax reads
    `substrate_smiles` and `cofactor_smiles` from the DataFrame,
    generates params for each unique SMILES (assigning S01, S02, ... /
    C01, C02, ... codes), and emits a relaxed PDB with two distinct
    non-canonical residues carrying the original chemistry.
    """
    assert FIXTURE.exists(), f"regression fixture missing: {FIXTURE}"

    # Sanity: fixture has the expected ligand atom counts before we touch it.
    fixture_counts = _heavy_atom_count_by_resname(FIXTURE)
    assert fixture_counts.get("LIG") == 9 + 53, (
        f"fixture is malformed: expected 62 LIG heavy atoms (9 indole + 53 FAD), "
        f"got {fixture_counts.get('LIG')}"
    )

    # Copy the fixture into tmp_path so FastRelax can find it via a real
    # `_chai.pdb` filename that matches the pipeline's _confidence_key_from_path.
    pose_path = tmp_path / "F5SYD3_2_chai.pdb"
    pose_path.write_text(FIXTURE.read_text())

    out_dir = tmp_path / "fastrelax_out"

    fr = FastRelax(
        output_dir=str(out_dir),
        substrate_smiles_col="substrate_smiles",
        cofactor_smiles_col="cofactor_smiles",
        mode="ligand_focused",
        top_k=1,
        drop_unrelaxed=True,
        shell_radius=6.0,
    )

    df = pd.DataFrame(
        {
            "Entry": ["F5SYD3"],
            "substrate_smiles": [INDOLE_SMILES],
            "cofactor_smiles": [FAD_SMILES],
            "chai_files_for_superimposition": [[str(pose_path)]],
            "boltz_files_for_superimposition": [[]],
            "vina_files_for_superimposition": [[]],
            "chai_ptm": [{"F5SYD3_2": 0.9}],
            "boltz2_confidence_score": [{}],
            "vina_affinities": [{}],
        }
    )

    out = fr.execute(df)

    relaxed_files = out.iloc[0]["chai_files_for_superimposition"]
    assert len(relaxed_files) == 1, (
        f"expected exactly one relaxed pose, got {relaxed_files}"
    )
    relaxed_path = Path(relaxed_files[0])
    assert relaxed_path.exists(), f"relaxed PDB not written: {relaxed_path}"

    counts = _heavy_atom_count_by_resname(relaxed_path)

    # The old generic LIG residue must NOT appear in the output -- we
    # invented distinct 3-letter codes precisely so Rosetta could keep
    # substrate and cofactor separate.
    assert "LIG" not in counts, (
        f"relaxed PDB still contains generic LIG residues -- SMILES-driven "
        f"renaming did not run. Full resname counts: {counts}"
    )

    # Filter to non-canonical residues only (i.e., not amino acids).
    # Amino acids have well-known 3-letter codes; any residue we don't
    # recognize as canonical is by definition a ligand we assigned a
    # code to.
    _AA_CODES = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
        "TYR", "VAL",
    }
    ligand_counts = {r: n for r, n in counts.items() if r not in _AA_CODES}

    # The core regression assertions: substrate and cofactor kept their
    # chemistry. If the bug is present, both residues collapse to 29-atom
    # generic templates and this exact (9, 53) shape will not appear.
    ligand_atom_totals = sorted(ligand_counts.values())
    assert ligand_atom_totals == [9, 53], (
        f"expected two non-canonical residues with 9 (indole) and 53 (FAD) "
        f"heavy atoms respectively; got {ligand_counts}. If both counts "
        f"read 29, PyRosetta fell back to its generic LIG template -- "
        f"the -extra_res_fa params flag was not applied."
    )
