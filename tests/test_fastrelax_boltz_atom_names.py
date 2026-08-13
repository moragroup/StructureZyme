"""Regression test: FastRelax must handle boltz's SMILES-index atom names.

Chai emits ligand PDBs with canonical element-grouped atom names (C1..C8,
N1 for indole). Boltz emits them in kekulized SMILES order with mixed
elements (C8, C9, C12, C15, N16, C13, C11, C14, C10). Rosetta's
`.params` files use the canonical form, so loading a boltz PDB without
first renaming the ligand atoms triggers::

    ERROR: too many tries in fill_missing_atoms!
    core.conformation.Conformation:  [ ERROR ] ...

This test locks in the fix: _prepare_pdb_for_pose must rename the
ligand atoms in every input PDB to match the `.params` template, using
RDKit substructure matching (NOT positional guessing -- boltz's atom
ORDER differs from ours as well).

Fixture: `A0A2N7TXS4_boltz_min.pdb` -- one of the boltz poses from
the fmo-fad-01 run that failed with `fill_missing_atoms`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from structurezyme.steps.fastrelax_step import FastRelax, _pyrosetta_available


FIXTURE = Path(__file__).parent / "fixtures" / "fastrelax" / "A0A2N7TXS4_boltz_min.pdb"

INDOLE_SMILES = "C1=CC=C2C(=C1)C=CN2"
FAD_SMILES = (
    "CC1=CC2=C(C=C1C)N(C3=NC(=O)NC(=O)C3=N2)C[C@@H]([C@@H]([C@@H]"
    "(COP(=O)(O)OP(=O)(O)OC[C@@H]4[C@H]([C@H]([C@@H](O4)N5C=NC6="
    "C(N=CN=C65)N)O)O)O)O)O"
)

_AA_CODES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL",
}


def _heavy_atom_count_by_resname(pdb_path: Path) -> dict[str, int]:
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
    reason="RosettaFastRelax venv not reachable (set FASTRELAX_ENV)",
)
def test_fastrelax_handles_boltz_atom_names(tmp_path):
    """FastRelax must succeed on a boltz PDB with SMILES-index atom names."""
    assert FIXTURE.exists(), f"fixture missing: {FIXTURE}"

    fixture_counts = _heavy_atom_count_by_resname(FIXTURE)
    assert fixture_counts.get("LIG") == 9 + 53, (
        f"fixture malformed: expected 62 LIG heavy atoms, "
        f"got {fixture_counts.get('LIG')}"
    )

    # Filename must match the pipeline's boltz-classifier so
    # _confidence_key_from_path picks the right column bucket.
    pose_path = tmp_path / "A0A2N7TXS4_model_0_boltz.pdb"
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
            "Entry": ["A0A2N7TXS4"],
            "substrate_smiles": [INDOLE_SMILES],
            "cofactor_smiles": [FAD_SMILES],
            "chai_files_for_superimposition": [[]],
            "boltz_files_for_superimposition": [[str(pose_path)]],
            "vina_files_for_superimposition": [[]],
            "chai_ptm": [{}],
            "boltz2_confidence_score": [{"A0A2N7TXS4_model_0": 0.9}],
            "vina_affinities": [{}],
        }
    )

    out = fr.execute(df)

    relaxed_files = out.iloc[0]["boltz_files_for_superimposition"]
    assert len(relaxed_files) == 1, (
        f"expected exactly one relaxed boltz pose, got {relaxed_files}. "
        f"If empty, the boltz atom-name mismatch bug is still present -- "
        f"Rosetta's fill_missing_atoms fails and the pose is dropped."
    )
    relaxed_path = Path(relaxed_files[0])
    assert relaxed_path.exists(), f"relaxed PDB not written: {relaxed_path}"

    counts = _heavy_atom_count_by_resname(relaxed_path)
    assert "LIG" not in counts, (
        f"relaxed PDB still contains generic LIG; renaming did not run: {counts}"
    )
    ligand_counts = {r: n for r, n in counts.items() if r not in _AA_CODES}
    ligand_atom_totals = sorted(ligand_counts.values())
    assert ligand_atom_totals == [9, 53], (
        f"expected 9 (indole) + 53 (FAD) heavy atoms; got {ligand_counts}"
    )
