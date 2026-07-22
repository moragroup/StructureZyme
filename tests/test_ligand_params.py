"""Tests for `structurezyme.utils.ligand_params`.

Covers the SMILES -> Rosetta .params helper used by FastRelax.
The module wraps the vendored `molfile_to_params.py` with a
Python-friendly, cache-aware API.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _rdkit_available() -> bool:
    try:
        import rdkit  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _rdkit_available(),
    reason="rdkit is not importable in this environment",
)


INDOLE_SMILES = "C1=CC=C2C(=C1)C=CN2"


def _params_atom_count(params_path: Path) -> int:
    """Count `^ATOM` lines in a Rosetta .params file (includes H)."""
    return sum(
        1 for line in params_path.read_text().splitlines() if line.startswith("ATOM")
    )


def _params_heavy_atom_count(params_path: Path) -> int:
    """Count non-H `^ATOM` lines in a Rosetta .params file.

    Params ATOM lines look like: `ATOM  C1  aroC  X  -0.09`
    Column 2 is the atom name; a leading 'H' identifies a hydrogen.
    """
    n = 0
    for line in params_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        atom_name = parts[1]
        if atom_name.startswith("H"):
            continue
        n += 1
    return n


# --- generate_ligand_params: happy path -------------------------------------


def test_generate_indole_params_produces_9_heavy_atoms(tmp_path):
    """Indole (SMILES C1=CC=C2C(=C1)C=CN2) has 9 heavy atoms. The
    generated .params must reflect that -- if it doesn't, we've fallen
    into the same trap as pose_from_pdb's 29-atom generic template.
    """
    from structurezyme.utils.ligand_params import generate_ligand_params

    result = generate_ligand_params(
        smiles=INDOLE_SMILES,
        resname="IND",
        cache_dir=tmp_path,
    )

    assert result.params_path.exists(), (
        f"generate_ligand_params returned a path that doesn't exist: "
        f"{result.params_path}"
    )
    assert result.params_path.name == "IND.params"
    assert result.smiles  # canonical SMILES echoed back
    assert result.resname == "IND"
    assert len(result.conformer_pdb_paths) >= 1
    assert all(p.exists() for p in result.conformer_pdb_paths)

    heavy = _params_heavy_atom_count(result.params_path)
    assert heavy == 9, (
        f"indole .params has {heavy} heavy atoms, expected 9. This means "
        f"molfile_to_params ran on the wrong input or the params file is "
        f"the 29-atom generic LIG template."
    )


# --- resname validation ------------------------------------------------------


@pytest.mark.parametrize(
    "bad_resname",
    [
        "",       # empty
        "AB",     # too short
        "ABCD",   # too long
        "abc",    # lowercase
        "12A",    # starts with digit
        "A B",    # whitespace
        "1AB",    # starts with digit
    ],
)
def test_generate_rejects_bad_resname(tmp_path, bad_resname):
    """Rosetta + PDB spec: 3 chars, first must be a letter, rest may be
    letters or digits (real examples: HEM, S01, HG1)."""
    from structurezyme.utils.ligand_params import generate_ligand_params

    with pytest.raises(ValueError, match="resname"):
        generate_ligand_params(
            smiles=INDOLE_SMILES,
            resname=bad_resname,
            cache_dir=tmp_path,
        )


@pytest.mark.parametrize("good_resname", ["IND", "HEM", "S01", "C12", "HG1"])
def test_generate_accepts_letter_plus_alnum_resname(tmp_path, good_resname):
    """Sanity: the common valid patterns must not be rejected by the
    validator. Uses `overwrite=True` so each parametrized case really
    exercises the codepath.
    """
    from structurezyme.utils.ligand_params import generate_ligand_params

    result = generate_ligand_params(
        smiles=INDOLE_SMILES,
        resname=good_resname,
        cache_dir=tmp_path,
    )
    assert result.params_path.name == f"{good_resname}.params"


# --- bad SMILES --------------------------------------------------------------


def test_generate_raises_on_unparseable_smiles(tmp_path):
    """Garbage SMILES must fail loudly via LigandParamsError, not
    silently produce a broken params file.
    """
    from structurezyme.utils.ligand_params import (
        generate_ligand_params,
        LigandParamsError,
    )

    with pytest.raises(LigandParamsError):
        generate_ligand_params(
            smiles="this is not a smiles",
            resname="BAD",
            cache_dir=tmp_path,
        )


# --- caching -----------------------------------------------------------------


def test_generate_uses_cache_on_second_call(tmp_path):
    """Second call with same (smiles, resname, cache_dir) must reuse
    the cached artifacts -- crucial for pipelines that relax dozens of
    poses with the same substrate/cofactor pair.
    """
    from structurezyme.utils.ligand_params import generate_ligand_params

    first = generate_ligand_params(
        smiles=INDOLE_SMILES, resname="IND", cache_dir=tmp_path
    )
    mtime_before = first.params_path.stat().st_mtime_ns

    second = generate_ligand_params(
        smiles=INDOLE_SMILES, resname="IND", cache_dir=tmp_path
    )

    assert second.params_path == first.params_path
    assert second.params_path.stat().st_mtime_ns == mtime_before, (
        "second call rewrote the params file; cache miss"
    )


def test_overwrite_forces_regeneration(tmp_path):
    """`overwrite=True` must ignore the cache and rewrite the params."""
    from structurezyme.utils.ligand_params import generate_ligand_params

    first = generate_ligand_params(
        smiles=INDOLE_SMILES, resname="IND", cache_dir=tmp_path
    )
    mtime_before = first.params_path.stat().st_mtime_ns

    # Sleep-free bump: touch the file back to an older mtime, then call
    # with overwrite. If cache is honored we keep the old mtime; if
    # overwrite works we get a fresher one.
    import os
    os.utime(first.params_path, ns=(mtime_before - 10**9, mtime_before - 10**9))

    second = generate_ligand_params(
        smiles=INDOLE_SMILES,
        resname="IND",
        cache_dir=tmp_path,
        overwrite=True,
    )
    assert second.params_path.stat().st_mtime_ns > mtime_before - 10**9


# --- canonical_smiles --------------------------------------------------------


def test_canonical_smiles_is_stable_across_equivalent_inputs():
    """Two equivalent SMILES for indole must canonicalize identically,
    so the cache key is the same regardless of how the caller wrote it.
    """
    from structurezyme.utils.ligand_params import canonical_smiles

    c1 = canonical_smiles("C1=CC=C2C(=C1)C=CN2")
    c2 = canonical_smiles("c1ccc2[nH]ccc2c1")  # aromatic form
    assert c1 == c2, f"canonical mismatch: {c1!r} vs {c2!r}"


def test_canonical_smiles_raises_on_garbage():
    from structurezyme.utils.ligand_params import (
        canonical_smiles,
        LigandParamsError,
    )

    with pytest.raises(LigandParamsError):
        canonical_smiles("not a smiles")
