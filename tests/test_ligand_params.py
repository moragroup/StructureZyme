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


# --- < 3 heavy-atom ligands (metal ions) -------------------------------------
#
# Real halogenases sometimes carry a single-atom metal cofactor (e.g. [Cu+2]
# for lytic-polysaccharide-monooxygenase-like enzymes). Rosetta's
# molfile_to_params.py refuses fragments with fewer than 3 atoms
# (see vendor/rosetta_tools/molfile_to_params.py line 660). We work around
# this by padding sub-3-atom ligands with virtual atoms (Rosetta type VIRT,
# zero energy, zero charge). These tests lock the workaround in place.


def _params_atom_types(params_path: Path) -> list[tuple[str, str]]:
    """Return [(atom_name, rosetta_type), ...] for every ATOM line."""
    out: list[tuple[str, str]] = []
    for line in params_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        out.append((parts[1], parts[2]))
    return out


def test_generate_params_pads_single_atom_metal_ion(tmp_path):
    """[Cu+2] has 1 heavy atom; the padded params must have exactly 1 real
    atom (Cu with rosetta type Cu2p) plus 2 VIRT padding atoms, totalling
    the 3-atom minimum required by molfile_to_params.py.
    """
    from structurezyme.utils.ligand_params import generate_ligand_params

    result = generate_ligand_params(
        smiles="[Cu+2]", resname="Z01", cache_dir=tmp_path / "cache"
    )
    atoms = _params_atom_types(result.params_path)
    assert len(atoms) == 3, f"expected 3 atoms after padding, got {atoms}"

    real = [(n, t) for n, t in atoms if t != "VIRT"]
    virt = [(n, t) for n, t in atoms if t == "VIRT"]
    assert len(real) == 1 and len(virt) == 2, (
        f"expected 1 real + 2 VIRT, got real={real} virt={virt}"
    )
    # Real atom must be copper.
    assert real[0][1] == "Cu2p", f"expected Cu2p atom type, got {real[0]}"


def test_generate_params_preserves_metal_formal_charge(tmp_path):
    """The full +2 formal charge must stay on Cu, not get diluted onto the
    padding atoms. Padding uses VIRT type; molfile_to_params.py assigns
    zero charge to VIRT atoms before distributing any charge offset, so
    Cu keeps its full formal charge. This is the whole reason we pad with
    element 'V' (virtual) rather than e.g. hydrogen or helium.
    """
    from structurezyme.utils.ligand_params import generate_ligand_params

    result = generate_ligand_params(
        smiles="[Cu+2]", resname="Z01", cache_dir=tmp_path / "cache"
    )
    params_text = result.params_path.read_text()
    # CHARGE line format: "CHARGE <name> FORMAL <value>"
    assert "CHARGE CU1  FORMAL 2" in params_text, (
        f"expected +2 formal charge on CU1; params:\n{params_text}"
    )


def test_generate_params_padded_mol_file_has_marker(tmp_path):
    """Padding writes a sidecar `.mol.padded` marker with the 0-indexed
    positions of the injected virtual atoms. The silent-virtualization
    warning check reads it to avoid mis-flagging deliberate padding.
    """
    from structurezyme.utils.ligand_params import generate_ligand_params

    result = generate_ligand_params(
        smiles="[Cu+2]", resname="Z01", cache_dir=tmp_path / "cache"
    )
    mol_path = result.params_path.parent / "Z01.mol"
    marker_path = mol_path.with_suffix(".mol.padded")
    assert marker_path.exists(), (
        f"expected padding marker at {marker_path}, got only "
        f"{list(mol_path.parent.iterdir())}"
    )
    indices = [int(x) for x in marker_path.read_text().split() if x.strip()]
    # [Cu+2] has 1 heavy atom, pads to 3 total, so 2 padding atoms
    # at indices 1 and 2 (Cu is atom 0).
    assert indices == [1, 2], f"expected pad indices [1, 2], got {indices}"


def test_padded_params_does_not_warn_about_padding(tmp_path, caplog):
    """The [Cu+2] warning path must NOT fire for the deliberate padding
    atoms. Only real elements the tool silently virtualized (e.g. actual
    vanadium in the SMILES) should trigger the warning.
    """
    import logging
    from structurezyme.utils.ligand_params import generate_ligand_params

    caplog.set_level(logging.WARNING, logger="structurezyme.utils.ligand_params")
    generate_ligand_params(
        smiles="[Cu+2]", resname="Z01", cache_dir=tmp_path / "cache"
    )
    warnings = [
        r for r in caplog.records
        if "VIRT" in r.getMessage() and "geometric anchor" in r.getMessage()
    ]
    assert not warnings, (
        f"expected NO silent-virtualization warning for padded Cu; got: "
        f"{[r.getMessage() for r in warnings]}"
    )


def test_vanadate_warns_about_silently_virtualized_metal(tmp_path, caplog):
    """Vanadate (V + 4 O, 5 heavy atoms) does NOT hit the padding path,
    but molfile_to_params.py silently virtualizes its vanadium atom due
    to a name-collision with its 'V' virtual-atom convention (comment
    at vendor line 175 acknowledges this). The four oxygens are typed
    correctly; only the V center is neutered. Users must be warned.
    """
    import logging
    from structurezyme.utils.ligand_params import generate_ligand_params

    caplog.set_level(logging.WARNING, logger="structurezyme.utils.ligand_params")
    result = generate_ligand_params(
        smiles="O=[V]([O-])([O-])[O-]",
        resname="Z03",
        cache_dir=tmp_path / "cache",
    )
    warnings = [
        r for r in caplog.records
        if "VIRT" in r.getMessage() and "geometric anchor" in r.getMessage()
    ]
    assert warnings, (
        f"expected silent-virtualization warning for vanadate; params:\n"
        f"{result.params_path.read_text()}"
    )
    # Sanity: the four oxygens must be typed correctly (OOC, carboxyl-like).
    atoms = _params_atom_types(result.params_path)
    ooc_count = sum(1 for _, t in atoms if t == "OOC")
    assert ooc_count == 4, f"expected 4 OOC oxygens, got atoms={atoms}"
