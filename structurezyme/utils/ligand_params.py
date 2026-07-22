"""SMILES -> Rosetta .params helper for non-canonical ligands.

Wraps the vendored `structurezyme/vendor/rosetta_tools/molfile_to_params.py`
in a Python-friendly API. Given a SMILES, produces a `.params` file
suitable for `pyrosetta.init(extra_options='-extra_res_fa <path>')`.

Params generation is expensive and deterministic in the SMILES, so
results are cached on disk under a content-addressable subdirectory
of the caller-supplied `cache_dir`. Cache subdir name is a stable
16-hex-char SHA-1 of `f"{resname}\\n{canonical_smiles(smiles)}"`.

Concurrency: `filelock.FileLock` around the generation step means two
processes running against the same `cache_dir` won't corrupt each
other's outputs.
"""
from __future__ import annotations

import hashlib
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# The vendored molfile_to_params.py lives at:
#   structurezyme/vendor/rosetta_tools/molfile_to_params.py
# This module lives at:
#   structurezyme/utils/ligand_params.py
_VENDOR_DIR = (Path(__file__).parent.parent / "vendor" / "rosetta_tools").resolve()
MOLFILE_TO_PARAMS = _VENDOR_DIR / "molfile_to_params.py"

# Per PDB spec, HETATM residue names are 1-3 chars of uppercase letters
# and digits, first char must be a letter. Common real codes include
# 'HEM', 'ATP', 'FAD', 'S01', 'MN2', 'HG1'.
_RESNAME_RE = re.compile(r"^[A-Z][A-Z0-9]{2}$")


class LigandParamsError(RuntimeError):
    """Raised when params generation fails.

    Wraps: SMILES parsing errors, 3D embedding failures (after retry),
    and non-zero exit from the molfile_to_params subprocess. Original
    stdout/stderr are attached to the message when available.
    """


@dataclass(frozen=True)
class LigandParams:
    """Generated `.params` bundle for one ligand.

    Attributes:
        smiles: canonical (RDKit-normalized) SMILES of the input.
        resname: 3-letter residue code embedded in the params file.
        params_path: absolute path to `<resname>.params`.
        conformer_pdb_paths: reference conformer PDB(s) produced
            alongside the params (at minimum `<resname>_0001.pdb`).
    """

    smiles: str
    resname: str
    params_path: Path
    conformer_pdb_paths: list[Path] = field(default_factory=list)


def canonical_smiles(smiles: str) -> str:
    """Return RDKit's canonical SMILES for the input.

    Raises `LigandParamsError` if the SMILES is unparseable. Used as
    the stable half of the cache key so callers can pass any
    equivalent form (Kekule vs aromatic, atom-ordered differently,
    etc.) and hit the same cache entry.
    """
    try:
        from rdkit import Chem
    except ImportError as e:
        raise LigandParamsError(
            "rdkit is required for ligand params generation"
        ) from e

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise LigandParamsError(f"could not parse SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol)


def _smiles_cache_key(smiles: str, resname: str) -> str:
    """SHA-1 hex (16 chars) of `f"{resname}\\n{canonical_smiles(smiles)}"`.

    Not part of the public API. Used as the cache-subdirectory name.
    """
    payload = f"{resname}\n{canonical_smiles(smiles)}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _validate_resname(resname: str) -> None:
    """Raise ValueError if `resname` does not match the PDB HETATM spec.

    Rule: exactly 3 chars; first must be an uppercase ASCII letter;
    remaining chars are uppercase ASCII letters or digits. Real
    examples: HEM, ATP, FAD, S01, MN2, HG1.
    """
    if not _RESNAME_RE.match(resname):
        raise ValueError(
            f"resname must be 3 chars, first an uppercase letter, rest "
            f"letters or digits (e.g. HEM, S01, HG1); got {resname!r}"
        )


def _build_mol_file(smiles: str, mol_path: Path, seed: int = 42) -> None:
    """Write an MDL V2000 mol file from `smiles` at `mol_path`.

    Pipeline: parse SMILES -> add explicit hydrogens -> embed 3D with
    ETKDG (deterministic via `seed`) -> MMFF-optimize -> kekulize
    (clearAromaticFlags=True) -> write.

    On embed failure, retries once with `useRandomCoords=True` and
    `maxAttempts=1000`. Raises `LigandParamsError` if both attempts
    fail.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as e:
        raise LigandParamsError(
            "rdkit is required for ligand params generation"
        ) from e

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise LigandParamsError(f"could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)

    # First attempt: default ETKDG.
    result = AllChem.EmbedMolecule(mol, randomSeed=seed)
    if result != 0:
        logger.info(
            "RDKit ETKDG embed failed for %r; retrying with random coords",
            smiles,
        )
        result = AllChem.EmbedMolecule(
            mol,
            randomSeed=seed,
            useRandomCoords=True,
            maxAttempts=1000,
        )
    if result != 0:
        raise LigandParamsError(
            f"RDKit could not embed 3D coordinates for SMILES {smiles!r} "
            f"after both default and useRandomCoords attempts"
        )

    # Best-effort optimization; failure is non-fatal.
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception as e:  # pragma: no cover - very rare
        logger.warning("MMFF optimization failed for %r: %s", smiles, e)

    Chem.Kekulize(mol, clearAromaticFlags=True)
    Chem.MolToMolFile(mol, str(mol_path))


def _run_molfile_to_params(mol_path: Path, resname: str, work_dir: Path) -> None:
    """Invoke the vendored `molfile_to_params.py` as a subprocess.

    Runs in `work_dir` so the emitted `<resname>.params` and
    `<resname>_NNNN.pdb` files land alongside the input mol file
    rather than in the CWD. Uses the current Python interpreter so
    the vendored script's rdkit/numpy dependencies come from the
    active env.

    Raises `LigandParamsError` on non-zero exit, attaching the last
    ~40 lines of combined stdout/stderr to the exception message.
    """
    if not MOLFILE_TO_PARAMS.exists():
        raise LigandParamsError(
            f"vendored molfile_to_params.py not found at {MOLFILE_TO_PARAMS}; "
            f"see structurezyme/vendor/rosetta_tools/README.md for how to "
            f"restore it"
        )

    cmd = [
        sys.executable,
        str(MOLFILE_TO_PARAMS),
        "-n",
        resname,
        "--clobber",
        mol_path.name,
    ]
    logger.debug("running molfile_to_params: %s (in %s)", cmd, work_dir)
    proc = subprocess.run(
        cmd,
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        combined = (proc.stdout or "") + (proc.stderr or "")
        tail = "\n".join(combined.splitlines()[-40:])
        raise LigandParamsError(
            f"molfile_to_params.py exited {proc.returncode} for resname "
            f"{resname!r}. Last output:\n{tail}"
        )

    expected = work_dir / f"{resname}.params"
    if not expected.exists():
        raise LigandParamsError(
            f"molfile_to_params.py exited 0 but did not produce {expected} "
            f"(stdout tail: {proc.stdout[-500:] if proc.stdout else '<empty>'})"
        )


def _list_conformer_pdbs(work_dir: Path, resname: str) -> list[Path]:
    """Return `<resname>_NNNN.pdb` files in `work_dir`, sorted by name."""
    return sorted(work_dir.glob(f"{resname}_[0-9][0-9][0-9][0-9].pdb"))


def generate_ligand_params(
    smiles: str,
    resname: str,
    cache_dir: Path | str,
    *,
    conformers: int = 1,  # noqa: ARG001 - reserved for future multi-conformer support
    random_seed: int = 42,
    overwrite: bool = False,
) -> LigandParams:
    """Generate a Rosetta `.params` file for a ligand from its SMILES.

    See module docstring for pipeline details. Cache key is
    `_smiles_cache_key(smiles, resname)`; artifacts live under
    `cache_dir / <hash> /`.

    Args:
        smiles: SMILES string. Any equivalent form (aromatic vs Kekule,
            atom order) hits the same cache entry.
        resname: 3-letter uppercase ASCII residue code (e.g. 'SUB').
        cache_dir: parent directory for the content-addressable cache.
            Created if missing.
        conformers: reserved for future use; currently only 1 conformer
            is emitted regardless of this value.
        random_seed: seed for RDKit's ETKDG embedder.
        overwrite: if True, ignore any cached artifacts and regenerate.

    Returns:
        A `LigandParams` bundle pointing at the generated files.

    Raises:
        ValueError: bad resname.
        LigandParamsError: RDKit or subprocess failure.
    """
    _validate_resname(resname)

    try:
        from filelock import FileLock
    except ImportError as e:
        raise LigandParamsError(
            "filelock is required for concurrent-safe params generation"
        ) from e

    canon = canonical_smiles(smiles)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _smiles_cache_key(smiles, resname)
    work_dir = cache_dir / key
    params_path = work_dir / f"{resname}.params"
    lock_path = cache_dir / f"{key}.lock"

    # Fast path: already cached, no lock needed.
    if params_path.exists() and not overwrite:
        return LigandParams(
            smiles=canon,
            resname=resname,
            params_path=params_path,
            conformer_pdb_paths=_list_conformer_pdbs(work_dir, resname),
        )

    # Slow path: acquire the lock, re-check, then generate.
    with FileLock(str(lock_path)):
        if params_path.exists() and not overwrite:
            return LigandParams(
                smiles=canon,
                resname=resname,
                params_path=params_path,
                conformer_pdb_paths=_list_conformer_pdbs(work_dir, resname),
            )

        work_dir.mkdir(parents=True, exist_ok=True)
        mol_path = work_dir / f"{resname}.mol"
        _build_mol_file(smiles, mol_path, seed=random_seed)
        _run_molfile_to_params(mol_path, resname, work_dir)

        return LigandParams(
            smiles=canon,
            resname=resname,
            params_path=params_path,
            conformer_pdb_paths=_list_conformer_pdbs(work_dir, resname),
        )


def _read_hetatm_atom_names(pdb_text: str) -> list[str]:
    """Return HETATM atom names in file order.

    Reads columns 13-16 of each HETATM line (PDB spec: "Atom name").
    Whitespace is stripped so callers get bare tokens like ``"C1"``,
    ``"N1"``, ``"C12"``.
    """
    names: list[str] = []
    for line in pdb_text.splitlines():
        if line.startswith("HETATM"):
            names.append(line[12:16].strip())
    return names


def build_atom_name_remap(
    reference_pdb: Path,
    target_pdb_text: str,
    target_chain: str,
    smiles: str,
) -> dict[str, str]:
    """Compute an atom-name mapping from `target_pdb_text` -> `reference_pdb`.

    Both PDBs contain the same small molecule (`smiles`); ``reference_pdb``
    was produced by ``molfile_to_params.py`` and its atom names match the
    ``.params`` file exactly. ``target_pdb_text`` is a docked-pose PDB
    whose ligand atoms may use a different naming scheme (e.g. boltz
    uses SMILES-kekulize order like ``C8, C9, C12, C15, N16, ...``).

    Strategy: load both as RDKit mols (with proximity bonding on the
    target since PDBs don't carry bond orders), infer bond orders on the
    target using the SMILES as a template, then use RDKit's
    ``GetSubstructMatch`` to find the atom-index bijection between
    reference and target. Emit a ``{old_name -> new_name}`` dict.

    Returns an empty dict on any failure -- caller should treat that as
    "no rename possible" and log a warning.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as e:
        raise LigandParamsError("rdkit is required for atom-name remap") from e

    # Extract just the ligand HETATMs on `target_chain` from the input PDB.
    target_lines = [
        line
        for line in target_pdb_text.splitlines()
        if line.startswith("HETATM") and len(line) > 21 and line[21] == target_chain
    ]
    if not target_lines:
        return {}
    target_block = "\n".join(target_lines) + "\nEND\n"

    ref_pdb_text = Path(reference_pdb).read_text()

    # Load both mols. removeHs=False so we can inspect original PDBResidueInfo
    # for hydrogen atom names if we need to (currently we only map heavy
    # atoms, since docked-pose PDBs typically don't carry ligand H's and
    # Rosetta rebuilds them from the params template).
    target_mol = Chem.MolFromPDBBlock(
        target_block, sanitize=False, removeHs=False, proximityBonding=True
    )
    ref_mol = Chem.MolFromPDBBlock(
        ref_pdb_text, sanitize=False, removeHs=False, proximityBonding=True
    )
    if target_mol is None or ref_mol is None:
        return {}

    # Remove H's BEFORE bond-order assignment. Rationale: the reference
    # conformer PDB from molfile_to_params carries explicit H's on the
    # aromatic N (e.g. indole N-H). AssignBondOrdersFromTemplate uses a
    # heavy-atom-only SMILES template, so if we leave the explicit H on
    # the N, the template's aromatic N + explicit H produces a valence
    # error ("Explicit valence for atom N, 5, is greater than permitted").
    # We only care about heavy-atom names anyway -- Rosetta fills H's
    # from the params template.
    target_noh = Chem.RemoveHs(target_mol, sanitize=False)
    ref_noh = Chem.RemoveHs(ref_mol, sanitize=False)

    # Assign bond orders from the SMILES template so aromaticity and
    # ring-bond types line up. Without this, GetSubstructMatch on a
    # PDB-parsed mol with all-single bonds against an aromatic template
    # returns no match.
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        return {}
    try:
        target_fixed = AllChem.AssignBondOrdersFromTemplate(template, target_noh)
        ref_fixed = AllChem.AssignBondOrdersFromTemplate(template, ref_noh)
    except Exception as e:
        logger.warning("build_atom_name_remap: bond-order assignment failed: %s", e)
        return {}

    # Substructure match on the heavy-atom skeleton.
    match = target_fixed.GetSubstructMatch(template)
    ref_match = ref_fixed.GetSubstructMatch(template)
    if not match or not ref_match or len(match) != len(ref_match):
        return {}

    # For each template atom index i:
    #   target atom idx  = match[i]      -> its PDB name in target
    #   reference atom idx = ref_match[i] -> its PDB name in reference
    mapping: dict[str, str] = {}
    for template_idx in range(len(match)):
        t_atom = target_fixed.GetAtomWithIdx(match[template_idx])
        r_atom = ref_fixed.GetAtomWithIdx(ref_match[template_idx])
        t_info = t_atom.GetPDBResidueInfo()
        r_info = r_atom.GetPDBResidueInfo()
        if t_info is None or r_info is None:
            continue
        old_name = t_info.GetName().strip()
        new_name = r_info.GetName().strip()
        if old_name and new_name:
            mapping[old_name] = new_name

    return mapping


def rewrite_chain_atom_names(
    pdb_text: str,
    chain: str,
    name_map: dict[str, str],
) -> str:
    """Rewrite the atom name (cols 13-16) of every HETATM line on `chain`
    whose current atom name appears in `name_map`.

    ``name_map`` is ``{old_name: new_name}``. Both are stripped, but the
    output is padded/aligned to fit the fixed 4-char PDB atom-name field
    following the same convention as ``pyrosetta``: names <= 3 chars
    are right-padded with a leading space (col 13 is the element
    alignment column), names of 4 chars fill the field exactly. Non-H
    HETATMs are more permissive here since molfile_to_params-generated
    names for our use case (indole, FAD) all fit in <= 3 chars.
    """
    def _pad(name: str) -> str:
        # PDB atom-name field is 4 chars (cols 13-16, 0-indexed 12:16).
        # Standard alignment: for names of length <= 3, place a leading
        # space so col 13 (index 12) is blank and the element letter
        # sits at col 14 (index 13). Names of length 4 fill the field.
        if len(name) >= 4:
            return name[:4]
        return " " + name.ljust(3)

    out: list[str] = []
    for line in pdb_text.splitlines():
        if (
            line.startswith("HETATM")
            and len(line) > 21
            and line[21] == chain
        ):
            old = line[12:16].strip()
            new = name_map.get(old)
            if new is not None:
                line = line[:12] + _pad(new) + line[16:]
        out.append(line)
    return "\n".join(out) + "\n"
