# PLACER auto ligand-resname resolution

Date: 2026-08-13
Status: approved

## Problem

`placer_predict_ligand` is a single static string in the run config, passed
verbatim as PLACER's `--predict_ligand`. PLACER selects the named ligand's
atoms; if the name matches nothing, `indices` is empty and PLACER crashes with
`IndexError: list index out of range` (`dataloader_pdb.py:347`).

Since 2026-07-22 (commit `3df8a1c`), FastRelax deterministically renames the
substrate ligand `LIG -> X0N` (Rosetta per-SMILES `.params` codes) via
`_assign_resname` / `_rewrite_ligand_resnames`. When FastRelax is enabled,
PLACER consumes the relaxed PDB whose ligand is now `X0N`, but the config still
says `LIG` -> 0 atoms matched -> crash. A static `X01` "works" only for a
single-substrate run; the code assigns `X01, X02, ...` in order of distinct
substrate SMILES, so a fixed value is wrong for multi-substrate screens.

This regression stayed latent because the full-DAG (fastrelax + placer) path had
not been exercised together since the rename landed.

## Approach

Add an `auto` mode to `placer_predict_ligand`. When the value is `auto` (or
unset), PLACER resolves the ligand resname **per row** from the prepared PDB it
is about to read, instead of using a single static value.

### Selection rule (approved)

Given the distinct HETATM residue names in a prepared PDB:

1. **Prefer the substrate.** If a substrate-style resname is present, predict it:
   - `^X\d{2}$` (FastRelax substrate code, e.g. `X01`), OR
   - `LIG` (the pre-FastRelax / no-FastRelax substrate name).
   If both appear (shouldn't in practice), prefer the `X\d{2}` form.
2. **Else, sole ligand.** If exactly one distinct non-standard ligand resname
   remains after excluding cofactor-style `^Z\d{2}$` codes and standard
   residues/ions/water, predict it.
3. **Else, ambiguous -> raise.** Raise `ValueError` naming the resnames found and
   instructing the user to set `placer_predict_ligand` explicitly. Never guess.

Cofactor-style `^Z\d{2}$` residues are always excluded from selection (they are
fixed, not predicted).

## Components / Changes

### 1. `structurezyme/steps/PLACER_step.py`

- New module-level constants:
  - `_SUBSTRATE_RESNAME_RE = re.compile(r"^X\d{2}$")`
  - `_COFACTOR_RESNAME_RE  = re.compile(r"^Z\d{2}$")`
  - A `_STANDARD_RESIDUES` set (20 aa + common ions/water: `HOH`, `WAT`, `NA`,
    `CL`, `MG`, `ZN`, `CA`, `K`, `MN`, `FE`, etc.) to exclude from the
    "sole ligand" branch.
- New helper `_distinct_hetatm_resnames(pdb_path) -> list[str]`: parse resname
  field `line[17:20].strip()` from `HETATM` lines (mirrors `_count_ligands`).
- New helper `_resolve_predict_ligand(pdb_path) -> str` implementing the rule
  above. Raises `ValueError` on the ambiguous case with an actionable message.
- `_build_cmd(pdb_path, n_ligands, predict_ligand)`: add `predict_ligand` as an
  explicit parameter (was `self.predict_ligand`).
- `execute()`: inside the per-row loop, if
  `str(self.predict_ligand).lower() == "auto"`, compute
  `resolved = self._resolve_predict_ligand(pdb_path)`; else `resolved =
  self.predict_ligand`. Pass `resolved` to both
  `_count_ligands(pdb_path, resolved)` and `_build_cmd(..., resolved)`.
  A `ValueError` from resolution is caught by the existing per-entry try/except
  and produces a `None` score row (logged), consistent with other per-entry
  failures — the run does not abort.

### 2. `structurezyme/step_runners.py` (`run_placer`)

- Change `predict_ligand = opts.get("placer_predict_ligand", None)` +
  hard-raise to `predict_ligand = opts.get("placer_predict_ligand", "auto")`.
  Unset now defaults to `auto` rather than raising.

### 3. `tools/gpu_run/run.yml`

- Set `placer_predict_ligand: auto` with a comment explaining the FastRelax
  `LIG -> X0N` coupling and that `auto` resolves the resname per row from the
  prepared PDB.

## Error handling

- Ambiguous multi-ligand -> `ValueError` (clear, names resnames) -> caught
  per-entry -> `None` scores for that entry, run continues.
- Missing prepared PDB -> existing `FileNotFoundError` behavior unchanged.
- Non-`auto` static values behave exactly as before (backward compatible).

## Testing

Unit tests (`tests/test_placer_auto_resname.py`, no PLACER binary needed):

- `_resolve_predict_ligand`: single `X01` -> `X01`; single `LIG` -> `LIG`;
  substrate + cofactor (`X01` + `Z01`) -> `X01`; `LIG` + `Z01` -> `LIG`;
  two non-substrate ligands (`ABC` + `DEF`) -> raises `ValueError`;
  standard-only (protein, no ligand) -> raises `ValueError`.
- `execute()` with `predict_ligand="auto"` + mocked `subprocess.run` and a small
  on-disk PDB: assert the resolved resname (`X01`) reaches the built command and
  is used by `_count_ligands`.

Integration: re-run the full-DAG GPU smoke
(`tools/smoke/run_full_dag_smoke.sbatch`) -> expect PLACER PASS (CSV produced,
`placer_*` columns populated) and `SMOKE RESULT: PASS`.

## Out of scope

- No change to FastRelax's naming scheme (`X0N`/`Z0N` stays).
- No threading of the resname through the dataframe/checkpoint; resolution reads
  the PDB directly (works for both fastrelax and no-fastrelax paths).
- No multi-substrate-per-entry disambiguation beyond the substrate-preference
  rule; the ambiguous case raises rather than guessing.
