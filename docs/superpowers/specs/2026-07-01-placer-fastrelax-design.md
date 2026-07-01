# PLACER + Rosetta FastRelax Integration Design

**Date:** 2026-07-01
**Status:** Approved, pending user review of this written spec

## Goal

Integrate two new pipeline stages into `filterzyme.pipeline_v2`:

1. **Rosetta FastRelax** — refines docked structures (Chai/Boltz/Vina outputs)
   immediately after docking, before Superimposition and GeometricFiltering
   consume them, so downstream geometric measurements operate on physically
   relaxed coordinates rather than raw docked poses.
2. **PLACER** — re-scores/validates the single final selected structure per
   entry, as the last step before the pipeline's final output file is
   written. This is deliberately placed last because PLACER is computationally
   expensive; running it only on the already-filtered survivor per entry
   keeps the cost bounded regardless of how many docking engines or poses
   were explored earlier in the run.

Both stages are **opt-in** (flags default to `False`), following the existing
`run_vina` convention in `Docking`/`Pipeline`.

## Context / Prior State

The repo analysis (`1_analysis-planning/1_repo_analysis/`) documented PLACER
as dead code before this design:

- Three separate `filterzyme/steps/PLACER_*.py` files each declare
  `class PLACER(Step)` — a name collision if two are ever imported together.
- None of the three is imported by `pipeline_v2.py`. They shell out to a
  `PLACER/run_PLACER.py` sibling repo that is not present in this
  environment.
- The prior recommendation (`12_assumptions_open_questions.md`, Q4) was:
  "drop, reintroduce later from a clean implementation if the use case is
  concrete." This design is that reintroduction.

Rosetta FastRelax has no prior implementation anywhere in this repository —
it is a net-new integration.

## Architecture: Pipeline Placement & Data Flow

```
Docking (existing)
  1. Squidly catalytic-residue prediction (existing, unchanged)
  2. Chai docking (existing, unchanged)
  3. Boltz docking (existing, unchanged)
  4. Vina docking, if run_vina=True (existing, unchanged)
  5. Docking-quality-metrics extraction (existing, unchanged)
  6. [NEW] FastRelax, if run_fastrelax=True
     - Ranks structures per docking engine (Chai/Boltz/Vina) by that
       engine's own confidence metric.
     - Relaxes the top-2 ranked structures per engine per entry.
     - Adds `fastrelax_dir`, `fastrelax_score` columns.

Superimposition (existing) — now operates on relaxed structures when
  FastRelax ran; unchanged otherwise.

GeometricFiltering (existing, unchanged)
  - Produces `structural_features_final.pkl`: one row per entry, the
    single selected best structure, with geometric/pocket/SASA/PLIP
    metrics attached.

[NEW] PLACERValidation, if run_placer=True
  - Consumes `structural_features_final.pkl`.
  - Runs PLACER once per entry on the single final selected structure.
  - Merges `placer_prmsd`, `placer_confidence`, `placer_dir` columns
    back onto the same final DataFrame.
  - Overwrites `structural_features_final.pkl` with the merged result.
```

FastRelax must run *after* docking-quality-metrics extraction, not before,
because it needs each engine's confidence score to select which structures
to relax.

PLACER runs strictly after GeometricFiltering, consuming its output file
directly, because by that point the pipeline has already reduced each entry
to a single best structure — the only point in the pipeline where PLACER's
expensive per-entry cost is bounded to exactly one structure per entry.

## Component: FastRelax Step

**File:** `filterzyme/steps/fastrelax_step.py` (new)
**Class:** `FastRelax(Step)`

### Constructor Parameters

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `input_col` | `str` | required | Column with path(s) to docked structures (mirrors `Vina`/old `PLACER` convention) |
| `confidence_col` | `str` | required | Column used to rank structures per engine (e.g. `chai_ptm`, `boltz_confidence`, `vina_affinity`) |
| `output_dir` | `str` | required | Directory for relaxed PDB outputs |
| `top_k` | `int` | `2` | Number of top-ranked structures per engine to relax |
| `mode` | `Literal["ligand_focused", "full"]` | `"ligand_focused"` | Relax scope |
| `shell_radius` | `float` | `8.0` | Å around ligand for the neighborhood residue selector (ligand-focused mode only) |
| `constraint_weight` | `float` | `1.0` | Coordinate-constraint strength on the shell backbone |
| `scorefunction` | `str` | `"ref2015"` | Rosetta scorefunction name |
| `ligand_resname` | `str` | required | Ligand residue name, used to build the neighborhood selector |
| `num_threads` | `int` | `1` | Parallelism, consistent with other steps |

### Behavior

1. Validate required columns exist in the input DataFrame; raise
   `ValueError` with a clear message otherwise (mirrors
   `Squidly._validate_input`).
2. Per entry, per docking engine present in the DataFrame, rank structures
   by `confidence_col` descending and keep the top `top_k`.
3. For each kept structure, call an internal
   `_relax_one(pdb_path) -> (relaxed_pdb_path, score)`:
   - `pyrosetta.init()` (idempotent, called once per process).
   - If `mode == "ligand_focused"`: build a `NeighborhoodResidueSelector`
     around a residue selector matching `ligand_resname`, radius
     `shell_radius`; construct a `MoveMapFactory` that only allows
     sidechain (and optionally limited backbone) flexibility within the
     selected shell; apply per-residue coordinate constraints (weighted by
     `constraint_weight`) to backbone atoms in the shell.
   - If `mode == "full"`: construct a MoveMap allowing all residues to
     move; coordinate constraints are looser/absent.
   - Build the scorefunction from `scorefunction`, add the constraint-weight
     term when using ligand-focused mode.
   - Apply Rosetta's `FastRelax` mover with the constructed scorefunction
     and MoveMap.
   - Write the relaxed pose to `output_dir`; return its path and final
     Rosetta score.
4. Attach `fastrelax_dir` and `fastrelax_score` columns to the DataFrame.
5. On any per-structure failure (Rosetta exception, missing/corrupt PDB),
   log the error and set that structure's result to `None` — never abort
   the whole run. This matches the existing per-entry-tolerant pattern used
   by Vina's missing-AF2 fallback (`dock_vina_step.py`).

### Testability

- `_select_top_k` (ranking/selection logic): pure Python, no Rosetta
  dependency — always-run unit tests.
- MoveMap/selector parameter construction (shell radius, constraint weight
  wiring): testable by asserting on the constructed selector/movemap
  objects' configuration, without calling `.apply()`.
- `_relax_one` (the actual Rosetta `.apply()` call): wrapped in a
  `pytest.mark.skipif(not pyrosetta_available())` smoke test, following the
  `test_squidly_step.py` convention (`_squidly_cli_available()`,
  `_gpu_available()` helpers).

## Component: PLACER Step

**File:** `filterzyme/steps/PLACER_step.py` (replaces the 3 existing dead,
colliding `PLACER_step.py` / `PLACER_forChai_step.py` / `PLACER_forVina_step.py`
files, which are deleted as part of this work)

**Class:** `PLACER(Step)`

### Constructor Parameters

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `input_col` | `str` | required | Column pointing to the final relaxed/selected structure per entry (from `structural_features_final.pkl`) |
| `entry_col` | `str` | `"Entry"` | Entry identifier column |
| `output_dir` | `str` | required | Directory for PLACER outputs |
| `placer_script_path` | `str` | required | Path to `run_PLACER.py` (or resolved from a `PLACER_HOME` env var); validated at construction — raises `FileNotFoundError` with a clear message if missing. No sibling-repo assumption, unlike the old dead code. |
| `predict_ligand` | `str` | required | Ligand resname passed to PLACER |
| `nsamples` | `int` | `50` | PLACER sample count |
| `rerank` | `str` | `"prmsd"` | PLACER rerank metric |
| `num_threads` | `int` | `1` | Parallelism |

### Behavior

1. Validate `placer_script_path` exists and required columns are present
   in the input DataFrame.
2. Per entry, count ligand instances in the structure (reuse the existing
   `_count_ligands` logic from `PLACER_forChai_step.py`) to decide single-
   vs multi-ligand mode.
3. Shell out via `subprocess.run` to `run_PLACER.py` with
   `--ifile / --odir / --rerank / -n / --predict_ligand`, adding
   `--predict_multi` when multi-ligand is detected. On `AssertionError`
   from multi-mode, fall back to single-mode (same fallback logic as the
   old `PLACER_forChai_step.py`).
4. Parse PLACER's output CSV; extract the configured rerank metric
   (`prmsd` by default) and any confidence score.
5. Merge `placer_prmsd`, `placer_confidence`, `placer_dir` back onto the
   input DataFrame by `Entry` — one row per entry in, one row per entry
   out; no row explosion.
6. On subprocess failure or missing output for an entry, log and set that
   entry's PLACER columns to `None`; never abort the whole run.

### Testability

- `_count_ligands`, CLI-argument construction, and CSV-parse/merge logic:
  pure Python — always-run unit tests.
- The actual `subprocess.run` invocation of `run_PLACER.py`: wrapped in a
  `pytest.mark.skipif(not placer_available())` smoke test (checks
  `placer_script_path` is set and executable), following the
  `test_squidly_step.py` convention.

## Pipeline Wiring

### `Docking` (in `filterzyme/pipeline_v2.py`)

New constructor parameters (mirroring the existing `run_vina` convention):

- `run_fastrelax: bool = False`
- `fastrelax_mode: Literal["ligand_focused", "full"] = "ligand_focused"`
- `fastrelax_top_k: int = 2`
- `fastrelax_shell_radius: float = 8.0`
- `fastrelax_constraint_weight: float = 1.0`
- `fastrelax_scorefunction: str = "ref2015"`

`Docking.run()` calls a new `_run_fastrelax(df_metrics)` method, conditional
on `run_fastrelax`, inserted immediately after
`_extract_docking_quality_metrics` and before `Docking.run()` returns.

### `Pipeline` (in `filterzyme/pipeline_v2.py`)

New constructor parameters:

- `run_placer: bool = False`
- `placer_script_path: str = ''`
- `placer_predict_ligand: str = ''`
- `placer_nsamples: int = 50`
- `placer_rerank: str = "prmsd"`

`Pipeline.run()` gains a new step after `gf.run()`: if `run_placer` is
`True`, construct and run a `PLACERValidation` stage (thin wrapper
composing the `PLACER` step, following the same
`Docking`/`Superimposition`/`GeometricFilters` class pattern already used in
`pipeline_v2.py`) over `geometricfiltering/structural_features_final.pkl`,
and overwrite that file with the merged result.

All new `Docking`/`Pipeline` constructor parameters are forwarded exactly
as the existing `squidly_*` and `run_vina`/`alternative_structure_for_vina`
parameters are today (`Pipeline.run()` passes them through to `Docking`).

## Out of Scope / Deferred

- **Checkpoint / resume (stage hashing, skip-if-exists).** The pipeline
  currently always re-executes every stage on `Pipeline.run()`; there is no
  hash-based caching or manifest (this matches feature request F-4 in
  `1_analysis-planning/1_repo_analysis/09_features_to_add.md`, which is not
  yet implemented). This design does not implement it. Running with
  `run_placer=True` today re-executes Squidly, Chai, Boltz, Vina,
  Superimposition, and GeometricFiltering in full — this is a known,
  accepted cost for this phase. Checkpoint/resume is explicitly deferred to
  a follow-up spec.
- **Full-structure FastRelax performance tuning.** `mode="full"` is
  supported as an option but is expected to be substantially more
  expensive (roughly an order of magnitude slower than ligand-focused mode
  for a ~300-residue enzyme) and is not the default; no special
  parallelization or GPU offload is designed for it in this phase.
- **PLACER/PyRosetta installation and packaging.** Neither dependency is
  added to `environment.yml` or `setup.py` as part of this design's
  scope beyond documenting the required constructor parameters
  (`placer_script_path`, PyRosetta import). Installation instructions are
  an implementation-plan concern, not a design concern.

## Testing Strategy (TDD)

Follows the existing `tests/test_squidly_step.py` convention exactly:

- **Always-run unit tests** (no external/binary dependency): column
  validation, top-K ranking logic, ligand-counting, CLI-argument
  construction, DataFrame merge/column-mapping logic, MoveMap/selector
  parameter construction (asserting on configured objects, not on
  `.apply()` results).
- **Skipped smoke tests**: gated by `pytest.mark.skipif`, checking
  `pyrosetta` importability (FastRelax) and `placer_script_path`
  executable-and-present (PLACER) — mirroring
  `_squidly_cli_available()` / `_gpu_available()`.
- Every new function is written test-first: red → green → refactor, per
  `superpowers:test-driven-development`.
- `test_pipeline.py`-style construction tests are added for the new
  `Docking`/`Pipeline` kwargs (`run_fastrelax`, `run_placer`, etc.),
  verifying signatures and defaults without executing anything — mirroring
  `test_docking_accepts_squidly_kwargs` / `test_pipeline_accepts_squidly_kwargs`.
