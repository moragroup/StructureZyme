# PLACER + Rosetta FastRelax Integration Design

**Date:** 2026-07-01
**Status:** Approved, pending user review of this written spec

**Revision note (2026-07-06):** A pre-implementation research pass across
the actual pipeline code (not just the prior repo-analysis docs) found that
several assumptions in the original version of this spec did not match
reality: the shape of the data at each pipeline stage, the ranking metrics
available per docking engine, where a full protein+ligand complex file
first exists for each engine, and how a PLACER-consumable PDB path is
obtained from `structural_features_final.pkl`. Sections below have been
corrected in place; see "Ground-Truth Corrections" for a summary of what
changed and why.

## Goal

Integrate two new pipeline stages into `filterzyme.pipeline_v2`:

1. **Rosetta FastRelax** — refines docked structures (Chai/Boltz/Vina outputs)
   after each engine's raw output has been converted into a single
   protein+ligand complex PDB file (see "Ground-Truth Corrections" — this is
   inside `Superimposition`, not `Docking`), before the actual pairwise
   superimposition step consumes them, so downstream geometric measurements
   operate on physically relaxed coordinates rather than raw docked poses.
2. **PLACER** — re-scores/validates one selected structure per entry, as the
   last step before the pipeline's final output file is written. This is
   deliberately placed last because PLACER is computationally expensive;
   running it only on one structure per entry keeps the cost bounded
   regardless of how many docking engines or poses were explored earlier in
   the run. (Note: unlike originally assumed, `structural_features_final.pkl`
   is not already reduced to one row per entry — PLACER's own step must
   perform this reduction; see "Ground-Truth Corrections".)

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

## Ground-Truth Corrections

A research pass reading `pipeline_v2.py`, `extract_docking_metrics_step.py`,
`computeligandRMSD_step.py`, `computeproteinRMSD_step.py`,
`superimposestructures_step.py`, `preparechai_step.py`, `prepareboltz_step.py`,
`preparevina_step.py`, and the underlying `docko`/Chai/Boltz writer code
found the following facts that differ from the original version of this
spec. Each is a hard fact backed by exact code citations (see the research
log for full detail); the rest of this document has been updated to be
consistent with them.

1. **`structural_features_final.pkl` is one row per `(Entry, docked_structure)`
   pose, not one row per entry.** Nothing in `GeometricFilters` (or any step
   it calls) reduces rows to a single best structure per entry. There IS an
   `is_best` boolean column (set in `computeligandRMSD_step.py`, based on
   whether any of 3 selection heuristics — `inter_tool_weighted_avg`,
   `inter_tool_min_per_tool`, `vina_avg_intra_tool` — picked that structure,
   recorded in a comma-joined `best_method` column), but nothing filters by
   it anywhere downstream today. **Decision:** PLACER's own step performs
   this reduction (see Component: PLACER Step, "Selecting one structure per
   entry" below) — it is not free from the existing pipeline.

2. **FastRelax cannot rank poses using a flat `confidence_col` at the
   `dockingmetrics.pkl` stage.** At that stage, Chai/Boltz confidence metrics
   (`chai_ptm`, `boltz2_confidence_score`, etc.) are **dict-valued columns**,
   one dict per entry-row, keyed by a per-engine structure-ID string (see
   Component: FastRelax Step, "Ranking structures per engine" below for the
   exact key formats, which differ between Chai and Boltz). A single scalar
   `confidence_col` parameter cannot express this.

3. **Vina's binding-affinity scores are calculated by Vina itself but never
   parsed back into the pipeline's DataFrame.** `dock_vina()`
   (`docko/vina.py`) writes a log file
   (`{output_dir}/{protein_name}-{ligand_name}_log.txt`) containing Vina's
   own `mode | affinity` table, and `extract_docking_metrics_step.py`
   already contains a `parse_vina_output(file_path)` function that can parse
   this exact log format — but it is **never called** anywhere in
   `DockingMetrics`. As a result, `vina_affinities` is always absent/NaN in
   the current pipeline. **Decision:** this integration adds a prerequisite
   task (Task 1 in the implementation plan) to wire `parse_vina_output` into
   `DockingMetrics`, populating a `vina_affinities` dict column (keyed by
   1-indexed pose number, matching `PrepareVina`'s `{entry}_{i}_vina.pdb`
   naming) analogous to the existing Chai/Boltz dict columns, so FastRelax
   can rank Vina poses the same way it ranks Chai/Boltz poses (ascending —
   more negative kcal/mol is better — instead of descending).

4. **Vina never produces a single-file protein+ligand complex until
   `PrepareVina` runs.** Vina's raw docking output
   (`{Entry}.pdb` = protein only, `{Entry}-{ligand_name}.pdb` = ligand poses
   only) is two separate files; `PrepareVina.split_ligands_and_combine`
   (`preparevina_step.py`) is what combines them into
   `{entry}_{i}_vina.pdb` complex files, and that combination only happens
   inside `Superimposition._prepare_files_for_superimposition`, which runs
   *after* `Docking` completes. Chai and Boltz, by contrast, already produce
   single-file complexes as their raw output (`{Entry}_{idx}.cif` /
   `{Entry}_model_{idx}.cif`). **Decision:** FastRelax is moved from being
   part of `Docking` (as originally specified) to being part of
   `Superimposition`, running immediately after
   `_prepare_files_for_superimposition` (which already produces uniform,
   ready-to-relax complex PDBs for all three engines via
   `chai_files_for_superimposition` / `boltz_files_for_superimposition` /
   `vina_files_for_superimposition`) and before `_superimposition` consumes
   those same columns. This also means FastRelax's relaxed output is already
   in the exact list-column format `_superimposition` expects — no separate
   "skip the old conversion step for relaxed structures" branch is needed,
   because FastRelax runs strictly after that conversion, replacing the
   paths in those same columns in place.

5. **`structural_features_final.pkl` has no PDB-path column.** It carries
   only the `docked_structure` string ID (e.g. `"Q97WW0_0_chai"`); every
   existing downstream consumer of this file/stage
   (`GeneralGeometricFiltering`, `EsteraseGeometricFiltering`, `LigandSASA`,
   `PLIP`, `Fpocket`) reconstructs the actual PDB path itself as
   `preparedfiles_dir / f"{docked_structure}.pdb"` against the
   `preparedfiles_for_superimposition/` directory — there is no `input_col`
   pointing at a ready-made path, contrary to the original version of this
   spec. **Decision:** the `PLACER` step takes a `preparedfiles_dir`
   parameter and reconstructs each selected row's PDB path the same way
   (see Component: PLACER Step, "Constructor Parameters").

## Architecture: Pipeline Placement & Data Flow

```
Docking (existing, unchanged apart from Task 1's Vina-affinity wiring)
  1. Squidly catalytic-residue prediction (existing, unchanged)
  2. Chai docking (existing, unchanged)
  3. Boltz docking (existing, unchanged)
  4. Vina docking, if run_vina=True (existing, unchanged)
  5. Docking-quality-metrics extraction
     - [Task 1, prerequisite] Now also parses Vina's log file via the
       existing (currently-unwired) `parse_vina_output` and populates a
       `vina_affinities` dict column (keyed by 1-indexed pose number),
       analogous to the existing `chai_ptm`/`boltz2_confidence_score` dict
       columns.

Superimposition (existing, extended)
  1. `_prepare_files_for_superimposition` (existing, unchanged) — converts
     each engine's raw docking output into per-entry lists of ready-to-use
     complex PDB paths: `chai_files_for_superimposition`,
     `boltz_files_for_superimposition`, `vina_files_for_superimposition`.
     This is the first point in the pipeline where all three engines have a
     single-file protein+ligand complex on disk (Vina's complex file does
     not exist until this step's `PrepareVina` runs).
  2. [NEW] FastRelax, if run_fastrelax=True — runs immediately after step 1
     and before `_superimposition`:
     - Ranks structures per docking engine (Chai/Boltz/Vina) by that
       engine's own confidence metric (descending for Chai/Boltz
       `*_ptm`/`*_confidence_score`; ascending for Vina `vina_affinities`,
       since more negative kcal/mol is better).
     - Relaxes the top-`fastrelax_top_k` ranked poses per engine per entry.
     - Replaces the relaxed poses' paths in place within
       `chai_files_for_superimposition` / `boltz_files_for_superimposition` /
       `vina_files_for_superimposition`. If `fastrelax_drop_unrelaxed=True`
       (default), lower-ranked unrelaxed poses for that engine are removed
       from these lists; if `False`, both relaxed and unrelaxed poses are
       kept side by side for comparison.
     - Adds `fastrelax_score` (dict keyed the same way as the confidence
       dicts) recording each relaxed pose's final Rosetta score.
  3. `_superimposition` (existing, unchanged) — now operates on relaxed
     structures when FastRelax ran.
  4. `_proteinRMSD`, `_ligandRMSD` (existing, unchanged).

GeometricFiltering (existing, unchanged)
  - Produces `structural_features_final.pkl`: one row per
    `(Entry, docked_structure)` pose that survived `LigandRMSD` (NOT
    reduced to one row per entry — see "Ground-Truth Corrections"), with
    geometric/pocket/SASA/PLIP metrics attached, plus the `is_best` /
    `best_method` columns carried over from `LigandRMSD`.

[NEW] PLACERValidation, if run_placer=True
  - Consumes `structural_features_final.pkl`.
  - Reduces to one structure per entry using `is_best`/`best_method` (see
    Component: PLACER Step, "Selecting one structure per entry").
  - Runs PLACER once per entry on that single selected structure.
  - Merges `placer_prmsd`, `placer_confidence`, `placer_dir` columns
    back onto the same final DataFrame.
  - Overwrites `structural_features_final.pkl` with the merged result.
```

FastRelax runs inside `Superimposition`, after file-prep and before the
actual pairwise superimposition, because that is the first (and only)
pipeline stage where every docking engine — including Vina — has already
produced a single-file protein+ligand complex PDB, which FastRelax requires
as its input.

PLACER runs strictly after GeometricFiltering, consuming its output file
directly. Unlike originally assumed, that file is not already reduced to a
single structure per entry, so PLACER's own step performs that reduction
before running — the only point in the pipeline where PLACER's expensive
per-entry cost is bounded to exactly one structure per entry.

## Component: FastRelax Step

**File:** `filterzyme/steps/fastrelax_step.py` (new)
**Class:** `FastRelax(Step)`

Unlike the other per-engine prepare steps, `FastRelax` takes the three
`*_files_for_superimposition` list-columns directly (its input is the output
of `_prepare_files_for_superimposition`, not a single `input_col`), because
ranking/selection is inherently cross-column (it needs to see Chai, Boltz,
and Vina poses for the same entry together to apply per-engine top-K
selection and rewrite each engine's list in place).

### Constructor Parameters

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `entry_col` | `str` | `"Entry"` | Entry identifier column |
| `chai_files_col` | `str` | `"chai_files_for_superimposition"` | Column with list of Chai complex PDB paths |
| `boltz_files_col` | `str` | `"boltz_files_for_superimposition"` | Column with list of Boltz complex PDB paths |
| `vina_files_col` | `str` | `"vina_files_for_superimposition"` | Column with list of Vina complex PDB paths |
| `chai_confidence_col` | `str` | `"chai_ptm"` | Dict-valued column (keyed by `{Entry}_{idx}`) used to rank Chai poses, descending |
| `boltz_confidence_col` | `str` | `"boltz2_confidence_score"` | Dict-valued column (keyed by `{Entry}_model_{idx}`) used to rank Boltz poses, descending |
| `vina_confidence_col` | `str` | `"vina_affinities"` | Dict-valued column (keyed by 1-indexed pose number) used to rank Vina poses, **ascending** (more negative = better). Populated by Task 1's `parse_vina_output` wiring. |
| `output_dir` | `str` | required | Directory for relaxed PDB outputs |
| `top_k` | `int` | `2` | Number of top-ranked poses per engine to relax |
| `drop_unrelaxed` | `bool` | `True` | If `True`, lower-ranked unrelaxed poses for a relaxed engine are removed from that engine's `*_files_for_superimposition` list. If `False`, both relaxed and unrelaxed poses are kept side by side (useful for before/after comparison). |
| `mode` | `Literal["ligand_focused", "full"]` | `"ligand_focused"` | Relax scope |
| `shell_radius` | `float` | `8.0` | Å around ligand for the neighborhood residue selector (ligand-focused mode only) |
| `constraint_weight` | `float` | `1.0` | Coordinate-constraint strength on the shell backbone |
| `scorefunction` | `str` | `"ref2015"` | Rosetta scorefunction name |
| `ligand_resname` | `str` | required | Ligand residue name, used to build the neighborhood selector |
| `num_threads` | `int` | `1` | Parallelism, consistent with other steps |

### Ranking structures per engine

Confidence lookups use the per-engine structure-ID conventions established
by the existing docking-metrics code (`extract_docking_metrics_step.py`):

- **Chai**: dict key = `"{Entry}_{idx}"` (e.g. `"Q97WW0_0"`), from
  `chai_ptm`. Higher is better (Chai's own ranking is descending by
  `aggregate_score`/`ptm`).
- **Boltz**: dict key = `"{Entry}_model_{idx}"` (e.g. `"Q97WW0_model_0"`),
  from `boltz2_confidence_score`. Higher is better (Boltz's own writer
  already ranks descending by `confidence_score` when assigning `model_N`
  indices).
- **Vina**: dict key = the 1-indexed pose number as used in
  `{entry}_{i}_vina.pdb` filenames (e.g. `1`), from `vina_affinities`
  (populated by Task 1). Lower (more negative, kcal/mol) is better.

To go from a dict key back to the actual file path in
`chai_files_for_superimposition` / `boltz_files_for_superimposition`, strip
the trailing `_chai`/`_boltz` engine suffix from the file's stem (mirroring
`get_tool_from_structure_name` in `computeligandRMSD_step.py`, applied in
reverse): a Chai path's stem `"{Entry}_{idx}_chai"` becomes confidence key
`"{Entry}_{idx}"`; a Boltz path's stem `"{Entry}_model_{idx}_boltz"` becomes
confidence key `"{Entry}_model_{idx}"`.

**Vina is different and does NOT use suffix-stripping**: a Vina path's stem
is `"{entry_name}_{i}_vina"` (from `PrepareVina`'s
`f"{entry_name}_{i}_vina.pdb"` naming), but `vina_affinities` is keyed by
the **bare integer pose number** `i` (e.g. `1`), not by the string
`"{entry_name}_{i}"` — stripping the `_vina` suffix would produce the wrong
key. Instead, extract `i` as `int(stem.split('_')[-2])`, mirroring the
existing `extract_vina_index` helper in `filterzyme/utils/helpers.py`
(`add_metrics`), which does exactly this to look up `vina_affinities`
elsewhere in the pipeline.

### Behavior

1. Validate required columns exist in the input DataFrame; raise
   `ValueError` with a clear message otherwise (mirrors
   `Squidly._validate_input`).
2. Per entry, per docking engine whose files-column is non-empty for that
   row, rank poses using that engine's confidence dict and sort direction
   (see above), and keep the top `top_k`.
3. For each kept pose, call an internal
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
4. Replace each relaxed pose's original path with its relaxed path in the
   corresponding engine's `*_files_for_superimposition` list, in place. If
   `drop_unrelaxed=True`, remove that engine's non-top-`top_k` paths from
   the list entirely; if `False`, leave them alongside the relaxed paths.
5. Attach a `fastrelax_score` dict column (same per-engine key convention as
   the confidence columns) recording each relaxed pose's final Rosetta
   score.
6. On any per-pose failure (Rosetta exception, missing/corrupt PDB), log the
   error, leave that pose's original (unrelaxed) path in place in the
   files-column, and continue — never abort the whole run. This matches the
   existing per-entry-tolerant pattern used by Vina's missing-AF2 fallback
   (`dock_vina_step.py`).

### Testability

- `_select_top_k` (per-engine ranking/selection logic, including the
  ascending-vs-descending direction switch): pure Python, no Rosetta
  dependency — always-run unit tests.
- Dict-key ↔ file-path matching (stripping `_chai`/`_boltz` suffixes for
  Chai/Boltz; extracting the integer pose number for Vina): pure Python —
  always-run unit tests.
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
| `preparedfiles_dir` | `str` | required | Directory of per-pose complex PDBs, matching the `GeneralGeometricFiltering`/`LigandSASA`/`PLIP` convention: the PDB for a row is reconstructed as `preparedfiles_dir / f"{docked_structure}.pdb"` (see "Ground-Truth Corrections" — `structural_features_final.pkl` has no PDB-path column) |
| `entry_col` | `str` | `"Entry"` | Entry identifier column |
| `structure_col` | `str` | `"docked_structure"` | Column with the structure ID string used to build the PDB filename above |
| `output_dir` | `str` | required | Directory for PLACER outputs |
| `placer_script_path` | `str` | `"/mnt/labs/data/mora/software/PLACER/run_PLACER.py"` | Path to `run_PLACER.py`; validated at construction — raises `FileNotFoundError` with a clear message if missing. Overridable for use outside this lab's shared software directory. No sibling-repo assumption, unlike the old dead code. |
| `placer_conda_env` | `str` | `"placer_env"` | Name of the conda env PLACER's subprocess is run through (its own dedicated env, pinned to `pytorch=2.3.*`/`dgl=2.4.0`, incompatible with the `filterzyme` env) |
| `predict_ligand` | `str` | required | Ligand resname passed to PLACER |
| `nsamples` | `int` | `50` | PLACER sample count |
| `rerank` | `str` | `"prmsd"` | PLACER rerank metric |
| `num_threads` | `int` | `1` | Parallelism |

### Selecting one structure per entry

`structural_features_final.pkl` has one row per `(Entry, docked_structure)`
pose (see "Ground-Truth Corrections"), with `is_best` (bool) and
`best_method` (comma-joined string of selection-method names that picked
this structure, e.g. `"inter_tool_min_per_tool,inter_tool_weighted_avg"`,
or empty string if `is_best=False`) columns already present. Before running
PLACER, a module-level function `_select_one_per_entry(df, entry_col)`
reduces to exactly one row per entry:

1. Filter to `is_best == True` rows. If an entry has zero `is_best` rows
   (shouldn't normally happen, but must not crash), keep all of that
   entry's rows as the candidate pool for step 2 instead (documented
   fallback, not silently dropping the entry).
2. Within each entry's candidate pool, count the number of methods in
   `best_method` (split on `,`) and keep the row(s) with the highest count.
3. If still tied, sort remaining candidates by `docked_structure` string
   ascending and keep the first — a fully deterministic tie-break.

This function is pure Python/pandas (no PLACER/subprocess dependency) and
is unit-tested directly, independent of the `PLACER` step class.

### Behavior

1. Validate `placer_script_path` exists, `preparedfiles_dir` exists, and
   required columns (`entry_col`, `structure_col`, `is_best`, `best_method`)
   are present in the input DataFrame.
2. Call `_select_one_per_entry` to reduce the input to one row per entry.
3. For each selected row, build its PDB path as
   `Path(preparedfiles_dir) / f"{row[structure_col]}.pdb"` (there is no
   PDB-path column to read directly — see "Ground-Truth Corrections").
4. Per entry, count ligand instances in the structure (reuse the existing
   `_count_ligands` logic from `PLACER_forChai_step.py`) to decide single-
   vs multi-ligand mode.
5. Shell out via `subprocess.run` to `run_PLACER.py`, invoked through the
   `placer_conda_env` conda environment (e.g.
   `["conda", "run", "-n", self.placer_conda_env, "python",
   str(self.placer_script_path), "--ifile", ..., "--odir", ..., "--rerank",
   ..., "-n", ..., "--predict_ligand", ...]`), adding `--predict_multi`
   when multi-ligand is detected. On `AssertionError` from multi-mode,
   fall back to single-mode (same fallback logic as the old
   `PLACER_forChai_step.py`).
6. Parse PLACER's output CSV; extract the configured rerank metric
   (`prmsd` by default) and any confidence score.
7. Merge `placer_prmsd`, `placer_confidence`, `placer_dir` back onto the
   (already-reduced-to-one-row-per-entry) DataFrame by `Entry` — one row per
   entry in, one row per entry out; no row explosion.
8. On subprocess failure or missing output for an entry, log and set that
   entry's PLACER columns to `None`; never abort the whole run.

### Testability

- `_select_one_per_entry` (is_best filter, method-count tie-break,
  alphabetical fallback): pure Python/pandas — always-run unit tests.
- `_count_ligands`, CLI-argument construction, and CSV-parse/merge logic:
  pure Python — always-run unit tests.
- The actual `subprocess.run` invocation of `run_PLACER.py`: wrapped in a
  `pytest.mark.skipif(not placer_available())` smoke test (checks
  `placer_script_path` is set and executable), following the
  `test_squidly_step.py` convention.

## Pipeline Wiring

### `Docking` (in `filterzyme/pipeline_v2.py`) — prerequisite change only

`DockingMetrics.execute()` (`filterzyme/steps/extract_docking_metrics_step.py`)
is extended (Task 1) to call the existing `parse_vina_output` and populate a
new `vina_affinities` dict column, matching the existing `chai_ptm`/
`boltz2_confidence_score` dict-column pattern. This requires no new
`Docking`/`Pipeline` constructor parameters — it always runs when `run_vina`
is enabled, since it merely fills in previously-dead parsing code.

### `Superimposition` (in `filterzyme/pipeline_v2.py`)

New constructor parameters (mirroring the existing `run_vina` convention):

- `run_fastrelax: bool = False`
- `fastrelax_mode: Literal["ligand_focused", "full"] = "ligand_focused"`
- `fastrelax_top_k: int = 2`
- `fastrelax_drop_unrelaxed: bool = True`
- `fastrelax_shell_radius: float = 8.0`
- `fastrelax_constraint_weight: float = 1.0`
- `fastrelax_scorefunction: str = "ref2015"`
- `ligand_resname: str = "LIG"` (matches the existing convention set by
  `PrepareChai`/`PrepareBoltz`, which already rename non-standard residues
  to `LIG` during CIF→PDB conversion — see `preparechai_step.py`)

`Superimposition.run()` calls a new `_run_fastrelax(df_prep)` method,
conditional on `run_fastrelax`, inserted between `_prepare_files_for_superimposition`
and `_superimposition`:

```python
def run(self):
    log_section('Superimposition')
    log_subsection('Superimposing docked structures')
    df_prep = self._prepare_files_for_superimposition()
    if self.run_fastrelax:
        log_subsection('Relaxing top-ranked docked structures')
        df_prep = self._run_fastrelax(df_prep)
    df_sup = self._superimposition(df_prep)
    ...
```

### `Pipeline` (in `filterzyme/pipeline_v2.py`)

New constructor parameters:

- `run_fastrelax: bool = False`
- `fastrelax_mode: Literal["ligand_focused", "full"] = "ligand_focused"`
- `fastrelax_top_k: int = 2`
- `fastrelax_drop_unrelaxed: bool = True`
- `fastrelax_shell_radius: float = 8.0`
- `fastrelax_constraint_weight: float = 1.0`
- `fastrelax_scorefunction: str = "ref2015"`
- `ligand_resname: str = "LIG"`
- `run_placer: bool = False`
- `placer_script_path: str = "/mnt/labs/data/mora/software/PLACER/run_PLACER.py"`
- `placer_conda_env: str = "placer_env"`
- `placer_predict_ligand: str = ''`
- `placer_nsamples: int = 50`
- `placer_rerank: str = "prmsd"`

`Pipeline.run()` forwards the `fastrelax_*`/`ligand_resname` parameters to
the `Superimposition(...)` constructor call (alongside the existing
`maxMatches`/`include_vina`/`num_threads` parameters already passed there).

`Pipeline.run()` gains a new step after `gf.run()`: if `run_placer` is
`True`, construct and run a `PLACERValidation` stage (thin wrapper
composing the `PLACER` step, following the same
`Docking`/`Superimposition`/`GeometricFilters` class pattern already used in
`pipeline_v2.py`) over `geometricfiltering/structural_features_final.pkl`,
passing `preparedfiles_dir=Path(self.base_output_dir) / "superimposition" /
"preparedfiles_for_superimposition"` (the same directory
`GeometricFilters` itself reads from, via
`Superimposition._prepare_files_for_superimposition`'s
`preparedfiles_dir`), and overwrite that file with the merged result.

All new `Superimposition`/`Pipeline` constructor parameters are
forwarded exactly as the existing `squidly_*` and
`run_vina`/`alternative_structure_for_vina` parameters are today
(`Pipeline.run()` passes them through to the respective stage classes).
(`Docking` itself gains no new constructor parameters — see "Docking —
prerequisite change only" above.)

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
- **Adding PyRosetta/PLACER to `environment.yml` or `setup.py` as a hard
  dependency.** Both remain optional, path-configured dependencies (see
  "Shared Software Locations" below); they are not added to the base
  `filterzyme` install requirements.
- **General-purpose Vina docking-log parsing beyond what FastRelax needs.**
  Task 1 wires `parse_vina_output` into `DockingMetrics` only far enough to
  produce a `vina_affinities` dict column usable for FastRelax's top-K
  ranking; it does not add new user-facing columns/reports beyond that, and
  does not change Vina docking behavior itself.

## Shared Software Locations

Both external tools are installed once under a shared lab directory
(`/mnt/labs/data/mora/software/`) rather than per-project, to avoid
redundant installs/downloads across projects that use them. The two tools
have different sharing models because of how they're invoked:

### PyRosetta — `/mnt/labs/data/mora/software/RosettaFastRelax/`

FastRelax calls `pyrosetta.init()` **in-process**, inside the same Python
interpreter that runs the filterzyme pipeline (see Component: FastRelax
Step above). PyRosetta ships as compiled binary wheels tied to a specific
Python version/ABI; importing a shared install into an unrelated project's
interpreter via `sys.path` manipulation risks ABI mismatches (import errors
or segfaults) and is not how any existing dependency in this codebase is
handled.

Therefore PyRosetta is **installed directly into the `filterzyme` conda
env** (matching the existing `squidly` CLI precedent in
`filterzyme/steps/squidly_step.py`, which is installed into the same env
and checked via `shutil.which`). `/mnt/labs/data/mora/software/RosettaFastRelax/`
stores the downloaded installer/wheel artifact (PyRosetta's own download,
which requires a free academic license from rosettacommons.org) so that
re-installing PyRosetta into other projects' envs does not re-download it
from the network each time. It does **not** store an installed
site-packages tree that gets imported cross-env.

`FastRelax.__init__` does not take a PyRosetta-location parameter — it
simply does `import pyrosetta` and raises a clear `RuntimeError` (mirroring
`Squidly`'s `shutil.which("squidly")` check) if the import fails, pointing
the user at the shared installer location for re-installation instructions.

### PLACER — `/mnt/labs/data/mora/software/PLACER/`

PLACER is invoked via `subprocess.run(["python", "run_PLACER.py", ...])` —
a separate OS process, isolated from the filterzyme interpreter. This has
no binary-compatibility constraint, so **one shared clone of the PLACER
repository is safe to reuse across projects**.

The PLACER repository (`github.com/baker-laboratory/PLACER`) is cloned
once to `/mnt/labs/data/mora/software/PLACER/`, containing `run_PLACER.py`,
its bundled model weights, and its own dedicated conda env definition
(`envs/placer_env.yml` — pinned to `pytorch=2.3.*`, `dgl=2.4.0`, and other
versions incompatible with the `filterzyme` env's own dependencies, which
is why PLACER runs as a subprocess in a separate env rather than being
imported).

`PLACER.__init__`'s `placer_script_path` parameter defaults to
`/mnt/labs/data/mora/software/PLACER/run_PLACER.py`, remaining overridable
for use outside this specific lab environment. The subprocess command is
invoked through the `placer_env` conda environment (e.g. via
`conda run -n placer_env python <placer_script_path> ...`), not the
`filterzyme` env's interpreter.

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
  `Superimposition`/`Pipeline` kwargs (`run_fastrelax`, `run_placer`, etc.),
  verifying signatures and defaults without executing anything — mirroring
  `test_docking_accepts_squidly_kwargs` / `test_pipeline_accepts_squidly_kwargs`.
