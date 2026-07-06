# PLACER + Rosetta FastRelax Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two opt-in pipeline stages to `filterzyme.pipeline_v2` — Rosetta
FastRelax (refines docked poses) inside `Superimposition`, and PLACER
(re-scores one structure per entry) as a new final stage after
`GeometricFilters` — plus a prerequisite fix that wires Vina's
already-existing-but-unused `parse_vina_output` into `DockingMetrics` so
FastRelax can rank Vina poses the same way it ranks Chai/Boltz poses.

**Architecture:** `DockingMetrics` gains a `vina_affinities` dict column
(Task 1). A new `FastRelax(Step)` class (`filterzyme/steps/fastrelax_step.py`)
ranks each docking engine's poses by that engine's own confidence dict, uses
PyRosetta's `FastRelax` mover on the top-K poses per engine per entry, and
rewrites the `*_files_for_superimposition` list-columns in place. It is
wired into `Superimposition.run()` between `_prepare_files_for_superimposition`
and `_superimposition`. A new `PLACER(Step)` class (rewriting
`filterzyme/steps/PLACER_step.py`, replacing 3 dead/colliding files) reduces
`structural_features_final.pkl` to one row per entry via a new
`_select_one_per_entry` function, then shells out to PLACER's
`run_PLACER.py` per entry through a dedicated `placer_env` conda
environment. It is wired into `Pipeline.run()` as a new step after
`GeometricFilters`, overwriting `structural_features_final.pkl` with the
merged result.

**Tech Stack:** Python 3.11, pandas, PyRosetta (in-process Python API),
PLACER (external subprocess via dedicated conda env), pytest.

## Global Constraints

- Both new stages are opt-in, flags default to `False`
  (`run_fastrelax`, `run_placer`), mirroring the existing `run_vina`
  convention.
- FastRelax and PLACER must each fail gracefully per-entry/per-pose (log +
  continue), never abort the whole pipeline run — mirrors the existing
  `dock_vina_step.py` per-entry-tolerant pattern.
- `FastRelax.__init__` does not accept a PyRosetta-location parameter; it
  raises a clear `RuntimeError` at `execute()` time if `import pyrosetta`
  fails, mirroring `Squidly`'s `shutil.which("squidly")` check.
- `PLACER.__init__`'s `placer_script_path` defaults to
  `/mnt/labs/data/mora/software/PLACER/run_PLACER.py` and is validated
  (raises `FileNotFoundError`) at construction time.
- PLACER's subprocess is invoked through `conda run -n {placer_conda_env}`
  (default `"placer_env"`), never the `filterzyme` interpreter directly.
- No checkpoint/resume system — out of scope for this work, per the spec.
- Every new function/method is written test-first (red → green → refactor),
  per `superpowers:test-driven-development`. Pure-logic tests always run;
  PyRosetta/PLACER-binary-dependent tests are wrapped in
  `pytest.mark.skipif`, following `tests/test_squidly_step.py`'s
  `_squidly_cli_available()` / `_gpu_available()` pattern exactly.
- Conda env for running tests: `filterzyme`
  (`source /mnt/nfs/vol8t/software/software/miniforge/25.9.1/etc/profile.d/conda.sh && conda activate filterzyme`).
  Test command: `python -m pytest tests/ test_pipeline.py -q` (baseline: 22
  passed, 2 skipped).
- Full design spec (with "Ground-Truth Corrections" for context on data
  shapes): `docs/superpowers/specs/2026-07-01-placer-fastrelax-design.md`.

---

## Task 1: Wire `parse_vina_output` into `DockingMetrics` (prerequisite)

**Files:**
- `filterzyme/steps/extract_docking_metrics_step.py` (edit)
- `tests/test_extract_docking_metrics_step.py` (new)

**Context:** `parse_vina_output(file_path)` already exists
(`extract_docking_metrics_step.py:17-43`) and correctly parses Vina's
`mode | affinity` log table into `{mode_int: affinity_float}`. It is never
called. `DockingMetrics.__execute` (`extract_docking_metrics_step.py:116-233`)
builds one `row_result` dict per input row inside a `for _, row in
df.iterrows()` loop, already populating dict-valued columns for Chai/Boltz
(e.g. `ptm_dict[fname] = ...` then `row_result.update({'chai_ptm': ptm_dict,
...})`). Task 1 adds the same pattern for Vina.

Vina's log file is written by `dock_vina()` in
`/mnt/storage01/home/lherrmann/docko_lab_LCH/docko/vina.py:115` to
`{output_dir}/{protein_name}-{ligand_name}_log.txt`, where `output_dir` is
the per-entry `label_dir` (`dock_vina_step.py:38,68`:
`label_dir = self.output_dir / label`, passed as `dock()`'s `output_dir`
param), `protein_name` is the entry label, and `ligand_name` is
`substrate_name`. So the full path pattern, relative to a row's existing
`vina_dir` column (set in `Docking._run_vina`,
`pipeline_v2.py:222`/`240`/`253`, which is `self.output_dir / label` i.e.
exactly `label_dir`), is:
`Path(row['vina_dir']) / f"{row['Entry']}-{row['substrate_name']}_log.txt"`.

- [ ] **1.1 (RED):** Write `tests/test_extract_docking_metrics_step.py` with
      a pure-logic unit test for a new module-level helper
      `_vina_log_path(vina_dir, entry, substrate_name) -> Path` that builds
      this exact path (`Path(vina_dir) / f"{entry}-{substrate_name}_log.txt"`).
      Run it — it must fail with `ImportError`/`AttributeError` since the
      helper doesn't exist yet.
- [ ] **1.2 (GREEN):** Add `_vina_log_path` to
      `extract_docking_metrics_step.py` (simple one-liner, no I/O). Run the
      test — it must pass.
- [ ] **1.3 (RED):** Add a test for `parse_vina_output` itself (it has no
      existing test coverage) using a small temp file with a realistic Vina
      log table, e.g.:
      ```
      mode |   affinity | dist from best mode
           | (kcal/mol) | rmsd l.b.| rmsd u.b.
      -----+------------+----------+----------
         1       -6.5          0          0
         2       -6.1        1.2        2.4
      ```
      assert the result is `{1: -6.5, 2: -6.1}`. (This test should already
      pass since the function exists — this step documents/locks its
      behavior before Task 1.4 changes `DockingMetrics` to depend on it.)
- [ ] **1.4 (RED):** Add a test for `DockingMetrics.__execute`/`.execute()`
      constructing a minimal on-disk fixture: a temp dir with a fake
      `{label_dir}/{entry}-{substrate}_log.txt` Vina log file (reuse the
      table from 1.3), a row with `vina_dir` pointing at `label_dir`, and
      (to satisfy the existing Chai/Boltz extraction code without erroring)
      empty `chai_dir`/`boltz_dir` directories that simply yield no
      npz/json files (the existing code already tolerates zero matches —
      confirm by reading `extract_docking_metrics_step.py:129-230`, which
      only iterates `.rglob`/`.glob` results, no error if empty). Assert
      that `output_df.loc[0, 'vina_affinities'] == {1: -6.5, 2: -6.1}`. Run
      it — must fail (`KeyError: 'vina_affinities'` or similar) since the
      column doesn't exist yet.
- [ ] **1.5 (GREEN):** In `DockingMetrics.__execute`
      (`extract_docking_metrics_step.py`), after the existing Boltz-affinity
      block (after line 228, before `row_result.update(boltz2_metrics_per_model)`
      /`results.append(row_result)` at lines 230-231), add a Vina block:
      ```python
      # ---Extract vina docking metrics---
      vina_affinities = {}
      vina_dir_val = row.get('vina_dir')
      if vina_dir_val and pd.notna(vina_dir_val):
          log_path = _vina_log_path(vina_dir_val, entry_name, ligand_name)
          if log_path.exists():
              try:
                  vina_affinities = parse_vina_output(log_path)
              except Exception as e:
                  print(f"Failed to parse vina log {log_path}: {e}")
      row_result['vina_affinities'] = vina_affinities
      ```
      Guard with `row.get('vina_dir')` (not `row['vina_dir']`) because
      `vina_dir` only exists when `run_vina=True`
      (`Docking._run_vina`/`_extract_docking_quality_metrics`,
      `pipeline_v2.py:262`) — `DockingMetrics` must not crash when Vina was
      never run. Run the test from 1.4 — must pass.
- [ ] **1.6 (REFACTOR):** Re-read the new block in context; confirm variable
      names (`entry_name`, `ligand_name` already defined at
      `extract_docking_metrics_step.py:123-124`) match; no behavior change,
      just cleanup if needed.
- [ ] **1.7 (Verify no regression):** Run
      `python -m pytest tests/ test_pipeline.py -q` in the `filterzyme` conda
      env — must still show the pre-existing 22 passed, 2 skipped, plus the
      new tests passing (new count higher, 0 new failures).

**Acceptance criteria for Task 1:** `vina_affinities` column always present
on `DockingMetrics.execute()`'s output (empty dict `{}` when Vina wasn't run
or its log is missing/unparseable), keyed by 1-indexed Vina pose number,
values are `float` affinities (kcal/mol, negative = better). No existing
test regresses.

---

## Task 2: `FastRelax` step — ranking and dict-key matching (pure logic)

**Files:**
- `filterzyme/steps/fastrelax_step.py` (new)
- `tests/test_fastrelax_step.py` (new)

This task covers everything in the FastRelax step that has no PyRosetta
dependency: column validation, per-engine top-K ranking (including the
ascending/descending direction switch), and dict-key <-> file-path matching.
The actual `pyrosetta`-dependent relax call is Task 3.

Reference: spec "Component: FastRelax Step" (`docs/superpowers/specs/2026-07-01-placer-fastrelax-design.md:206-325`),
particularly "Ranking structures per engine" (lines 239-271) for the exact
dict-key conventions, and `filterzyme/utils/helpers.py:245-251`
(`extract_vina_index`, inside `add_metrics`) for the Vina key-extraction
logic to mirror exactly: `int(stem.split('_')[-2])`, NOT suffix-stripping.

- [ ] **2.1 (RED):** Create `tests/test_fastrelax_step.py`. Write
      `test_chai_key_from_path` / `test_boltz_key_from_path` /
      `test_vina_key_from_path`, asserting a new module-level function
      `_confidence_key_from_path(path: str, engine: str) -> str | int`:
      - `_confidence_key_from_path("/x/Q97WW0_0_chai.pdb", "chai") == "Q97WW0_0"`
        (strip trailing `_chai` from stem).
      - `_confidence_key_from_path("/x/Q97WW0_model_0_boltz.pdb", "boltz") == "Q97WW0_model_0"`
        (strip trailing `_boltz` from stem).
      - `_confidence_key_from_path("/x/Q97WW0_1_vina.pdb", "vina") == 1` (int,
        via `int(stem.split('_')[-2])` — mirrors `extract_vina_index`).
      Run — must fail (function doesn't exist).
- [ ] **2.2 (GREEN):** Implement `_confidence_key_from_path` in
      `fastrelax_step.py`:
      ```python
      def _confidence_key_from_path(path, engine):
          stem = Path(path).stem
          if engine == "vina":
              return int(stem.split("_")[-2])
          suffix = f"_{engine}"
          if stem.endswith(suffix):
              return stem[: -len(suffix)]
          return stem
      ```
      Run tests from 2.1 — must pass.
- [ ] **2.3 (RED):** Write `test_select_top_k_descending` /
      `test_select_top_k_ascending` for a new module-level function
      `_select_top_k(paths: list[str], confidence: dict, engine: str, top_k: int) -> list[str]`:
      - Descending case (`engine="chai"`): given paths
        `["/x/E_0_chai.pdb", "/x/E_1_chai.pdb", "/x/E_2_chai.pdb"]` and
        `confidence = {"E_0": 0.5, "E_1": 0.9, "E_2": 0.7}`, `top_k=2` ->
        returns `["/x/E_1_chai.pdb", "/x/E_2_chai.pdb"]` (highest first, or
        at least both are present — assert as a set of len 2 containing
        the two highest).
      - Ascending case (`engine="vina"`): given
        `["/x/E_1_vina.pdb", "/x/E_2_vina.pdb", "/x/E_3_vina.pdb"]` and
        `confidence = {1: -4.0, 2: -8.0, 3: -6.0}`, `top_k=2` -> the two
        most negative (`E_2`, `E_3`) are selected.
      - Missing-key case: a path whose key is absent from `confidence` is
        excluded from ranking entirely (treated as unrankable, not
        artificially last) — assert it's dropped when `top_k` >= remaining
        count, i.e. it never appears in the returned list.
      Run — must fail.
- [ ] **2.4 (GREEN):** Implement `_select_top_k`:
      ```python
      def _select_top_k(paths, confidence, engine, top_k):
          scored = []
          for p in paths:
              key = _confidence_key_from_path(p, engine)
              if key in confidence:
                  scored.append((p, confidence[key]))
          reverse = engine != "vina"  # descending for chai/boltz, ascending for vina
          scored.sort(key=lambda t: t[1], reverse=reverse)
          return [p for p, _ in scored[:top_k]]
      ```
      Run tests — must pass.
- [ ] **2.5 (RED):** Write `test_validate_columns_raises_on_missing` for a
      new `FastRelax._validate_input(df)` method (mirrors
      `Squidly._validate_input`, `squidly_step.py:123-145`): construct
      `FastRelax(output_dir=..., ligand_resname="LIG")` with default column
      names, call `_validate_input` on a `pd.DataFrame` missing
      `chai_files_for_superimposition` — assert `pytest.raises(ValueError,
      match="missing required columns")`.
- [ ] **2.6 (GREEN):** Implement `FastRelax.__init__` (all constructor
      params per the spec table, `docs/.../2026-07-01-placer-fastrelax-design.md:220-237`)
      and `_validate_input`, checking `entry_col`, `chai_files_col`,
      `boltz_files_col`, `vina_files_col` are present (confidence-dict
      columns are optional per-row — absence just means that engine's poses
      are unrankable and skipped, not a hard error, matching "no
      `vina_affinities`" for `run_vina=False` runs). Run test — must pass.
- [ ] **2.7 (RED):** Write `test_rank_and_select_all_engines_per_row` for a
      new `FastRelax._rank_and_select(row) -> dict[str, list[str]]` method
      returning `{"chai": [...top-k chai paths...], "boltz": [...],
      "vina": [...]}`, given a `pd.Series`-like row with all three
      `*_files_for_superimposition` list columns and all three confidence
      dict columns populated realistically (small example, `top_k=1` for
      brevity). Assert the correct single top path per engine.
- [ ] **2.8 (GREEN):** Implement `_rank_and_select` by calling
      `_select_top_k` once per engine (using `self.chai_confidence_col`
      etc., defaulting to empty list when a files-column is empty/NaN for
      that row — use the existing `valid_file_list` helper from
      `filterzyme/utils/helpers.py:541-547` to guard, mirroring
      `Superimposition._superimposition`'s
      `df[df['vina_files_for_superimposition'].apply(valid_file_list)]`
      pattern at `pipeline_v2.py:313-314`).
      Run test — must pass.
- [ ] **2.9 (RED):** Write `test_drop_unrelaxed_true` /
      `test_drop_unrelaxed_false` for a new
      `FastRelax._apply_relaxed_paths(files: list[str], selected: list[str], relaxed_map: dict[str, str], drop_unrelaxed: bool) -> list[str]`
      pure function: given original `files` list, the `selected` subset that
      was relaxed, and a `relaxed_map` (`{original_path: relaxed_path}`):
      - `drop_unrelaxed=True`: result contains only the relaxed paths (for
        `selected` entries) — non-selected original paths for that engine
        are removed.
      - `drop_unrelaxed=False`: result contains the relaxed paths for
        `selected` entries PLUS the untouched original paths for
        non-selected entries (order doesn't matter — assert as sets).
      Run — must fail.
- [ ] **2.10 (GREEN):** Implement `_apply_relaxed_paths`:
      ```python
      def _apply_relaxed_paths(files, selected, relaxed_map, drop_unrelaxed):
          selected_set = set(selected)
          out = []
          for f in files:
              if f in selected_set:
                  out.append(relaxed_map.get(f, f))
              elif not drop_unrelaxed:
                  out.append(f)
          return out
      ```
      Run tests — must pass.
- [ ] **2.11 (Verify no regression):** Run
      `python -m pytest tests/ test_pipeline.py -q` — all prior tests still
      pass; new Task 2 tests pass.

**Acceptance criteria for Task 2:** All pure-logic ranking/selection/dict-key
functions are implemented and unit-tested with zero PyRosetta dependency;
`FastRelax.__init__`/`_validate_input` exist and raise clear errors on
missing columns.

---

## Task 3: `FastRelax` step — PyRosetta relax logic + `execute()`

**Files:**
- `filterzyme/steps/fastrelax_step.py` (edit)
- `tests/test_fastrelax_step.py` (edit)

This task adds the actual Rosetta `.apply()` call (`_relax_one`) and wires
everything from Task 2 into `FastRelax.execute(df) -> pd.DataFrame`.
PyRosetta-dependent tests are skipped when `pyrosetta` isn't importable,
mirroring `_squidly_cli_available()` (`tests/test_squidly_step.py:176-191`).

- [ ] **3.1 (RED, no pyrosetta needed):** Write
      `test_execute_missing_column_raises` — `FastRelax(...).execute(df)`
      on a DataFrame missing a required column raises `ValueError` (reuses
      `_validate_input` from Task 2). Must fail first (execute() doesn't
      exist yet as a full method — only stub/partial).
- [ ] **3.2 (GREEN):** Implement the top of `FastRelax.execute`: call
      `self._validate_input(df)`, then iterate rows, call
      `self._rank_and_select(row)` per row (Task 2), collecting a
      `per_row_selected: dict[engine, list[str]]`. Stub `_relax_one` to
      `raise NotImplementedError` for now so the test from 3.1 passes
      without requiring pyrosetta.
- [ ] **3.3 (pyrosetta-gated, RED):** Add a `_pyrosetta_available()` helper
      (mirrors `_squidly_cli_available`):
      ```python
      def _pyrosetta_available() -> bool:
          try:
              import pyrosetta  # noqa: F401
              return True
          except Exception:
              return False
      ```
      Write `test_relax_one_smoke`, gated by
      `@pytest.mark.skipif(not _pyrosetta_available(), reason=...)`, that
      builds a tiny synthetic single-residue-plus-ligand PDB fixture (or
      reuses any existing small test PDB fixture in the repo if one exists
      — check `tests/` and `test_data`-like directories first via
      `glob`/`find` before writing a new one), calls
      `FastRelax(..., ligand_resname=<its ligand resname>)._relax_one(path)`,
      and asserts it returns `(relaxed_path, score)` where `relaxed_path`
      exists on disk and `score` is a `float`. This test will not run in
      this environment (pyrosetta not installed in `filterzyme` env yet —
      confirmed via `ModuleNotFoundError` in prior research) but must be
      present and correctly skipped, not erroring, when run.
- [ ] **3.4 (GREEN):** Implement `_relax_one(self, pdb_path) -> tuple[str, float]`
      per spec Behavior steps 3 (`docs/.../2026-07-01-placer-fastrelax-design.md:281-297`):
      - `import pyrosetta` lazily inside the method (mirrors Squidly's lazy
        `enzymetk` import, `squidly_step.py:172`); raise
        `RuntimeError("PyRosetta is required for FastRelax; ...")` if the
        import fails (do NOT let a bare `ImportError` propagate — catch and
        re-raise with the clearer message, matching the spec's "Shared
        Software Locations" section, `docs/.../2026-07-01-placer-fastrelax-design.md:547-550`).
      - Call `pyrosetta.init(silent=True)` guarded by a module-level
        `_pyrosetta_initialized` flag so it's only called once per process
        (PyRosetta's `init()` is not safely re-entrant across repeated
        calls in some versions).
      - Build `pose = pyrosetta.pose_from_pdb(str(pdb_path))`.
      - If `self.mode == "ligand_focused"`: build a
        `ResidueNameSelector` for `self.ligand_resname`, wrap in
        `NeighborhoodResidueSelector(selector, self.shell_radius,
        include_focus_in_subset=True)`; build a `MoveMapFactory`, set
        backbone/chi flexibility true only within the selector, false
        elsewhere; add per-residue `CoordinateConstraintGenerator`
        (weight=`self.constraint_weight`) restricted to the same shell via
        `AddConstraints`.
      - If `self.mode == "full"`: build a `MoveMap` with `set_bb(True)`,
        `set_chi(True)` for all residues; no constraints.
      - `scorefxn = pyrosetta.create_score_function(self.scorefunction)`;
        if `mode == "ligand_focused"`, also enable the
        `coordinate_constraint` score term at `self.constraint_weight`.
      - `relax = pyrosetta.rosetta.protocols.relax.FastRelax(scorefxn)`;
        `relax.set_movemap(movemap)` (or `set_movemap_factory` for the
        `MoveMapFactory` case); `relax.apply(pose)`.
      - `relaxed_path = self.output_dir / f"{Path(pdb_path).stem}_relaxed.pdb"`;
        `pose.dump_pdb(str(relaxed_path))`.
      - Return `(str(relaxed_path), scorefxn(pose))`.
      - Wrap the whole method body (after the lazy-import check) in
        `try/except Exception as e:` that logs and re-raises (so
        `execute()`'s per-pose try/except in 3.5 is the single place that
        swallows failures, keeping `_relax_one` itself a clean, testable
        unit that raises on error rather than silently returning `None`).
- [ ] **3.5 (GREEN):** Complete `FastRelax.execute`: for each row, for each
      engine's selected top-K paths (from 3.2's `per_row_selected`), call
      `self._relax_one(path)` inside `try/except Exception as e:` — on
      success, record `(path -> relaxed_path)` in `relaxed_map` and
      `(confidence_key -> score)` in a new `fastrelax_score` dict for that
      engine; on failure, `logger.error(...)` and continue (leave that
      pose's original path in place per spec Behavior step 6,
      `docs/.../2026-07-01-placer-fastrelax-design.md:305-309`). After
      processing all engines for a row, call `_apply_relaxed_paths` (Task
      2.10) to rewrite each of the three `*_files_for_superimposition`
      columns for that row in place, and attach the row's
      `fastrelax_score` dict. Return the full DataFrame with these columns
      updated.
- [ ] **3.6 (RED, no pyrosetta):** Write
      `test_execute_relax_failure_keeps_original_path` — monkeypatch
      `FastRelax._relax_one` (via `monkeypatch.setattr`) to raise an
      exception unconditionally; run `execute()` on a small DataFrame with
      real-looking (but not necessarily existing) file path strings and
      confidence dicts; assert the output DataFrame's
      `chai_files_for_superimposition` for that row is unchanged (original
      paths preserved, not dropped/replaced with `None`), and no exception
      propagates out of `execute()`. This exercises the per-pose
      try/except from 3.5 without needing pyrosetta.
- [ ] **3.7 (GREEN if needed):** Fix any bugs surfaced by 3.6 (e.g. ensure
      `relaxed_map` defaults correctly so `_apply_relaxed_paths` falls back
      to the original path when relaxation failed).
- [ ] **3.8 (Verify no regression):** Run
      `python -m pytest tests/ test_pipeline.py -q` in the `filterzyme` env
      — all tests pass or are correctly skipped (pyrosetta-gated ones skip
      with a clear reason string, verified by running with `-rs` to show
      skip reasons: `python -m pytest tests/test_fastrelax_step.py -rs`).

**Acceptance criteria for Task 3:** `FastRelax.execute(df) -> pd.DataFrame`
is fully implemented per the spec's Behavior section; per-pose failures
never abort the run; pyrosetta-dependent tests are present and correctly
skip in this environment.

---

## Task 4: Wire `FastRelax` into `Superimposition` (`pipeline_v2.py`)

**Files:**
- `filterzyme/pipeline_v2.py` (edit `Superimposition` class, lines 271-350)
- `test_pipeline.py` (edit)

- [ ] **4.1 (RED):** In `test_pipeline.py`, add
      `test_superimposition_accepts_fastrelax_kwargs`, mirroring
      `test_docking_accepts_squidly_kwargs` (`test_pipeline.py:22-32`):
      `inspect.signature(Superimposition.__init__).parameters` must contain
      `run_fastrelax` (default `False`), `fastrelax_mode` (default
      `"ligand_focused"`), `fastrelax_top_k` (default `2`),
      `fastrelax_drop_unrelaxed` (default `True`),
      `fastrelax_shell_radius` (default `8.0`),
      `fastrelax_constraint_weight` (default `1.0`),
      `fastrelax_scorefunction` (default `"ref2015"`),
      `ligand_resname` (default `"LIG"`). Run — must fail (`KeyError` /
      `AssertionError`, params don't exist yet).
- [ ] **4.2 (GREEN):** Add all 8 new parameters to
      `Superimposition.__init__` (`pipeline_v2.py:272-279`), storing each as
      `self.<name> = <name>` following the existing pattern (e.g.
      `self.maxMatches = maxMatches`). Run test — must pass.
- [ ] **4.3 (GREEN scaffold):** Add a method stub
      `Superimposition._run_fastrelax(self, df_prep)` that just
      `return df_prep` (no-op) for now, and a conditional call in `.run()`
      exactly as specified (`docs/.../2026-07-01-placer-fastrelax-design.md:441-450`):
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
      This scaffold is a no-op so it introduces no new behavior yet; it
      exists so 4.4's test can `monkeypatch` a real attribute.
- [ ] **4.4 (RED then GREEN):** Add
      `test_superimposition_run_fastrelax_skips_when_disabled` — construct
      `Superimposition(maxMatches=1000, run_fastrelax=False)`, monkeypatch
      `Superimposition._prepare_files_for_superimposition` /
      `_superimposition` / `_proteinRMSD` / `_ligandRMSD` to lightweight
      stubs, monkeypatch `Superimposition._run_fastrelax` with a
      `unittest.mock.MagicMock`; call `.run()`; assert the mock was NOT
      called. This should already pass against the 4.3 scaffold (the `if
      self.run_fastrelax:` guard already exists) — if it doesn't, fix the
      guard before proceeding.
- [ ] **4.5 (GREEN):** Now write/run the test from 4.4 — must pass (mock
      not called when `run_fastrelax=False`). Add
      `test_superimposition_run_fastrelax_called_when_enabled` (mirror,
      `run_fastrelax=True`, assert mock WAS called with `df_prep`'s
      return value flowing into `_superimposition`). Run — must pass with
      the 4.4 wiring.
- [ ] **4.6 (RED):** Add
      `test_run_fastrelax_constructs_fastrelax_step_correctly` — monkeypatch
      `filterzyme.pipeline_v2.FastRelax` (the class, once imported at
      module level per 4.8) with a `MagicMock` class; call
      `Superimposition(..., run_fastrelax=True, fastrelax_top_k=3,
      ligand_resname="ABC", ...)._run_fastrelax(df_prep)`; assert the mock
      was constructed with `top_k=3`, `ligand_resname="ABC"`, and an
      `output_dir` under `self.output_dir` (e.g.
      `Path(self.output_dir) / "fastrelax"`), and that `.execute(df_prep)`
      was called on the instance, and its return value is what
      `_run_fastrelax` returns. Must fail (`_run_fastrelax` is still the
      4.4 no-op).
- [ ] **4.7 (GREEN):** Implement the real `_run_fastrelax` body:
      ```python
      def _run_fastrelax(self, df_prep):
          fastrelax_dir = Path(self.output_dir) / 'fastrelax'
          fastrelax_dir.mkdir(exist_ok=True, parents=True)
          step = FastRelax(
              output_dir=fastrelax_dir,
              mode=self.fastrelax_mode,
              top_k=self.fastrelax_top_k,
              drop_unrelaxed=self.fastrelax_drop_unrelaxed,
              shell_radius=self.fastrelax_shell_radius,
              constraint_weight=self.fastrelax_constraint_weight,
              scorefunction=self.fastrelax_scorefunction,
              ligand_resname=self.ligand_resname,
              num_threads=self.num_threads,
          )
          return step.execute(df_prep)
      ```
      Run the 4.6 test — must pass.
- [ ] **4.8 (GREEN, import):** Add
      `from filterzyme.steps.fastrelax_step import FastRelax` to
      `pipeline_v2.py`'s import block (near the other step imports,
      `pipeline_v2.py:13-26`).
- [ ] **4.9 (Verify no regression):** Run
      `python -m pytest tests/ test_pipeline.py -q` — all pass/skip
      correctly.

**Acceptance criteria for Task 4:** `Superimposition` accepts all new
`fastrelax_*`/`ligand_resname` kwargs; `.run()` conditionally relaxes
top-ranked poses between file-prep and pairwise superimposition, exactly as
specified; existing behavior with `run_fastrelax=False` (the default) is
byte-for-byte unchanged (`_run_fastrelax` never called).

---

## Task 5: Delete dead PLACER files; scaffold new `PLACER` step

**Files:**
- `filterzyme/steps/PLACER_step.py` (delete existing dead file, then
  recreate with new content)
- `filterzyme/steps/PLACER_forChai_step.py` (delete)
- `filterzyme/steps/PLACER_forVina_step.py` (delete)
- `tests/test_placer_step.py` (new)

All three existing files declare a colliding `class PLACER(Step)`; none is
imported by `pipeline_v2.py`; two are confirmed broken (`PLACER_step.py`
does `from step import Step` without a package prefix; both `PLACER_step.py`
and `PLACER_forVina_step.py` hardcode relative paths like
`"PLACER/run_PLACER.py"` that don't exist in this repo). This task removes
all three and starts a clean `PLACER_step.py`, reusing `_count_ligands` from
the old `PLACER_forChai_step.py` (`filterzyme/steps/PLACER_forChai_step.py:55-70`)
verbatim since it has no bugs.

- [ ] **5.1:** `git rm filterzyme/steps/PLACER_step.py
      filterzyme/steps/PLACER_forChai_step.py
      filterzyme/steps/PLACER_forVina_step.py`. Commit this deletion alone
      (`git commit -m "refactor: remove dead PLACER_* step files"`) so it's
      easy to review/revert independent of the new implementation.
- [ ] **5.2 (RED):** Create `tests/test_placer_step.py` with
      `test_count_ligands_single` / `test_count_ligands_multi` /
      `test_count_ligands_missing_file`, importing
      `from filterzyme.steps.PLACER_step import _count_ligands` (a
      module-level function now, not a method — promote it out of the class
      for standalone testability, unlike the old `PLACER_forChai_step.py`
      where it was `self._count_ligands`). Use a temp PDB text fixture with
      2 `HETATM` ligand residues at different (chain, resseq) to test the
      multi case. Run — must fail (module doesn't exist yet).
- [ ] **5.3 (GREEN):** Create `filterzyme/steps/PLACER_step.py` with the
      module-level `_count_ligands(pdb_path, ligand_resname)` function
      (copied from `PLACER_forChai_step.py:55-70`, adapted to module level)
      and a bare `class PLACER(Step):` with just `__init__` (constructor
      params exactly per spec table,
      `docs/.../2026-07-01-placer-fastrelax-design.md:335-348`):
      `preparedfiles_dir` (required), `entry_col="Entry"`,
      `structure_col="docked_structure"`, `output_dir` (required),
      `placer_script_path="/mnt/labs/data/mora/software/PLACER/run_PLACER.py"`,
      `placer_conda_env="placer_env"`, `predict_ligand` (required),
      `nsamples=50`, `rerank="prmsd"`, `num_threads=1`. Constructor
      validates `Path(placer_script_path).exists()` — raise
      `FileNotFoundError` with a clear message if not (per spec line 343);
      does NOT validate at import time, only at construction, so tests can
      construct `PLACER` with a temp-file stand-in path. Run the 5.2 tests
      — must pass.
- [ ] **5.4 (RED):** Add
      `test_placer_init_raises_on_missing_script` — construct
      `PLACER(preparedfiles_dir=".", output_dir=".", predict_ligand="LIG",
      placer_script_path="/nonexistent/run_PLACER.py")`; assert
      `pytest.raises(FileNotFoundError)`.
- [ ] **5.5 (GREEN if needed):** Confirm 5.3's constructor already satisfies
      5.4 (it should, since the check was added in 5.3) — if not, fix.
- [ ] **5.6 (Verify no regression):** Run
      `python -m pytest tests/ test_pipeline.py -q` — all pass.

**Acceptance criteria for Task 5:** The 3 old dead/colliding PLACER files are
gone from the tree (confirmed via `git status`/`git log` showing the
deletion commit); a new minimal `PLACER` class exists with a validated
constructor and a working, unit-tested `_count_ligands`.

---

## Task 6: `_select_one_per_entry` (PLACER's row-reduction logic)

**Files:**
- `filterzyme/steps/PLACER_step.py` (edit)
- `tests/test_placer_step.py` (edit)

Reference: spec "Selecting one structure per entry"
(`docs/.../2026-07-01-placer-fastrelax-design.md:350-370`). Pure
pandas/Python, no subprocess/PLACER dependency — always-run tests.

`structural_features_final.pkl` has one row per `(Entry, docked_structure)`
with `is_best` (bool) and `best_method` (comma-joined string, e.g.
`"inter_tool_min_per_tool,inter_tool_weighted_avg"` or `""`/`NaN` when
`is_best=False` — see `computeligandRMSD_step.py:522,529-530` for how these
are actually produced: `is_best = best_method.notna()`, i.e. `best_method`
is `NaN`, not empty string, when a structure was never picked. Handle both
`NaN` and `""` defensively in the method-count logic).

- [ ] **6.1 (RED):** Write `test_select_one_per_entry_prefers_is_best`:
      build a small DataFrame with 2 entries, each having 3
      `docked_structure` rows, exactly one `is_best=True` row per entry;
      assert `_select_one_per_entry(df, entry_col="Entry")` returns exactly
      1 row per entry and it's the `is_best=True` row.
- [ ] **6.2 (RED):** Write `test_select_one_per_entry_method_count_tiebreak`:
      one entry with 2 rows both `is_best=True`, `best_method` values
      `"inter_tool_min_per_tool"` (1 method) and
      `"inter_tool_min_per_tool,inter_tool_weighted_avg"` (2 methods);
      assert the 2-method row wins.
- [ ] **6.3 (RED):** Write
      `test_select_one_per_entry_alphabetical_final_tiebreak`: one entry
      with 2 rows, both `is_best=True`, both `best_method` with the same
      method count (e.g. both single-method, different method names or
      same), `docked_structure` values `"Q1_1_chai"` and `"Q1_0_chai"`;
      assert the alphabetically-first (`"Q1_0_chai"`) wins.
- [ ] **6.4 (RED):** Write
      `test_select_one_per_entry_fallback_when_no_is_best`: one entry where
      ALL rows have `is_best=False`; assert the function does not drop the
      entry — it still returns exactly 1 row for it (from the full
      candidate pool per the spec's documented fallback,
      `docs/.../2026-07-01-placer-fastrelax-design.md:360-363`).
      Run 6.1-6.4 — all must fail (`_select_one_per_entry` doesn't exist).
- [ ] **6.5 (GREEN):** Implement `_select_one_per_entry(df, entry_col="Entry")`
      as a module-level function in `PLACER_step.py`:
      ```python
      def _method_count(best_method):
          if pd.isna(best_method) or best_method == "":
              return 0
          return len(str(best_method).split(","))

      def _select_one_per_entry(df, entry_col="Entry"):
          rows = []
          for entry, group in df.groupby(entry_col, sort=False):
              pool = group[group["is_best"] == True]
              if pool.empty:
                  pool = group
              pool = pool.copy()
              pool["_method_count"] = pool["best_method"].apply(_method_count)
              max_count = pool["_method_count"].max()
              pool = pool[pool["_method_count"] == max_count]
              pool = pool.sort_values("docked_structure")
              rows.append(pool.iloc[0])
          result = pd.DataFrame(rows).drop(columns=["_method_count"])
          return result.reset_index(drop=True)
      ```
      Run all 4 tests from 6.1-6.4 — must pass.
- [ ] **6.6 (REFACTOR):** Re-read for clarity; confirm `groupby(...,
      sort=False)` preserves the original entry order in the output (not
      required by the spec but nice for reproducibility) — adjust/add a
      comment if it doesn't matter, no behavior change needed.
- [ ] **6.7 (Verify no regression):** Run
      `python -m pytest tests/ test_pipeline.py -q` — all pass.

**Acceptance criteria for Task 6:** `_select_one_per_entry` reduces any
input DataFrame to exactly one row per unique `entry_col` value, following
the 3-step tie-break (is_best filter -> method count -> alphabetical), with
a documented, tested fallback when an entry has zero `is_best=True` rows.

---

## Task 7: PLACER `execute()` — CLI construction, subprocess, merge

**Files:**
- `filterzyme/steps/PLACER_step.py` (edit)
- `tests/test_placer_step.py` (edit)

Reference: spec "Behavior" (`docs/.../2026-07-01-placer-fastrelax-design.md:372-398`).
CLI-argument construction and CSV-parse/merge are pure-logic, always-run
tested; the actual `subprocess.run` invocation is `pytest.mark.skipif`-gated
on `placer_available()`.

- [ ] **7.1 (RED):** Write `test_build_pdb_path` for a new
      `PLACER._build_pdb_path(row) -> Path` method: given
      `preparedfiles_dir="/x/preparedfiles"`, `structure_col="docked_structure"`,
      and a row with `docked_structure="Q97WW0_1_vina"`, assert result is
      `Path("/x/preparedfiles/Q97WW0_1_vina.pdb")` (mirrors the
      `GeneralGeometricFiltering` convention exactly,
      `geometric_filtering_cofactor_MCS.py:466`:
      `self.preparedfiles_dir / f"{docked_structure_name}.pdb"`).
- [ ] **7.2 (GREEN):** Implement `_build_pdb_path`. Run test — pass.
- [ ] **7.3 (RED):** Write `test_build_placer_cmd_single_ligand` /
      `test_build_placer_cmd_multi_ligand` for a new
      `PLACER._build_cmd(pdb_path, n_ligands) -> list[str]` method:
      - Single (`n_ligands <= 1`): result is
        `["conda", "run", "-n", self.placer_conda_env, "python",
        str(self.placer_script_path), "--ifile", str(pdb_path), "--odir",
        str(self.output_dir), "--rerank", self.rerank, "-n",
        str(self.nsamples), "--predict_ligand", self.predict_ligand]`
        (per spec line 386-388).
      - Multi (`n_ligands > 1`): same list plus `"--predict_multi"`
        appended (per spec line 388-389).
      Run — must fail.
- [ ] **7.4 (GREEN):** Implement `_build_cmd`. Run tests — pass.
- [ ] **7.5 (RED):** Write `test_parse_placer_output_csv` for a new
      module-level function
      `_parse_placer_csv(csv_path, rerank_col="prmsd") -> dict`: given a
      small fixture CSV with columns matching PLACER's real output shape
      (check the PLACER repo's README/example output format if accessible
      via the cloned repo at `/mnt/labs/data/mora/software/PLACER/` — if
      still empty/inaccessible, use a reasonable placeholder schema
      documented with a `# ASSUMPTION:` comment: columns
      `rank,prmsd,confidence,...`, one row per sample, PLACER's own
      `rerank` sorting already applied by PLACER itself so row 0 is the
      best), assert the function returns
      `{"placer_prmsd": <row0 prmsd value>, "placer_confidence": <row0
      confidence value>}`. If the fixture CSV is missing/unreadable, return
      `{"placer_prmsd": None, "placer_confidence": None}` (never raise).
      Run — must fail.
- [ ] **7.6 (GREEN):** Implement `_parse_placer_csv`. Run — pass. Flag the
      `# ASSUMPTION:` comment clearly for a follow-up correction once the
      actual PLACER repo/output format is available for direct inspection
      (this is a known gap — the PLACER software itself is not yet
      installed at `/mnt/labs/data/mora/software/PLACER/`, confirmed empty
      during research).
- [ ] **7.7 (RED, pyrosetta-independent but PLACER-binary-gated):** Add a
      `_placer_available()` helper:
      ```python
      def _placer_available() -> bool:
          import shutil
          script = Path("/mnt/labs/data/mora/software/PLACER/run_PLACER.py")
          return script.exists() and shutil.which("conda") is not None
      ```
      Write `test_execute_smoke`, gated by
      `@pytest.mark.skipif(not _placer_available(), reason=...)`, that runs
      `PLACER(...).execute(df)` end-to-end on a tiny real fixture DataFrame
      and asserts `placer_prmsd`/`placer_confidence`/`placer_dir` columns
      are present in the output, one row per entry. This will be skipped
      in this environment (PLACER not yet installed) but must be present.
- [ ] **7.8 (GREEN):** Implement `PLACER.execute(df) -> pd.DataFrame` tying
      everything together per spec Behavior steps 1-8
      (`docs/.../2026-07-01-placer-fastrelax-design.md:374-398`):
      1. Validate `entry_col`, `structure_col`, `is_best`, `best_method`
         are present in `df` — raise `ValueError` otherwise (mirror
         `Squidly._validate_input`'s error style).
      2. `reduced = _select_one_per_entry(df, self.entry_col)` (Task 6).
      3. For each row in `reduced`: `pdb_path = self._build_pdb_path(row)`;
         skip with a logged warning if it doesn't exist on disk.
      4. `n_ligands = _count_ligands(pdb_path, self.predict_ligand)`.
      5. `cmd = self._build_cmd(pdb_path, n_ligands)`;
         `result = subprocess.run(cmd, capture_output=True, text=True)`.
         On `result.returncode != 0` and `"AssertionError" in
         result.stderr` and `n_ligands > 1`: retry with the single-ligand
         cmd (`self._build_cmd(pdb_path, 1)`), mirroring the old
         `PLACER_forChai_step.py:150-158` fallback logic.
      6. On success, find PLACER's output CSV (glob
         `self.output_dir.glob(f"{pdb_path.stem}*.csv")`, excluding any
         `*_summary.csv`, mirroring the old file's
         `existing_files`/`stem`-matching pattern,
         `PLACER_forChai_step.py:116-117`) and call `_parse_placer_csv`.
      7. Collect `{entry: {"placer_prmsd":..., "placer_confidence":...,
         "placer_dir": str(self.output_dir) if success else None}}` per
         row.
      8. Merge these three new columns onto `reduced` by `self.entry_col`
         (NOT onto the original un-reduced `df` — the spec is explicit that
         PLACER's output is one row per entry, replacing
         `structural_features_final.pkl`'s previous multi-row-per-entry
         shape). Return the merged, reduced DataFrame.
      On subprocess failure for any single entry, log and set that entry's
      3 PLACER columns to `None`; never raise out of `execute()` (spec
      Behavior step 8, `docs/.../2026-07-01-placer-fastrelax-design.md:397-398`).
- [ ] **7.9 (Verify no regression):** Run
      `python -m pytest tests/ test_pipeline.py -q` — all pass/skip
      correctly; use `-rs` to double check skip reasons are informative.

**Acceptance criteria for Task 7:** `PLACER.execute()` is fully implemented;
CLI-argument construction and CSV parsing are pure-logic tested; the
subprocess smoke test is present and correctly skips without PLACER
installed; the returned DataFrame has exactly one row per entry with
`placer_prmsd`/`placer_confidence`/`placer_dir` columns.

---

## Task 8: Wire `run_fastrelax`/`run_placer` into `Pipeline` (`pipeline_v2.py`)

**Files:**
- `filterzyme/pipeline_v2.py` (edit `Pipeline` class, lines 421-495)
- `test_pipeline.py` (edit)

Reference: spec "Pipeline Wiring" (`docs/.../2026-07-01-placer-fastrelax-design.md:453-492`).

- [ ] **8.1 (RED):** Add `test_pipeline_accepts_fastrelax_and_placer_kwargs`
      to `test_pipeline.py`, mirroring `test_pipeline_accepts_squidly_kwargs`
      (`test_pipeline.py:35-43`): assert `inspect.signature(Pipeline.__init__).parameters`
      contains all of: `run_fastrelax` (default `False`), `fastrelax_mode`
      (default `"ligand_focused"`), `fastrelax_top_k` (default `2`),
      `fastrelax_drop_unrelaxed` (default `True`), `fastrelax_shell_radius`
      (default `8.0`), `fastrelax_constraint_weight` (default `1.0`),
      `fastrelax_scorefunction` (default `"ref2015"`), `ligand_resname`
      (default `"LIG"`), `run_placer` (default `False`),
      `placer_script_path` (default
      `"/mnt/labs/data/mora/software/PLACER/run_PLACER.py"`),
      `placer_conda_env` (default `"placer_env"`), `placer_predict_ligand`
      (default `''`), `placer_nsamples` (default `50`), `placer_rerank`
      (default `"prmsd"`). Run — must fail.
- [ ] **8.2 (GREEN):** Add all 14 new parameters to `Pipeline.__init__`
      (`pipeline_v2.py:423-439`), storing each as `self.<name> = <name>`.
      Run test — pass.
- [ ] **8.3 (RED):** Add `test_pipeline_forwards_fastrelax_kwargs_to_superimposition`
      — construct a `Pipeline(df=..., boltz_cache_dir=..., run_fastrelax=True,
      fastrelax_top_k=5, ligand_resname="XYZ", base_output_dir="/tmp/...")`,
      monkeypatch `filterzyme.pipeline_v2.Superimposition` (module-level
      class reference) with a `MagicMock`, monkeypatch `Docking` and
      `GeometricFilters` similarly (so `.run()` doesn't actually execute
      anything), monkeypatch `pd.read_pickle` to return an empty
      DataFrame, call `pipeline.run()`; assert `Superimposition` was
      constructed with `run_fastrelax=True`, `fastrelax_top_k=5`,
      `ligand_resname="XYZ"` among its call kwargs (inspect
      `Superimposition.call_args.kwargs`). Run — must fail (not forwarded
      yet).
- [ ] **8.4 (GREEN):** In `Pipeline.run()`'s `Superimposition(...)`
      construction (`pipeline_v2.py:478-484`), add the 8 fastrelax/ligand
      kwargs:
      ```python
      superimp = Superimposition(
          maxMatches=self.max_matches,
          input_dir=Path(self.base_output_dir) / "docking",
          output_dir=Path(self.base_output_dir) / "superimposition",
          include_vina=self.run_vina,
          num_threads=self.num_threads,
          run_fastrelax=self.run_fastrelax,
          fastrelax_mode=self.fastrelax_mode,
          fastrelax_top_k=self.fastrelax_top_k,
          fastrelax_drop_unrelaxed=self.fastrelax_drop_unrelaxed,
          fastrelax_shell_radius=self.fastrelax_shell_radius,
          fastrelax_constraint_weight=self.fastrelax_constraint_weight,
          fastrelax_scorefunction=self.fastrelax_scorefunction,
          ligand_resname=self.ligand_resname,
      )
      ```
      Run test from 8.3 — pass.
- [ ] **8.5 (RED):** Add `test_pipeline_runs_placer_when_enabled` /
      `test_pipeline_skips_placer_when_disabled` — monkeypatch
      `filterzyme.pipeline_v2.PLACER` (module-level class reference, added
      in 8.7's import) with a `MagicMock`, monkeypatch `Docking`,
      `Superimposition`, `GeometricFilters` to no-ops, monkeypatch
      `pd.read_pickle` to return a small stub DataFrame; construct
      `Pipeline(..., run_placer=True, placer_predict_ligand="LIG")`, call
      `.run()`; assert the `PLACER` mock's `.execute()` was called exactly
      once when `run_placer=True`, and NOT called when `run_placer=False`
      (default). Must fail (PLACER not wired into `Pipeline.run()` yet).
- [ ] **8.6 (GREEN):** In `Pipeline.run()`, after `gf.run()`
      (`pipeline_v2.py:495`), add:
      ```python
      if self.run_placer:
          log_section('PLACER validation')
          placer_input_path = Path(self.base_output_dir) / "geometricfiltering" / "structural_features_final.pkl"
          df_placer_input = pd.read_pickle(placer_input_path)
          placer_dir = Path(self.base_output_dir) / "placer"
          placer_step = PLACER(
              preparedfiles_dir=Path(self.base_output_dir) / "superimposition" / "preparedfiles_for_superimposition",
              output_dir=placer_dir,
              placer_script_path=self.placer_script_path,
              placer_conda_env=self.placer_conda_env,
              predict_ligand=self.placer_predict_ligand,
              nsamples=self.placer_nsamples,
              rerank=self.placer_rerank,
              num_threads=self.num_threads,
          )
          df_placer_out = placer_step.execute(df_placer_input)
          df_placer_out.to_pickle(placer_input_path)
      ```
      matching the spec's exact wiring
      (`docs/.../2026-07-01-placer-fastrelax-design.md:476-485`) — note it
      reads and overwrites the SAME `structural_features_final.pkl` path
      that `GeometricFilters.run()` just wrote
      (`pipeline_v2.py:379-380`: `out_final = Path(self.output_dir) /
      'structural_features_final.pkl'; df_final.to_pickle(out_final)`).
      Confirm `self.output_dir` there equals
      `Path(self.base_output_dir) / "geometricfiltering"` from
      `Pipeline.run()`'s `GeometricFilters(...)` construction
      (`pipeline_v2.py:488-493`), so the paths do match.
      Run tests from 8.5 — pass.
- [ ] **8.7 (GREEN, import):** Add
      `from filterzyme.steps.PLACER_step import PLACER` to `pipeline_v2.py`'s
      import block.
- [ ] **8.8 (Verify no regression):** Run
      `python -m pytest tests/ test_pipeline.py -q` — all pass/skip
      correctly. Also run `python -m pytest test_pipeline.py -v` to
      visually confirm every new signature/forwarding test name appears and
      passes.

**Acceptance criteria for Task 8:** `Pipeline` accepts and correctly
forwards all 14 new kwargs; `run_placer=True` triggers a new PLACER stage
after `GeometricFilters` that reads, augments, and overwrites
`structural_features_final.pkl` with exactly one row per entry; default
behavior (`run_fastrelax=False`, `run_placer=False`) is unchanged from
today's pipeline.

---

## Task 9: Final full-suite verification and cleanup

**Files:** none (verification only)

- [ ] **9.1:** Run the full test suite one more time:
      `python -m pytest tests/ test_pipeline.py -v` in the `filterzyme` conda
      env. Confirm: (a) the original 22 tests still pass, (b) all new tests
      from Tasks 1-8 pass or are correctly skipped with an informative
      reason, (c) `python -m pytest tests/ test_pipeline.py -q` summary line
      shows 0 failures/errors.
- [ ] **9.2:** Run `python -c "from filterzyme.pipeline_v2 import Pipeline,
      Docking, Superimposition, GeometricFilters"` and
      `python -c "from filterzyme.steps.fastrelax_step import FastRelax"`
      and `python -c "from filterzyme.steps.PLACER_step import PLACER"` to
      confirm no import-time errors (e.g. accidental hard `import
      pyrosetta` at module level instead of lazily inside `_relax_one`).
- [ ] **9.3:** `git status` — confirm no stray untracked files, no leftover
      debug prints. `git log --oneline` — review the full commit sequence
      for this branch reads cleanly task-by-task.
- [ ] **9.4:** Update the "Out of Scope / Deferred" section of the spec if
      anything discovered during implementation needs to be flagged for a
      follow-up (e.g. the `# ASSUMPTION:` comment on PLACER's CSV schema
      from Task 7.6, if the real PLACER repo still isn't available to
      verify against by the time this task runs).
- [ ] **9.5:** Present a summary of what was implemented, what remains
      environment-gated (pyrosetta install, PLACER repo clone + `placer_env`
      conda env — both explicitly out of scope for this implementation
      phase per the spec's "Shared Software Locations" section), and hand
      off for review per `superpowers:requesting-code-review`.

**Acceptance criteria for Task 9:** Full test suite green (pass or
correctly-skipped); no import-time regressions; branch history is clean and
reviewable; open follow-ups are explicitly documented rather than silently
dropped.
