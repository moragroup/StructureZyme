# FastRelax + PLACER inclusion & verification — design

Date: 2026-08-13
Status: Approved (brainstorming)
Branch: `chore/lab-handoff`

## Problem

The lab-handoff GPU smoke proved the docking chain (Squidly → Chai → Boltz →
Vina → docking metrics) plus the default-on downstream analysis steps. Two
optional, default-disabled steps were never exercised on the fresh env and must
be included and tested for the handoff:

- **FastRelax** (`structurezyme/steps/fastrelax_step.py`): relaxes top-K docked
  poses per engine with Rosetta's FastRelax mover. It calls `import pyrosetta`
  **in-process** (`fastrelax_step.py:480`) and its error message says to install
  pyrosetta "into the current env". But pyrosetta is **not** in the structurezyme
  conda env; it lives only in a separate venv at
  `/mnt/labs/data/mora/software/RosettaFastRelax/env` (py3.12, PyRosetta
  quarterly release). Enabling `fastrelax` today therefore raises `RuntimeError`.

- **PLACER** (`structurezyme/steps/PLACER_step.py`): predicts ligand poses via an
  external install. It is **already** subprocess-correct — it shells out to
  `/mnt/labs/data/mora/software/PLACER/env/bin/python run_PLACER.py`. The shared
  install (weights `PLACER_model_1.pt`, py3.10 env) is present. It only needs to
  be **enabled** and fed a `placer_predict_ligand` value.

Both are GPU-relevant and both depend on shared-filesystem installs under
`/mnt/labs/data/mora/software/`, deliberately not in `environment.yml`.

## Goals

1. Make FastRelax actually run from the structurezyme env by reusing the existing
   RosettaFastRelax venv **via subprocess**, mirroring the PLACER pattern (no
   pyrosetta added to the shared env).
2. Wire PLACER into a smoke run (enable + supply inputs); no code change expected.
3. Prove both end-to-end on GPU against the fresh env by extending the existing
   smoke to run the **full DAG** with `fastrelax` + `placer` enabled.
4. Document the two shared installs and their override env vars in getting_started
   / known-issues.

## Non-goals

- Adding pyrosetta or PLACER to `environment.yml` (licensed / heavy / external).
- Refactoring PLACER (already correct).
- Any scientific tuning of relax parameters or PLACER sampling.

## Architecture

### FastRelax subprocess refactor

The pyrosetta surface is isolated to `_relax_one` (`fastrelax_step.py:457-580`).
Its inputs are fully serializable: the prepared PDB path, the ligand resname for
the neighborhood selector, the list of `.params` files (for `-extra_res_fa`),
`mode`, `shell_radius`, `constraint_weight`, `scorefunction`, and an output path.
Its output is `(relaxed_pdb_path, final_score)`. This is a clean process boundary.

Everything else in the step is pure-Python and stays in the main structurezyme
process: ligand registration + `.params` generation (`_register_ligands`),
`_prepare_pdb_for_pose` (resname rewrite + atom-name remap), ranking/selection
(`_rank_and_select`, `_select_top_k`), and path rewriting
(`_apply_relaxed_paths`). Only the actual Rosetta relaxation crosses the boundary.

**New worker script:** `structurezyme/steps/_fastrelax_worker.py`
- Standalone; imports only stdlib + `pyrosetta` (NO `structurezyme` imports, so it
  runs cleanly under the RosettaFastRelax venv's py3.12 interpreter).
- Reads a JSON spec (path passed as argv, spec written by the main process):
  ```json
  {
    "prepared_pdb": "...for_pose.pdb",
    "extra_res_fa": ["/.../params/XXX.params", "..."],
    "mode": "ligand_focused",
    "ligand_resname": "X01",
    "shell_radius": 8.0,
    "constraint_weight": 1.0,
    "scorefunction": "ref2015",
    "out_pdb": "/.../<stem>_relaxed.pdb"
  }
  ```
- Runs `pyrosetta.init(extra_options="-mute all -extra_res_fa ...", silent=True)`,
  builds the movemap/selector/constraints exactly as the current inline code
  (`fastrelax_step.py:504-575`), applies FastRelax, dumps the relaxed PDB, and
  prints `{"relaxed_path": "...", "score": <float>}` as the LAST line of stdout.
- On error: prints a JSON error object and exits non-zero.

**`_relax_one` becomes a subprocess caller** (mirrors PLACER `_build_cmd` /
`subprocess.run`):
- Resolve the venv python: default
  `/mnt/labs/data/mora/software/RosettaFastRelax/env/bin/python`, overridable via
  env var `FASTRELAX_ENV` (points at the env dir; python is `<env>/bin/python`).
  Resolution + existence check happen once (lazily, cached on the instance),
  raising an actionable `FileNotFoundError` (same message style as PLACER
  `PLACER_step.py:200-204`) if missing.
- Locate the worker via `importlib.resources` / `Path(__file__).with_name(...)` so
  it works from an installed package.
- Write the JSON spec to `output_dir` (per pose), run
  `subprocess.run([env_python, worker, spec_json], capture_output=True, text=True,
  check=False)`, parse the last stdout line as JSON, return `(relaxed_path, score)`.
- On non-zero exit or parse failure: raise (execute() already swallows per-pose
  exceptions at `fastrelax_step.py:625-636`, leaving the original path in place).

**pyrosetta init cost:** `init()` is once-per-process. With per-pose subprocesses
it becomes once per pose (~a few seconds each). At `top_k=2` × 3 engines ≈ 6
poses/entry this is acceptable for the smoke and typical runs. A persistent
stdin-loop worker is a future optimization, explicitly out of scope here.

**Behavior preservation:** the `-extra_res_fa` list, the `.for_pose.pdb`
preparation, the ligand-focused vs full movemap logic, the coordinate constraints,
the output filename (`<stem>_relaxed.pdb`), and the returned score are all
identical to the current in-process code — only the interpreter running them
changes.

### PLACER wiring

No code change to `PLACER_step.py` (already subprocess-correct). Work is limited
to enabling + supplying inputs:
- Enable `placer` for the smoke run (default `enabled=False`, `config.py:27`).
- Supply `placer_predict_ligand` (chain-resname-resnum, e.g. `A-HEM-154`), which
  `run_placer` (`step_runners.py:733-762`) requires and otherwise raises on.
- Keep the existing path defaults (`PLACER_step.py:176-177`); they already point
  at the present shared install. Document the override kwargs.

**Smoke ligand string:** the smoke target is CalB / P41365 (same as the docking
smoke). CalB is a serine hydrolase with no heme/cofactor, so the correct
`--predict_ligand` value is not obviously a heme. During implementation the value
will be derived by inspecting the prepared PDB's HETATM records (chain/resname/
resnum of the docked substrate) and asserting a valid ligand exists; if none fits,
fail loudly rather than emit empty PLACER scores. The concrete string is an
implementation detail resolved against the actual smoke PDB, not fixed here.

### Smoke / verification

Extend the existing `tools/smoke/run_phase_c_smoke.py` +
`tools/smoke/run_phase_c_smoke.sbatch` to run the **full DAG** with `fastrelax`
and `placer` enabled (single source of truth; no second smoke). One GPU-partition
job runs the whole pipeline as one process — FastRelax (CPU-bound) runs on the GPU
node's CPUs, PLACER runs on the GPU. Resources unchanged from the current smoke
except any time bump needed for relax + PLACER sampling.

Validation block (PASS/FAIL, rglob the timestamped output subdir like the current
block) asserts:
- **fastrelax**: at least one `*_relaxed.pdb` under `superimposition/fastrelax/`;
  the `fastrelax_score` column is populated for relaxed entries; the worker
  subprocess exited 0 (surfaced via the step, not a silent skip).
- **placer**: `placer_*` score columns populated and a PLACER output CSV present
  under the placer output dir.
- **end-to-end**: `structural_features_final.pkl` still produced, with the relaxed
  poses flowing through the downstream analysis steps (relaxed paths published to
  the prepared-files dir via `_publish_relaxed_to_prepared`, `step_runners.py:51`).

## Error handling

- Missing RosettaFastRelax venv → actionable `FileNotFoundError` at first relax
  attempt (not at import), matching PLACER's install-guard style.
- Per-pose relax failure (worker non-zero / bad JSON) → logged, original path
  retained, run continues (unchanged execute() semantics).
- Missing PLACER install → existing `FileNotFoundError` in `PLACER.__init__`.
- Missing `placer_predict_ligand` → existing raise in `run_placer`.

## Testing

- Unit: worker-spec builder (JSON shape), venv-python resolution + `FASTRELAX_ENV`
  override, stdout-JSON parsing (happy path + malformed → raises). These run
  without pyrosetta (no venv needed) by mocking `subprocess.run`.
- Keep the existing pyrosetta-gated smoke tests
  (`test_fastrelax_*`), now gated on the venv existing rather than in-env
  pyrosetta import; update `_pyrosetta_available()` accordingly.
- Integration: the extended GPU smoke is the end-to-end proof.
- Full suite must stay green (currently 242 passed / 5 skipped / 0 failed).

## Files touched

- `structurezyme/steps/_fastrelax_worker.py` (new)
- `structurezyme/steps/fastrelax_step.py` (`_relax_one` → subprocess; move inline
  pyrosetta body to worker; `_pyrosetta_available` → venv check; add
  `FASTRELAX_ENV`/`fastrelax_env` resolution)
- `tests/test_fastrelax_*.py` (adjust gating; add builder/parse unit tests)
- `tools/smoke/run_phase_c_smoke.py` + `.sbatch` (enable fastrelax+placer, add
  validation)
- `docs/getting_started.md`, `docs/known-issues.md` (document both shared installs
  + `FASTRELAX_ENV` / PLACER paths + enable steps)

## Open items resolved during implementation

- Exact `placer_predict_ligand` string for the CalB smoke (derive from prepared
  PDB HETATM).
- Any smoke SLURM `--time` bump for the added steps.
