# 10. Open-Ended Phased Roadmap

Six phases, each with a definite success criterion. Estimates are calendar days for a single full-time engineer familiar with the codebase. The plan is open-ended: each phase can be paused without leaving the codebase in a broken state.

## Phase 0 â Repo hygiene (~Â½â1 day)

The smallest changes that remove the most embarrassing problems and unblock everything downstream.

**Tasks.**

- Fix the `alternative_strucuture_for_vina` typo at `pipeline.py:151, 154` (P0-1).
- Fix the `ppython_requires` typo at `setup.py:50` and drop the obsolete Python 3.6/3.7/3.8 classifiers at `setup.py:37-39` (P1-4, CQ-10).
- Remove `.DS_Store` files (`filterzyme/squidly_final_models/.DS_Store`) and add `**/.DS_Store` to `.gitignore` (P3-1).
- Fix `test_pipeline.py` imports â either delete the file or rewrite as a pytest smoke test (P0-6).
- Replace hard-coded `/nvme2/helen/...` paths in README, `submit_pipeline.sh`, and `benchmarking/*/run_*.py` with parameterised paths (P1-5).
- Remove the wildcard `from docko.docko import *` at `dock_vina_step.py:3` (CQ-8).
- Strip the dead triple-quoted block in `step.py:21-49` and the duplicate `Step` definition at `step.py:51-64` (CQ-2, P3-4, P3-5).
- Delete (or wire up) the three dead PLACER scripts. Leaving them in the tree is the more dangerous option because of the `class PLACER` name collision risk (P0-5, dead-code section).
- Replace `print()` outside `__main__` blocks with `logger.info` / `logger.warning` (CQ-9).
- Fix the `environment.yml:23` self-reference (P2-9).

**Success criterion.**

```bash
pip install -e .
python -c "import filterzyme; print(filterzyme.__version__)"
```

both succeed on a clean Python 3.11 venv. `git status` shows no `.DS_Store`. `grep -r 'nvme2/helen' .` returns nothing outside notebooks (notebooks handled in Phase 5).

## Phase 1 â Critical bug fixes (~3â5 days)

Fix the bugs that crash on documented inputs and make the consolidation decision real.

**Tasks.**

- Consolidate to `pipeline_v2.py` as the canonical entry point. Port the `skip_catalytic_residue_prediction` flag and the metagenomic-Vina fallback (with the typo fixed) from `pipeline.py`. Delete `pipeline.py` and the local `predict_catalyticsite_step.py` / `predict_catalyticsite_run.py` (P0-4, item 1 of `07_top10_improvements.md`).
- Fix `helpers.py:add_metrics` â wire `dict_columns` and `extract_vina_index`, or replace the function with a call to the working `extract_docking_metrics` at `helpers.py:277` (P0-3).
- Settle the `_proteinRMSD` / `_ligandRMSD` return shape on the 2-tuple form from v2 (P0-2).
- Update README, `benchmarking/martinez/run_martinez.py:38`, and `benchmarking/serine_hydrolases/run_serine_hydrolases.py:36` to import from the consolidated pipeline.
- Add the input-DataFrame validator (item 5 of `07_top10_improvements.md`, F-2 partial).
- Move duplicated helpers into `utils/` (item 2 of `07_top10_improvements.md`).
- Fix `int(r) + 1` crash on empty residue strings at `dock_vina_step.py:36` (P1-3).
- Replace the brittle `Path(self.output_dir).parent / 'docking/dockingmetrics.pkl'` with an explicit constructor argument (P1-6) â this work disappears with the consolidation.
- Add `tests/test_pipeline_smoke.py` running end-to-end on `examples/DEHP-MEHP.pkl` with all upstream tools mocked.

**Success criterion.** End-to-end run on `examples/DEHP-MEHP.pkl` completes without error and produces `structural_features_final.pkl`. `pytest -q` is green.

## Phase 2 â Model-integration cleanup (~1 week)

Remove the dual Squidly path, make upstream-tool boundaries clean, decide PLACER's fate.

**Tasks.**

- Replace the local Squidly subprocess with the in-process `enzymetk.ActiveSitePred` everywhere (item 3 of `07_top10_improvements.md`, CQ-13).
- Make AF2 fetch cacheable (item 9 of `07_top10_improvements.md`, F-9 partial).
- Unify cofactor handling (F-2 main).
- Decide PLACER's fate. If kept: pick one of the three PLACER scripts, rename the others to `PLACER_chai` / `PLACER_vina` to break the class collision, wire into the Docking stage. If dropped: delete all three.
- Fix the `Vina.__execute` name mangling at `dock_vina_step.py:28` (CQ-3, P1-7).
- Make `Vina` inherit from the local `Step`, not `enzymetk.step.Step` (CQ-7).
- Add a basic integration test that exercises the AF2-fallback path (CQ-15).

**Success criterion.** All advertised models are reachable from one Python process. The model-integration audit (`04_model_integration_audit.md`) goes from 8 partials / 1 dead to 9 working.

## Phase 3 â New model backends (~1â2 weeks)

Add the recommended models from `08_recommended_models.md`. Each addition is one config-flag toggle.

**Tasks.**

- Add **GNINA** as a docking-engine option. Add `dock_gnina_step.py`. Extend `DockingMetrics` to parse GNINA scores. (~2 days)
- Add **Smina** as a docking-engine option. (~1 day; mostly copy-paste from Vina.)
- Add **DiffDock** as a docking-engine option behind an optional install extra. (~4 days)
- Add **ESMFold** as a tertiary AF2-fallback step. (~2 days)
- Add **AlphaFold3** (OpenFold reimplementation) as a structure-prediction option alongside Chai and Boltz. (~5 days, mostly weight-licensing and install)

**Success criterion.** User can set `docking_engine: [vina | gnina | smina | diffdock]` and `structure_engines: [chai, boltz, alphafold3]` via the YAML config (Phase 4 ships the config; this phase ships the backends).

## Phase 4 â Productionisation (~1â2 weeks)

Make the pipeline operable, configurable, and resilient.

**Tasks.**

- YAML/JSON config schema with Pydantic (item 4 of `07_top10_improvements.md`, F-3).
- CLI entry point `filterzyme run ...` (F-7).
- Checkpoint/resume with stage hashing (item 8 of `07_top10_improvements.md`, F-4).
- Per-entry error CSV (F-5).
- `ProcessPoolExecutor` for CPU-bound stages (item 10 of `07_top10_improvements.md`).
- Multi-GPU worker pool (F-1).
- Reproducibility manifest (F-8).
- Rotating-file structured logging (item 7 of `07_top10_improvements.md`, F-6).
- GitHub Actions CI: lint (`ruff`) + `pytest` (CQ-11).
- Pre-commit hooks: `ruff`, `mypy --warn-unused-ignores` on `utils/helpers.py` initially.

**Success criterion.** Reproducible runs from config. A killed run resumes correctly. `pytest -q` is green on every push to `main`.

## Phase 5 â Quality and documentation (~3â5 days)

Make the project pleasant to adopt.

**Tasks.**

- Docstrings on all public APIs (`Pipeline`, `Docking`, `Superimposition`, `GeometricFilters`, plus each `Step` subclass `execute`).
- Replace `""" Execute some shit """` placeholders (`step.py:54`, P3-3).
- API reference: `mkdocs-material` + `mkdocstrings` (or Sphinx).
- Tutorial notebook against `examples/DEHP-MEHP.pkl` with explanatory prose at each stage.
- Update README to reflect the consolidated pipeline, the CLI, and the config.
- Replace hard-coded paths in `tests/visualize_results.ipynb` (lines 233â339) and the benchmarking notebooks with `pathlib.Path(__file__).parent / ...` constructs.
- Optional: deploy docs to GitHub Pages via the CI workflow.

**Success criterion.** README â tutorial â API docs render cleanly and link to each other. A new user can install, run, and interpret the example in under 30 minutes.

## Roadmap dependency graph

```
Phase 0 (hygiene)
  â
Phase 1 (critical fixes + consolidation)
  â
Phase 2 (model integration cleanup)
  â                â
Phase 3 (new models)  Phase 4 (productionisation)
                  â
                  Phase 5 (docs)
```

Phase 3 and Phase 4 can run in parallel after Phase 2.
