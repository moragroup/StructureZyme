# 12. Assumptions and Open Questions

## Assumptions made during this analysis

1. **Filterzyme and StructureZyme refer to the same project.** Repository root is `StructureZyme`; Python package is `filterzyme` (`filterzyme/__init__.py` declares `__version__ = "0.0.6"`); README, `setup.py`, and benchmarks all refer to "filterzyme".
2. **Static analysis only.** No code was executed. All bug claims rest on file reads and reasoning, not on observed runtime failure. All `file:line` citations were verified by direct file read.
3. **The user wants per-section markdown files.** The original plan called for a single `analysis.md`. After the build-mode toggle the user asked to split the write task into one file per section. Final deliverable is 12 markdown files plus `index.md` in `1_analysis-planning/`.
4. **`pipeline_v2.py` is canonical going forward.** The user explicitly agreed with this recommendation. All consolidation guidance in `05_bugs_by_severity.md`, `07_top10_improvements.md`, and `10_roadmap.md` assumes v2 is kept and v1 is deleted.
5. **The example pickle `examples/DEHP-MEHP.pkl` represents a valid happy-path input.** Not inspected (binary). Treated as the canonical smoke-test fixture.
6. **The reader is the codebase owner.** Tone is direct, with concrete fixes rather than soft suggestions.
7. **`enzymetk` and `docko` exist and provide the documented interfaces.** Neither is in this repository, and `setup.py` does not version-pin them. Their existence is taken on faith from the import statements at `pipeline_v2.py:29-31` and `dock_vina_step.py:3`.
8. **No code path outside the two `pipeline.py` files invokes the PLACER scripts.** This was confirmed by grep over the repository at recon time; the three PLACER step files are unreferenced.
9. **The `AS_inference` conda env is required for the v1 Squidly path.** Documented at `predict_catalyticsite_run.py:19`. Not verified at install time anywhere â a missing env is a silent failure.
10. **Hard-coded `/nvme2/helen/...` paths reflect a single-host development environment.** They are not load-bearing for any external user.

## Resolved questions

- **"Which pipeline is canonical, v1 or v2?"** â **v2**, locked in by the user.

## Open questions for the owner

### Q1: Is `docko` actively maintained alongside Filterzyme?

`docko` is the Vina backbone (`dock_vina_step.py:3 from docko.docko import *`). If it is owned by the same team, several issues become easier: the search-box hard-code at `dock_vina_step.py:71-73`, the hard-coded `pH=7.4` at line 69, and the lack of timeout / retry are all fixable upstream. If `docko` is an external dependency outside our control, the workaround is to subclass / wrap.

**Owner action needed:** clarify ownership and pin the version in `setup.py`.

### Q2: Is `enzymetk` a separate repo under our control?

`enzymetk` provides Chai, Boltz, and (in v2) `ActiveSitePred`. The Phase 2 plan to "kill the local Squidly subprocess" depends on `enzymetk.ActiveSitePred` being maintained and willing to accept any patches we need (e.g. exposing `as_threshold` as a kwarg rather than a hard-coded value). If `enzymetk` is external, we may need to mirror or vendor it.

**Owner action needed:** clarify ownership and pin the version in `setup.py`.

### Q3: Target deployment environment?

Several recommendations branch on this:

- **Single-GPU workstation:** multi-GPU rotation (F-1) is YAGNI. `multiprocessing.dummy` is fine. Checkpoint/resume is still essential.
- **SLURM cluster:** Phase 4's multi-GPU pool is undersized; we should jump straight to Dask/Ray with one job per entry, and `submit_pipeline.sh` should be replaced by `dask-jobqueue` or an equivalent.
- **Cloud (AWS/GCP) batch:** different again; needs container packaging and managed object-store IO.

**Owner action needed:** name the target. Recommend "single workstation now, SLURM later" as the default if no preference.

### Q4: PLACER â keep or drop?

Three PLACER step files exist and none is wired. Keeping them is a maintenance cost (they all declare `class PLACER`, so importing two together is broken); dropping them loses a documented intent to add a Rosetta-flavoured re-scorer.

**Owner action needed:** decide. Recommendation: drop, reintroduce later from a clean implementation if the use case is concrete.

### Q5: Boltz-2 confidence-key stability?

`extract_docking_metrics_step.py:230-242` hard-codes Boltz JSON keys (`affinity_pred_value`, `affinity_pred_value1`, `affinity_pred_value2`, etc.). These keys have changed between Boltz releases. Is there a contract with the Boltz / enzymetk authors about key stability?

**Owner action needed:** if no contract, add a schema-validation step that fails loud rather than silently producing NaN.

### Q6: Should `submit_pipeline.sh` survive at all?

The script references a missing `run_pipeline_on_multiple_substrates.py`. The recommended replacement is a proper CLI (`filterzyme run`, Phase 4). Should the SLURM wrapper survive as a shell script around the CLI, or should we move to `dask-jobqueue` / Ray directly?

**Owner action needed:** depends on Q3.

### Q7: Reporting / downstream consumers?

Are there downstream tools (notebooks, dashboards, wet-lab handoff scripts) that depend on the schema of `structural_features_final.pkl`? Renaming columns during consolidation (e.g. the cofactor unification in F-2) could break them.

**Owner action needed:** list downstream consumers; pin a schema version into the output pickle.

### Q8: License of the Squidly weights under `filterzyme/squidly_final_models/`?

The weights are committed to the repository. If they have a license other than the repo's LICENSE file, that needs to be called out (and possibly the weights should not be in source control at all â move to a separate release tarball).

**Owner action needed:** check.

### Q9: Public PyPI presence?

`setup.py` is structured as if for PyPI publication, but the classifiers are wrong (Python 3.6/3.7/3.8), `python_requires` is typo'd, and no `long_description_content_type` is set. Should we publish to PyPI? If yes, Phase 0 cleanup is mandatory before the first release.

**Owner action needed:** confirm publication intent.

### Q10: Reproducibility expectations?

Are runs expected to be bit-identical given the same input + config + seed? Chai-1 and Boltz-2 are stochastic. The reproducibility manifest in F-8 captures everything we need to reproduce a run "up to upstream-tool stochasticity" but not bit-identically. Is that good enough?

**Owner action needed:** name the bar.

## Next steps (suggested)

1. Owner reviews this directory and resolves Q1â4 (they gate Phase 2 + 3 design).
2. Engineer executes Phase 0 (Â½â1 day). No design decisions required.
3. Phase 1 begins after Q4 (PLACER fate) is decided.
4. Owner schedules Phase 4 design review after Q3 (deployment target) is decided.
