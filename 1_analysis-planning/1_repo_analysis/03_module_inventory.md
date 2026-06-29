# 3. Module / Component Inventory

Status legend:

- **working** â code path appears to execute on the happy path
- **partial** â works but has known bugs, missing validation, or API mismatches between v1 and v2
- **dead** â defined but never imported by either pipeline
- **broken** â known to raise on the documented call path

## Top-level

| File | Lines | Key class | Imported by | Status | Notes |
|---|---|---|---|---|---|
| `filterzyme/pipeline.py` | 390 | `Pipeline`, `Docking`, `Superimposition`, `GeometricFilters` | README example; benchmarks | **partial** | Has `alternative_strucuture_for_vina` typo at lines 151, 154. Uses local Squidly subprocess. **Deprecate per consolidation plan.** |
| `filterzyme/pipeline_v2.py` | 410 | same names | â (not imported by any benchmark) | **partial** | Canonical going forward. Uses `enzymetk.ActiveSitePred`. `_proteinRMSD` returns tuple; downstream depends on this. |
| `filterzyme/__init__.py` | 1 | â | â | working | Exports `__version__ = "0.0.6"` only. No public API surface defined. |
| `setup.py` | 52 | â | â | **partial** | `ppython_requires` typo (line 50); classifiers list Python 3.6/3.7/3.8 (lines 37â39) contradicting README and `environment.yml`. |
| `environment.yml` | 27 | â | â | **partial** | Lists `filterzyme` as a pip dep of its own env (line 23) â self-referential / circular for development install. |
| `submit_pipeline.sh` | 21 | â | â | **broken** | Calls `run_pipeline_on_multiple_substrates.py` (line 16) â file does not exist in the repo. |
| `test_pipeline.py` | 52 | â | â | **broken** | Imports `filtering_pipeline.pipeline` (lines 5â8) â old package name. Hard-coded `/nvme2/helen/...` path at line 12. Passes `find_closest_nuc=1` kwarg to `GeometricFilters` (line 44) â not accepted by either pipeline constructor â `TypeError`. |

## `filterzyme/steps/`

| File | Lines | Key class | Imported by | Status | Notes |
|---|---|---|---|---|---|
| `step.py` | 74 | `Pipeline`, `Step` | every step module | **partial** | Two `Step` definitions: lines 51â64 and lines 66â74. Second shadows first; both expose operators so behaviour is preserved. Placeholder docstring `""" Execute some shit """` at line 54. Triple-quoted block lines 21â49 contains two additional dead `Pipeline` class definitions in a string. |
| `save_step.py` | â | `Save` | both pipelines | working | Pickles incoming DataFrame to disk; passthrough on return. |
| `predict_catalyticsite_step.py` | 71 | `ActiveSitePred` | `pipeline.py` only | **partial** | Wraps `predict_catalyticsite_run.py` via `subprocess.run` (line 41) with `capture_output=True`. Errors logged but not raised. |
| `predict_catalyticsite_run.py` | 47 | `run_as_inference` | run as subprocess by `predict_catalyticsite_step.py` | **partial** | Uses `conda run -n AS_inference ...` (line 19) and `os.system(command)` (line 23) â return code is discarded. `main()` is unconditionally called at module top level (no `if __name__ == "__main__"` guard, by design). |
| `dock_vina_step.py` | 96 | `Vina` | both pipelines | **partial** | `int(r) + 1` at line 36 raises on empty residue strings. Tries `docko.get_alphafold_structure` at line 44. Uses `multiprocessing.dummy.Pool` (line 8) â threads, not processes, GIL-bound. Inherits from `enzymetk.step.Step` (line 1), not the local `Step`. |
| `extract_docking_metrics_step.py` | 262 | `DockingMetrics` | both pipelines | **partial** | `execute` at line 252. Silently falls back to empty dicts when parsing fails (line 245 swallows exceptions and prints). |
| `preparevina_step.py` | â | `PrepareVina` (line 163) | both | working | Flattens Vina pose files into a uniform layout. |
| `preparechai_step.py` | â | `PrepareChai` (line 155) | both | working | Has a `heme = 0` kwarg with no callsite using it â dead flag. |
| `prepareboltz_step.py` | â | `PrepareBoltz` (line 64) | both | working | â |
| `superimposestructures_step.py` | â | `SuperimposeStructures` (line 236) | both | working | Wraps `biotite.structure.superimpose_homologs`. |
| `computeproteinRMSD_step.py` | â | `ProteinRMSD` (line 177) | both | **partial** | `execute` returns `(pairwise, summary)` tuple â only v2 uses both, v1 silently relies on first. |
| `computeligandRMSD_step.py` | â | `LigandRMSD` (line 370) | both | **partial** | Re-defines `get_hetatm_chain_ids` (line 43), `extract_chain_as_rdkit_mol` (line 58), `_norm_l1_dist` (line 321), `closest_ligands_by_element_composition` (line 338) â already present in `utils/helpers.py`. |
| `geometric_filtering_cofactor.py` | 595 | `GeneralGeometricFiltering` (line 450) | `pipeline.py` (v1) | **partial** | Also re-defines the 4 helpers (lines 54/68/84/124). Accepts `esterase=0, find_closest_nucleophile=0` kwargs â never set by either pipeline. **Slated for deletion in favour of `_MCS` variant.** |
| `geometric_filtering_cofactor_MCS.py` | 580 | `GeneralGeometricFiltering` (line 435) | `pipeline_v2.py` (canonical) | **partial** | Imports helpers from `utils/helpers.py` correctly. Constructor signature simpler than the non-MCS sibling. |
| `geometric_filtering_esterase.py` | 555 | `EsteraseGeometricFiltering` (line 418) | both | working | Does not duplicate helpers. Closest to a clean reference implementation. |
| `fpocket_step.py` | â | `Fpocket` (line 232) | both | working | Shells out to `fpocket` binary; imports the deduplicated helpers from `utils/helpers.py`. |
| `ligandSASA_step.py` | â | `LigandSASA` (line 81) | both | working | â |
| `plip_step.py` | â | `PLIP` (line 81) | both | working | â |
| `cleanPDB_step.py` | â | `CleanPDB` (line 22) | â (only used inside Vina prep) | working | â |
| `PLACER_step.py` | â | `PLACER` (line 15) | â | **dead** | Never imported by either pipeline. References `python PLACER/run_PLACER.py` (line 46) â expects a sibling PLACER repo not present. |
| `PLACER_forChai_step.py` | â | `PLACER` (line 13) | â | **dead** | Same as above. Class is named `PLACER` â would collide if any of the three were imported together. |
| `PLACER_forVina_step.py` | â | `PLACER` (line 13) | â | **dead** | Same class name collision risk. |

## `filterzyme/utils/`

| File | Lines | Key symbols | Status | Notes |
|---|---|---|---|---|
| `helpers.py` | 558 | `log_section/subsection/boxed_note`, `log_usage`, `generate_chai_structure_path`, `generate_boltz_structure_path`, `clean_protein_sequence`, `delete_empty_subdirs`, `valid_file_list`, `add_metrics`, `add_metrics_to_best_structures`, `extract_docking_metrics`, `get_hetatm_chain_ids`, `extract_chain_as_rdkit_mol`, `closest_ligands_by_element_composition`, `atom_composition_fingerprint`, `norm_l1_dist`, `SingleLigandSelect`, `suppress_stdout_stderr`, `as_mol`, `ensure_3d` | **partial** | `add_metrics` (line 221) is broken â references undefined `dict_columns` (line 251) and `extract_vina_index` (line 264). The working counterpart `extract_docking_metrics` is at line 277. |

## `filterzyme/squidly_final_models/`

Contains pre-trained Squidly weights â binary `.pt` / `.pth` files. Also contains a `.DS_Store` (macOS Finder metadata) that is checked in â remove and add to `.gitignore`.

## `benchmarking/`

| File | Status | Notes |
|---|---|---|
| `martinez/run_martinez.py` | **partial** | `sys.path.insert(0, '/nvme2/helen/EnzymeStructuralFiltering/')` at line 11; hard-coded `squidly_dir='/nvme2/helen/EnzymeStructuralFiltering/filtering_pipeline/...'` at line 45. Imports `filterzyme.pipeline` (v1). |
| `martinez/N1_Benchmark_martinez.ipynb` | **partial** | Hard-coded `/nvme2/helen/...` paths to CSV inputs (lines 32, 9115). |
| `serine_hydrolases/run_serine_hydrolases.py` | **partial** | Same pattern; hard-coded paths at lines 9, 43. |
| `serine_hydrolases/N2_Benchmark_serine_hydrolases.ipynb` | not audited | Notebook outputs likely contain hard-coded paths analogous to martinez. |

## `tests/`

| File | Status | Notes |
|---|---|---|
| `tests/PredictTPPFiltering.ipynb` | not audited | Notebook only. |
| `tests/visualize_results.ipynb` | **partial** | Numerous hard-coded `/nvme2/helen/...` paths (lines 233â339). |

No pytest, no unittest, no doctest in the entire repository.

## Summary counts

- 24 step modules, of which 3 are dead (the PLACER trio) and 2 are deeply redundant (`geometric_filtering_cofactor.py` is the older sibling of `_MCS`; `computeligandRMSD_step.py` duplicates 4 helpers).
- 2 top-level pipelines, one of which (`pipeline.py`) is the documented entry point and the other (`pipeline_v2.py`) is the better-engineered one.
- 0 unit tests.
- 1 example pickle (`examples/DEHP-MEHP.pkl`).
