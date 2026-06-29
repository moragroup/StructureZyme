# 6. Code Quality Issues

This section catalogues the cross-cutting code-quality problems that are not single-bug fixes but rather structural debt. Numbered for cross-reference from the roadmap.

## CQ-1: Helper duplication

`get_hetatm_chain_ids`, `extract_chain_as_rdkit_mol`, `closest_ligands_by_element_composition`, and `_norm_l1_dist` are defined in `utils/helpers.py` (the canonical location, lines 410, 425, 484, plus a sibling `norm_l1_dist` without underscore-prefix), and **redefined verbatim or near-verbatim** in:

- `filterzyme/steps/computeligandRMSD_step.py:43, 58, 321, 338`
- `filterzyme/steps/geometric_filtering_cofactor.py:54, 68, 84, 124`

Other consumers (`fpocket_step.py:17`, `plip_step.py:13`, `ligandSASA_step.py:15`, `geometric_filtering_cofactor_MCS.py:24-26`) correctly import from `utils/helpers.py`. The two laggards are obvious copy-paste descendants. **Action:** delete the duplicate definitions and import from `utils/helpers.py`.

## CQ-2: Two `Step` / three `Pipeline` definitions in `step.py`

`step.py` is 74 lines and contains:

- `Pipeline` at line 6 (active)
- Two more `Pipeline` definitions at lines 24, 38â39 (inside a triple-quoted string â harmless but confusing)
- `Step` at line 51 (with `__rshift__` / `__rlshift__`)
- `Step` at line 66 (shadows the first; also has `__rshift__` / `__rlshift__`)
- Placeholder docstring `""" Execute some shit """` at line 54

The operator-overloading actually works because the second `Step` (line 66) implements the dunders. But anyone subclassing `Step` to add a new operator will silently fail to override the first definition. **Action:** strip to one `Pipeline` and one `Step`; remove the triple-quoted block; replace the placeholder docstring with real documentation.

## CQ-3: `Vina.__execute` name mangling + thread pool

`dock_vina_step.py:28` uses a double-underscore method name (`__execute`). Python name-mangles it to `_Vina__execute`. The `multiprocessing.dummy.Pool` call (which is inside the class so the mangling resolves correctly) hides a subclassing bug: a subclass that overrides `__execute` will not change what the pool calls. **Action:** rename to `_execute` and document the override contract.

## CQ-4: No retry, no resume, no checkpoint

Every stage writes its output pickle to disk (good), but no stage checks for the existence of its output pickle before re-running (bad). A 4-hour Vina sweep that fails on the last entry has to be rerun from scratch. **Action:** add a stage-level idempotent cache:

```python
out_path = Path(self.output_dir) / 'vina.pkl'
if out_path.exists() and not self.force:
    log_boxed_note(f'Skipping Vina, using cached {out_path}')
    return pd.read_pickle(out_path)
```

Apply the same pattern to Chai, Boltz, ProteinRMSD, LigandRMSD, fpocket, LigandSASA, PLIP. Provide a global `--force` flag and a per-stage `--force-stage <name>` mode.

## CQ-5: No config file, all params as kwargs

Every parameter (`max_matches`, `esterase`, `metagenomic_enzymes`, `skip_catalytic_residue_prediction`, `alternative_structure_for_vina`, `num_threads`, `squidly_dir`, `base_output_dir`) is a constructor kwarg. A real run requires editing Python source. **Action:** introduce a Pydantic config schema loadable from YAML/JSON â see Phase 4 of the roadmap.

## CQ-6: Almost no type hints

Of ~30 step `execute` methods, fewer than half have parameter or return type hints. Public-API classes (`Pipeline`, `Docking`, `Superimposition`, `GeometricFilters`) annotate kwargs but not return types. `pipeline_v2.py:266` returns `Tuple[pd.DataFrame, pd.DataFrame]` with no annotation. **Action:** progressively annotate; run `mypy --strict` on `utils/helpers.py` first (smallest surface), then expand.

## CQ-7: Inconsistent module import style

`from filterzyme.steps.step import Step` is the dominant style. But `predict_catalyticsite_step.py:11`, `preparechai_step.py:10`, `preparevina_step.py:10`, `save_step.py:1` use the relative `from .step import Step`. And `dock_vina_step.py:1` imports `from enzymetk.step import Step` â the **enzymetk** step base class, not the local one. The result: `Vina` is not a subclass of the same `Step` as the rest of the pipeline. This works because operator dispatch is duck-typed, but it is a latent footgun. **Action:** choose one style (absolute imports recommended) and apply uniformly; make `Vina` extend the local `Step`.

## CQ-8: Wildcard import from external package

`dock_vina_step.py:3 from docko.docko import *` imports an unknown surface area at module load. Any name collision with future docko releases silently changes behaviour. **Action:** import only what's needed: `from docko.docko import dock, get_alphafold_structure, clean_one_pdb, pdb_to_pdbqt_protein`.

## CQ-9: `print()` mixed with `logger.*`

At least 30 instances of `print(...)` in `filterzyme/steps/*.py` and `filterzyme/pipeline.py`, alongside a configured root logger (`helpers.py:19`). The two write to different sinks, which makes run logs hard to read and hard to redirect. **Action:** replace `print` with `logger.info` / `logger.warning` / `logger.debug` outside of `__main__` blocks.

## CQ-10: Misleading PyPI metadata

`setup.py:37-39` lists Python 3.6/3.7/3.8 classifiers. `setup.py:50` has the `ppython_requires` typo. README and `environment.yml` target 3.10+/3.11.8. The PyPI page would tell users to install on 3.6 â which would fail at the first f-string or walrus operator. **Action:** fix classifiers, fix `python_requires` typo.

## CQ-11: No CI

No `.github/workflows/`, no GitLab CI, no pre-commit hooks. Every push could break import, and there's no green-light signal. **Action:** add a minimal GitHub Actions workflow that runs `pip install -e .` and `pytest -q` â see Phase 4 of the roadmap.

## CQ-12: `os.system` use

`predict_catalyticsite_run.py:23` uses `os.system(command)`, which (a) discards return codes and (b) shell-expands `command`, opening the door to quoting bugs when paths contain spaces. **Action:** replace with `subprocess.run(command, shell=False, check=True)` constructing the args list explicitly.

## CQ-13: Triple-nested process boundaries

The local Squidly path crosses three process boundaries: parent â `subprocess.run(python predict_catalyticsite_run.py ...)` â `os.system("conda run -n AS_inference python SQUIDLY_run_model_LSTM.py ...")`. Each boundary loses an environment variable, a stack trace, or an exit code. **Action:** kill the local path entirely; use `enzymetk.ActiveSitePred` in-process (v2 path).

## CQ-14: Hard-coded magic numbers

`dock_vina_step.py:71-73` hard-codes the Vina search box at 10 Ã  cubic. `dock_vina_step.py:69` hard-codes `pH=7.4`. `predict_catalyticsite_run.py:14` hard-codes `as_threshold=0.97`. `pipeline.py:71` and `pipeline_v2.py:78` always normalise sequences via `clean_protein_sequence`. None of these is configurable from the public API. **Action:** expose via the config file from CQ-5.

## CQ-15: Brittle path-shape coupling

`generate_chai_structure_path` (`helpers.py:115`) and `generate_boltz_structure_path` (`helpers.py:105`) encode the on-disk layout of upstream tools. A Chai upgrade that renames `chai/{name}_0.cif` to `chai/{name}_pred_0.cif` silently breaks the fallback path. **Action:** add a small integration test (Phase 1) that asserts these paths resolve against `examples/DEHP-MEHP.pkl`.

## Summary

| Item | Severity | Effort | Roadmap phase |
|---|---|---|---|
| CQ-1 helper duplication | medium | 1 day | Phase 1 |
| CQ-2 Step/Pipeline shadowing | low | 1 hour | Phase 0 |
| CQ-3 Vina name mangling | low | 1 hour | Phase 2 |
| CQ-4 no checkpoint/resume | high | 3 days | Phase 4 |
| CQ-5 no config file | high | 2 days | Phase 4 |
| CQ-6 no type hints | medium | rolling | Phase 5 |
| CQ-7 inconsistent imports | low | 1 hour | Phase 0 |
| CQ-8 wildcard import | low | 1 hour | Phase 0 |
| CQ-9 print vs logger | low | half-day | Phase 0 |
| CQ-10 misleading PyPI meta | low | 10 min | Phase 0 |
| CQ-11 no CI | medium | 1 day | Phase 4 |
| CQ-12 os.system | low | 1 hour | Phase 1 |
| CQ-13 triple subprocess | high | 2 days (part of pipeline consolidation) | Phase 2 |
| CQ-14 magic numbers | medium | rolling | Phase 4 |
| CQ-15 path-shape coupling | medium | 2 hours (just the integration test) | Phase 1 |
