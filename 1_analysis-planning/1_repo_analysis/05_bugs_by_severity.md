# 5. Bugs by Severity

All citations were verified by direct file read. Severities:

- **P0** â crash on documented input, data corruption, or blocking import failure
- **P1** â silent failure, partial result, or environment-specific crash
- **P2** â wrong behaviour under load, missing validation, or misleading output
- **P3** â cosmetic, hygiene, or technical-debt items

## P0 â critical

### P0-1: `alternative_strucuture_for_vina` typo in `pipeline.py`

`pipeline.py:151` and `pipeline.py:154`:

```python
if self.alternative_strucuture_for_vina == 'Chai':       # line 151 (sic)
    ...
elif self.alternative_strucuture_for_vina == 'Boltz':    # line 154 (sic)
```

The attribute is set as `alternative_structure_for_vina` at `pipeline.py:50`. Reachable only when `metagenomic_enzymes == 1`. Raises `AttributeError`. **Fix:** rename both references. `pipeline_v2.py` is correct already (lines 158, 161, 178, 186).

### P0-2: `_proteinRMSD` / `_ligandRMSD` return-shape mismatch between v1 and v2

`pipeline_v2.py:269`:

```python
return df_proteinRMSD_pairwise, df_proteinRMSD
```

`pipeline_v2.py:281`:

```python
return df_ligandRMSD_pairwise, df_ligandRMSD_w_metrics
```

Both return 2-tuples. The corresponding `pipeline.py:258` and `pipeline.py:267` return single DataFrames. Any code that calls into one pipeline expecting the other's shape unpacks incorrectly. **Fix:** during consolidation, settle on the 2-tuple shape in v2 and update all callers; pipeline.py is being deleted anyway.

### P0-3: `helpers.add_metrics` references undefined symbols

`utils/helpers.py:221-273`:

```python
def add_metrics(best_strucutures_df, df_dockmetrics):
    ...
    df_dockmetrics_reduced = df_dockmetrics[
        ["Entry"] + dict_columns + ["vina_affinities"]      # line 251 â dict_columns undefined
    ].drop_duplicates(subset="Entry")
    ...
    for col in dict_columns:                                # line 257 â same
        ...
    vina_indices = merged_df["docked_structure"].map(extract_vina_index)   # line 264 â extract_vina_index undefined
```

`dict_columns` and `extract_vina_index` are not defined anywhere in `helpers.py` or imported. The function defines local helpers `extract_structure_id` (line 237) and `extract_index` (line 243) but never uses `extract_index` and never assembles `dict_columns`. `add_metrics` is imported by `pipeline_v2.py:11`. Calling it raises `NameError`. **Fix:** assemble `dict_columns = chai_columns + boltz_columns` after their definitions (lines 226 and 230) and rename `extract_index` to `extract_vina_index`. The sibling `extract_docking_metrics` at line 277 already does the right thing and can serve as a template.

### P0-4: Two divergent pipelines with incompatible intermediate schemas

`pipeline.py:80-83`:

```python
df_cat_res = pred_in << ActiveSitePred('Entry', 'Sequence', self.squidly_dir, self.num_threads)
df_cat_res = df_cat_res.merge(reps, left_on='label', right_on='rep_entry', how='left')
df_squidly = self.df.merge(
    df_cat_res[['Sequence', 'Squidly_CR_Position']].drop_duplicates('Sequence'),
    on='Sequence', how='left')
```

`pipeline_v2.py:86-89`:

```python
df_cat_res = pred_in << ActiveSitePred('Entry', 'Sequence')
residues = dict(zip(df_cat_res.id, df_cat_res.residues))
df_squidly = self.df.copy()
df_squidly['Squidly_CR_Position'] = [residues.get(e) for e in df_squidly['Entry'].values]
```

v1 expects `df_cat_res` to carry `label` and `Squidly_CR_Position`; v2 expects `id` and `residues`. The two cannot share intermediates. The README example points to v1; the benchmarks also import v1; v2 is the better-instrumented file. **Fix (locked in):** keep v2 as canonical. Port the v1-only feature `skip_catalytic_residue_prediction` and the local-Squidly fallback (if needed) into v2 behind a flag, then delete `pipeline.py` and the local `predict_catalyticsite_step.py` / `predict_catalyticsite_run.py`.

### P0-5: `submit_pipeline.sh` calls a non-existent script

`submit_pipeline.sh:16`:

```bash
python run_pipeline_on_multiple_substrates.py "$name" "${LIGANDS[$name]}" \
```

`run_pipeline_on_multiple_substrates.py` does not exist in the repo. **Fix:** either commit the script or rewrite the loop to call a documented CLI entry point (see Phase 4 of the roadmap â a `filterzyme run` CLI is recommended).

### P0-6: `test_pipeline.py` imports a long-renamed package

`test_pipeline.py:5-8`:

```python
from filtering_pipeline.pipeline import Pipeline
from filtering_pipeline.pipeline import Docking
from filtering_pipeline.pipeline import Superimposition
from filtering_pipeline.pipeline import GeometricFilters
```

The package was renamed to `filterzyme` long ago. `test_pipeline.py:44` also passes `find_closest_nuc=1` to `GeometricFilters` â a kwarg that neither `pipeline.py:271` nor `pipeline_v2.py:285` accepts â `TypeError`. **Fix:** delete `test_pipeline.py` and replace with a pytest smoke test under `tests/` (see Phase 1 of the roadmap).

## P1 â high

### P1-1: Three-deep subprocess chain for Squidly (v1)

`predict_catalyticsite_step.py:41`:

```python
result = subprocess.run(['python', Path(__file__).parent/'predict_catalyticsite_run.py',
                        '--out', str(tmp_dir),
                        '--input', input_filename,
                        '--squidly_dir', f'{self.squidly_dir}/',
                        '--esm2_model', self.esm2_model],
                       capture_output=True, text=True)
```

`predict_catalyticsite_run.py:19-23`:

```python
command = f'conda run -n AS_inference python {squidly_dir}SQUIDLY_run_model_LSTM.py \
          {fasta_file} {esm2_model} {cr_model_as} {lstm_model_as} {output_dir} \
          --toks_per_batch {toks_per_batch} --AS_threshold {as_threshold}'
print(command)
os.system(command)
```

Three nested process boundaries. `os.system` discards the return code. If `AS_inference` does not exist or `conda` is not on `PATH`, the call appears to succeed but emits no `_results.pkl`, and the outer step then crashes on the missing file. **Fix:** remove the entire local Squidly path; use `enzymetk.ActiveSitePred` (v2 path) which calls the model directly in-process.

### P1-2: `as_threshold` silently overridden inside subprocess

`predict_catalyticsite_run.py:14`:

```python
as_threshold = 0.97
```

This line overwrites the value passed from the parent process â whatever the caller specifies on the command line is discarded. The CLI default at line 32 is `0.90`. The override is silent. **Fix:** delete line 14.

### P1-3: `int(r) + 1` crashes on empty residue strings

`dock_vina_step.py:35-36`:

```python
residues = str(residues)
residues = [int(r) + 1 for r in residues.split('|')]
```

`'' .split('|')` yields `['']`; `int('')` raises `ValueError`. Entries with `catalytic_residues == ''` (which v2 explicitly creates at `pipeline_v2.py:95-104` before its filter) hit this path if the upstream filter fails to drop them. **Fix:** `residues = [int(r) + 1 for r in residues.split('|') if r.strip()]` and skip the entry if the resulting list is empty.

### P1-4: `setup.py` typo silently disables Python-version floor

`setup.py:50`:

```python
ppython_requires=">=3.10",
```

Setuptools accepts unknown kwargs without warning. The project declares Python 3.10+ but does not enforce it; `environment.yml` pins 3.11.8; classifiers at `setup.py:37-39` list 3.6/3.7/3.8. **Fix:** rename to `python_requires` and remove the 3.6/3.7/3.8 classifiers.

### P1-5: Hard-coded `/nvme2/helen/...` paths

`README.md:65`:

```python
squidly_dir='/nvme2/helen/EnzymeStructuralFiltering/filterzyme/squidly_final_models/'
```

`benchmarking/martinez/run_martinez.py:11`:

```python
sys.path.insert(0, '/nvme2/helen/EnzymeStructuralFiltering/')
```

`benchmarking/serine_hydrolases/run_serine_hydrolases.py:9`:

```python
sys.path.insert(0, '/nvme2/helen/EnzymeStructuralFiltering/')
```

Also at `run_martinez.py:45`, `run_serine_hydrolases.py:43`, `test_pipeline.py:12`, plus many notebook cells (`tests/visualize_results.ipynb:233-339`, `benchmarking/martinez/N1_Benchmark_martinez.ipynb:32, 9115`). **Fix:** replace with `pkg_resources.resource_filename('filterzyme', 'squidly_final_models/')` (or `importlib.resources` for 3.10+), and parameterise notebook paths via an env var.

### P1-6: Brittle `.parent / 'docking/...'` path in v1

`pipeline.py:265`:

```python
df_best_structures_w_metrics = add_metrics_to_best_structures(
    df_best_structures,
    pd.read_pickle(Path(self.output_dir).parent / 'docking/dockingmetrics.pkl'))
```

Assumes `output_dir` is a sibling of `docking/` (i.e. that `Superimposition.output_dir == base/superimposition`). If a caller sets `output_dir` elsewhere, the read fails. **Fix:** pass `docking_metrics_path` explicitly to `Superimposition.__init__`, defaulting to the conventional location.

### P1-7: `Vina.__execute` name mangling + thread pool

`dock_vina_step.py:28`:

```python
def __execute(self, df: pd.DataFrame) -> pd.DataFrame:
```

Double-underscore triggers Python name mangling (`_Vina__execute`). Combined with `multiprocessing.dummy.Pool` (line 8 import), this works only because the pool is invoked from inside the same class. Any subclass override of `__execute` will not be picked up. **Fix:** rename to single-underscore `_execute`.

## P2 â medium

### P2-1: No input DataFrame validation

The pipeline never checks that the required columns (`Entry`, `Sequence`, `substrate_name`, `substrate_smiles`, `substrate_moiety`) are present. A missing column surfaces several stages downstream as a confusing `KeyError`. **Fix:** add a `validate_input(df)` function called from `Pipeline.__init__` â see Phase 4 of the roadmap.

### P2-2: Cofactor sentinel mismatch between consecutive stages

`pipeline.py:127-138`:

```python
if 'cofactor_smiles' not in df_squidly.columns:
    df_squidly['cofactor_smiles'] = ''           # â '' for Chai
df_chai = df_squidly << (Chai(...) >> Save(...))
...
if 'cofactor_smiles' not in df_chai.columns:
    df_chai['cofactor_smiles'] = None            # â None for Boltz
```

The same column is filled with `''` for Chai and `None` for Boltz. Whatever `enzymetk.Chai` and `enzymetk.Boltz` do with the sentinel is by accident, not by design. **Fix:** initialise once at the top of `Docking.run` and use a single consistent sentinel (recommended: `None`, since Boltz appears to handle it).

### P2-3: `multiprocessing.dummy` is threads, not processes

`dock_vina_step.py:8`:

```python
from multiprocessing.dummy import Pool as ThreadPool
```

The `dummy` submodule is the threading-backed analog of `multiprocessing.Pool`. For CPU-bound work (Vina's own loop is in a subprocess, but the surrounding RDKit / docko Python code is not), this is GIL-bound and provides limited speedup. **Fix:** use `concurrent.futures.ProcessPoolExecutor` for CPU-bound stages; keep threads for I/O-bound stages (AF2 fetch, file copy).

### P2-4: No GPU device routing

When `num_threads > 1` and Chai / Boltz are running, every worker sees the same `CUDA_VISIBLE_DEVICES` and contends for one GPU. **Fix:** add a `gpu_devices: list[int]` config option; round-robin assignment per worker via `CUDA_VISIBLE_DEVICES` env override.

### P2-5: `DockingMetrics.execute` silently falls back to empty dicts

`extract_docking_metrics_step.py:244-245`:

```python
except Exception as e:
    print(f"failed to parse {json_file.name}: {e}")
```

The exception is swallowed and an empty dict is returned for the affected entry, which then propagates as NaN through `add_metrics_to_best_structures`. **Fix:** log via `logger.warning` and record the failed entry in a per-run error CSV (see Phase 4).

### P2-6: Inconsistent diagnostics â `print()` next to `logger.*`

`dock_vina_step.py:18`, `:49`, `:79`, `predict_catalyticsite_step.py:63`, `pipeline.py:175`, and ~30 other locations use `print()` alongside `logger.info` / `logger.error`. The two write to different sinks (stdout vs the logging handler chain), so reading a run log is harder than it should be. **Fix:** replace `print` with `logger.info` everywhere outside `__main__` blocks.

### P2-7: Three near-identical geometric-filtering files

`geometric_filtering_cofactor.py` (595 lines), `geometric_filtering_cofactor_MCS.py` (580 lines), `geometric_filtering_esterase.py` (555 lines) â ~1,730 LOC total, much of it duplicated logic for distance / SMARTS / chain-extraction. The MCS variant is the modern one; `cofactor.py` is its older sibling that also re-defines four helpers already present in `utils/helpers.py`. **Fix:** delete `geometric_filtering_cofactor.py` after `pipeline.py` is removed; refactor common code (atom selection, distance kernels) out of the remaining two into `utils/`. Estimated reduction: 600â800 LOC.

### P2-8: Dead `find_closest_nucleophile` kwarg

`geometric_filtering_cofactor.py:450`:

```python
def __init__(self, preparedfiles_dir: str = '', esterase = 0, find_closest_nucleophile = 0, output_dir: str= ''):
```

`find_closest_nucleophile` is never set by any pipeline (`test_pipeline.py:44` tries `find_closest_nuc=1` â wrong name, wrong target class, anyway). **Fix:** drop the kwarg (or wire it through if the feature is intended).

### P2-9: `environment.yml` self-reference

`environment.yml:23`:

```yaml
  - filterzyme
```

The env declares `filterzyme` as a pip dep of its own dev env. For source development this should be `-e .` against the cloned repo, not the PyPI release. **Fix:** remove the entry and instruct developers to `pip install -e .` after creating the env.

## P3 â low

### P3-1: `.DS_Store` in tree

`filterzyme/squidly_final_models/.DS_Store` is checked in. Add `**/.DS_Store` to `.gitignore` and `git rm --cached` the file.

### P3-2: No pytest

`tests/` contains only Jupyter notebooks (`PredictTPPFiltering.ipynb`, `visualize_results.ipynb`). No `pytest` config, no `conftest.py`. The closest thing is the broken `test_pipeline.py` at repo root.

### P3-3: Sparse and unprofessional docstrings

`step.py:54`:

```python
def execute(self, df: pd.DataFrame) -> pd.DataFrame:
    """ Execute some shit """
```

Most step `execute` methods have no docstring at all. Public-API classes (`Pipeline`, `Docking`, `Superimposition`, `GeometricFilters`) have one-line summaries that do not document parameters or return shapes.

### P3-4: Triple-quoted dead code inside `step.py`

`step.py:21-49` is a triple-quoted string containing two more `Pipeline` definitions. Harmless because it's a string, but it signals abandoned refactoring and clutters the file. **Fix:** delete.

### P3-5: Duplicate `Step` class definition in `step.py`

`step.py:51` and `step.py:66` both define a `Step` class. The second shadows the first; both expose the operator dunders, so behaviour is preserved. **Fix:** keep the second (more concise) version; delete lines 51â64.

### P3-6: `preparechai_step.py:155` has dead `heme = 0` kwarg

The kwarg is accepted but never consulted internally. Either wire it through or drop it.

### P3-7: `extract_docking_metrics` `Vina` import shadowing

`dock_vina_step.py:3`:

```python
from docko.docko import *
```

Wildcard import. Pollutes the module namespace and makes it impossible to grep for the origin of any symbol used at module level.

### P3-8: Pipeline class shadowed inside `step.py` docstring block

Three `Pipeline` definitions appear in `step.py` (lines 6, 24, 38, 39), the latter three inside a triple-quoted block. Confusing on first read.
