# 7. Top 10 Code Improvements

Ranked by impact-to-effort ratio. Each entry names the problem, the fix, the success criterion, and the roadmap phase that contains it.

## 1. Consolidate to a single canonical pipeline

**Problem.** `pipeline.py` and `pipeline_v2.py` are siblings with incompatible intermediate schemas (`label`/`Squidly_CR_Position` vs `id`/`residues` at `pipeline.py:80-83` vs `pipeline_v2.py:86-89`) and different return shapes from `_proteinRMSD` / `_ligandRMSD`. The README example and benchmarks point to v1; v2 is better-instrumented.

**Fix.** Promote `pipeline_v2.py` to canonical. Port `skip_catalytic_residue_prediction` and the metagenomic Vina-fallback flag from v1 into v2 (with the typo fixed). Update README, `benchmarking/martinez/run_martinez.py:38`, and `benchmarking/serine_hydrolases/run_serine_hydrolases.py:36` to import the consolidated pipeline. Delete `pipeline.py`, `predict_catalyticsite_step.py`, and `predict_catalyticsite_run.py`.

**Success.** Single `from filterzyme.pipeline import Pipeline` works; benchmarks reproduce their previous outputs (within numerical noise of Chai/Boltz stochasticity); `grep -r pipeline_v2` returns nothing.

**Phase.** 1.

## 2. Move duplicated helpers to `utils/`

**Problem.** `get_hetatm_chain_ids`, `extract_chain_as_rdkit_mol`, `_norm_l1_dist`, `closest_ligands_by_element_composition` are duplicated in `computeligandRMSD_step.py` (lines 43, 58, 321, 338) and `geometric_filtering_cofactor.py` (lines 54, 68, 84, 124). The canonical copies live in `utils/helpers.py` at lines 410, 425, 484.

**Fix.** Delete the duplicates in both files. Add `from filterzyme.utils.helpers import get_hetatm_chain_ids, extract_chain_as_rdkit_mol, closest_ligands_by_element_composition, norm_l1_dist` at the top. (`_norm_l1_dist` with leading underscore in the dup is a copy of the unprefixed `norm_l1_dist` in `helpers.py` â pick one name during the refactor.)

**Success.** `wc -l filterzyme/steps/computeligandRMSD_step.py` drops by ~40; `wc -l filterzyme/steps/geometric_filtering_cofactor.py` drops by ~60. Single source of truth.

**Phase.** 1.

## 3. Replace local Squidly subprocess with direct import

**Problem.** `predict_catalyticsite_step.py:41` and `predict_catalyticsite_run.py:23` chain `subprocess.run` â `os.system("conda run -n AS_inference ...")`. Three nested process boundaries, no exit-code propagation, silent failure on missing env. `predict_catalyticsite_run.py:14` hard-codes `as_threshold = 0.97`, silently overriding the CLI value.

**Fix.** Use `enzymetk.ActiveSitePred` directly (the v2 path at `pipeline_v2.py:86`). Delete the local files. If the host env lacks the Squidly weights at runtime, fall back to a clear `RuntimeError` with installation instructions.

**Success.** No subprocess, no `os.system`, no separate conda env. Squidly runs in the same process as the rest of the pipeline.

**Phase.** 2.

## 4. Add a Pydantic config schema loadable from YAML

**Problem.** Every parameter (`max_matches`, `esterase`, `metagenomic_enzymes`, `num_threads`, `squidly_dir`, `base_output_dir`, `alternative_structure_for_vina`, plus all the magic numbers from CQ-14) is a constructor kwarg. A real run requires editing Python source.

**Fix.** Introduce `filterzyme/config.py` with a Pydantic `PipelineConfig` model. Load from `--config config.yaml`. Default config under `examples/config.example.yaml`.

```python
class VinaConfig(BaseModel):
    exhaustiveness: int = 8
    num_modes: int = 9
    box_size: float = 10.0
    ph: float = 7.4

class PipelineConfig(BaseModel):
    base_output_dir: Path
    num_threads: int = 1
    gpu_devices: list[int] = [0]
    esterase: bool = False
    metagenomic_enzymes: bool = False
    alternative_structure_for_vina: Literal["Chai", "Boltz"] = "Chai"
    vina: VinaConfig = VinaConfig()
```

**Success.** `Pipeline.from_config(Path("config.yaml"))` works; no parameter is set in Python source.

**Phase.** 4.

## 5. Add input-DataFrame validator

**Problem.** Missing required columns (`Entry`, `Sequence`, `substrate_name`, `substrate_smiles`, `substrate_moiety`) surface as a confusing `KeyError` several stages downstream.

**Fix.** Add `filterzyme/utils/validation.py:validate_input_df(df) -> None` that checks for required columns, dtype consistency, non-empty strings, and SMILES parseability via RDKit. Call from `Pipeline.__init__` after the DataFrame is bound.

```python
REQUIRED = {"Entry": str, "Sequence": str, "substrate_name": str,
            "substrate_smiles": str, "substrate_moiety": str}

def validate_input_df(df):
    missing = REQUIRED.keys() - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {sorted(missing)}")
    # ... dtype + SMILES checks
```

**Success.** A DataFrame missing `substrate_smiles` raises `ValueError` at construction with a clear message, not several stages later.

**Phase.** 1.

## 6. Add pytest suite

**Problem.** No unit tests, no integration tests. `tests/` contains only notebooks; `test_pipeline.py` at the root is broken (P0-6).

**Fix.** Create `tests/test_helpers.py` (unit tests for `clean_protein_sequence`, `closest_ligands_by_element_composition`, `extract_docking_metrics`), `tests/test_validation.py` (input-validator edge cases), and `tests/test_pipeline_smoke.py` (mock Chai/Boltz/Vina, run end-to-end on `examples/DEHP-MEHP.pkl`). Use `pytest-mock` for the upstream-tool boundaries.

**Success.** `pytest -q` completes in <60 s in CI; coverage >40% on `utils/helpers.py`.

**Phase.** 1 (smoke test) + 4 (CI).

## 7. Add structured logging

**Problem.** `print()` mixed with `logger.*` across ~30 files (CQ-9). Log lines from concurrent workers interleave.

**Fix.** Configure a single root logger in `filterzyme/__init__.py` with a rotating file handler, a console handler, and worker-id injected via `contextvars`. Add a `--log-level` CLI flag. Replace all `print(` outside `__main__` blocks with `logger.info(` / `logger.warning(`.

**Success.** A single run produces one log file at `<output_dir>/filterzyme.log`; no stray `print` outputs reach stderr.

**Phase.** 0 (replace prints) + 4 (rotating handler + worker tagging).

## 8. Add checkpoint / resume

**Problem.** No stage reads its output pickle as a cache. A four-hour failure restarts from stage zero.

**Fix.** Wrap each stage's `run` with a cache check:

```python
def cached_run(self, key: str, fn: Callable):
    out = Path(self.output_dir) / f"{key}.pkl"
    if out.exists() and not self.force:
        log_boxed_note(f"Skipping {key}, cache hit: {out}")
        return pd.read_pickle(out)
    df = fn()
    df.to_pickle(out)
    return df
```

Pair with a `--force` global flag and a `--force-stage <name>` selective flag.

**Success.** Killing the run after Vina completes, then restarting, skips straight to Superimposition.

**Phase.** 4.

## 9. Make AF2 fetch cacheable and offline-capable

**Problem.** `dock_vina_step.py:44` calls `docko.get_alphafold_structure` on every run with no shared cache, and crashes the entry on network failure.

**Fix.** Add `af2_cache_dir: Path` to `VinaConfig`. Before fetching, check `af2_cache_dir / f"{uniprot_id}.pdb"`. On fetch success, copy into the cache. Add an `--offline` mode that errors clearly if the cache misses.

**Success.** A second run completes the AF2 step in <1 s per entry.

**Phase.** 3.

## 10. Use `ProcessPoolExecutor` for CPU stages, threads only for I/O

**Problem.** `dock_vina_step.py:8 from multiprocessing.dummy import Pool as ThreadPool` is thread-based and GIL-bound. CPU-bound Python around Vina (RDKit prep, residue parsing, PDB cleanup) does not scale.

**Fix.** Use `concurrent.futures.ProcessPoolExecutor` for Vina prep, PLIP, fpocket, and the geometric filters. Keep threads for AF2 download, file copies, and other I/O-bound work. Gate by `num_workers: int = max(1, os.cpu_count() // 2)` from the config.

**Success.** Wall-clock time on a 20-entry sweep drops by ~Nx where N is the number of cores (modulo Vina-binary parallelism).

**Phase.** 4.
