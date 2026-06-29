# Task 09 — Recipe for installing & implementing future models

**Depends on:** Task 07 (so the integration pattern is proven). **Owner unit:** the "how to add models in the future" deliverable.

This generalizes how Filterzyme plugs in tools, so new models follow one repeatable pattern. Source detail: `../08_recommended_models.md` and `../04_model_integration_audit.md`.

## The integration pattern (how every model is wired)

1. **A Step subclass** in `filterzyme/steps/<tool>_step.py` subclassing the local `Step` (`filterzyme/steps/step.py`), implementing `execute(df) -> df` (and usually a private `_execute`). It either:
   - shells out to a binary (`fpocket`, `vina` via docko, `foldseek`), or
   - imports a Python/GPU lib (`chai_lab`, `boltz` via enzymetk wrappers).
2. **Wired into a stage** in `filterzyme/pipeline.py` using the `<<` (apply) and `>>` (chain) operators, e.g. `df << (Step(...) >> Save(path))`.
3. **Metrics parsing** added to `filterzyme/steps/extract_docking_metrics_step.py` if the model emits scores (note: Chai `.npz` and Boltz JSON keys are currently hard-coded there).
4. **Install**: binary tools come from conda (bioconda) or a static release on PATH; Python/GPU models via pip/conda into `filterpipeline` (or the enzymetk tool-env — see Task 05).

## Two install classes

- **Binary, drop-in (easiest):** GNINA, Smina, AutoDock Vina, foldseek, fpocket. Single executable on PATH; new `*_step.py` shells out to it. No Python deps.
- **GPU Python model (heavier):** Chai, Boltz, DiffDock, ESMFold, AF3. Needs torch matched to CUDA (Task 04 rule), weight downloads, VRAM budget; ship DiffDock/AF3 as optional pip extras.

## Recommended additions (priority order, from `../08_recommended_models.md`)

| Model | Effort | Install | Where it plugs in |
|---|---|---|---|
| **GNINA** (best first add) | ~2d | single CUDA binary on PATH | new `dock_gnina_step.py` modeled on `dock_vina_step.py`; add `docking_engine` switch; parse `CNNscore`/`CNNaffinity` in DockingMetrics |
| **Smina** | ~1d | single binary | `dock_smina_step.py` (~30 LOC over Vina); shares Vina CLI conventions |
| **ESMFold** | ~2d | `pip install fair-esm` + weights (~3.5 GB) | `esmfold_step.py` as AF2-miss fallback in `Vina.__execute` (tertiary: AF2 cache → AF2 fetch → ESMFold → Chai/Boltz) |
| **AlphaFold3 / OpenFold** | ~5d | gated weights / OpenFold; >24 GB VRAM (use h100) | `enzymetk.AF3` + `Docking._run_af3` mirroring `_run_chai`; 3-way superimposition |
| **DiffDock** | ~4d | optional extra: torch_geometric + weights (~1 GB); needs TRILL/micromamba env | `dock_diffdock_step.py`; dispatch on `docking_engine`; OUT OF SCOPE for first pass |

## Concrete "add a docking engine" checklist (use GNINA as template)

1. Install the binary: download GNINA static release → `chmod +x` → put on PATH (or bioconda). Verify `gnina --version` in `filterpipeline`.
2. Create `filterzyme/steps/dock_gnina_step.py`: same constructor signature as `Vina` (`id_col, structure_col, sequence_col, substrate_col, substrate_name_col, active_site_col, output_dir, num_threads`); in `_execute` call the gnina binary with the receptor pdbqt + ligand + box centered on residues.
3. Extend `extract_docking_metrics_step.py` to parse gnina output (`affinity`, `CNNscore`, `CNNaffinity`).
4. Add a `docking_engine` param to `Docking` / `Pipeline` and branch `_run_vina` vs `_run_gnina`.
5. Add a smoke test under `tests/` running the new step on `examples/DEHP-MEHP.pkl` (upstream binary mocked).

## Cross-cutting must-dos for any new model

- Pin versions (current repo pins almost nothing — see `../04_model_integration_audit.md` cross-cutting risks).
- Pick GPU device explicitly when `num_threads>1` (all threads currently fight over one device).
- Add timeouts + retries on subprocess calls (none exist today).
- Don't assume score scales are comparable (Vina kcal/mol vs Boltz probabilities vs Chai ptm/iptm vs PLIP counts).
