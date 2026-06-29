# 2. Current State / Architecture

## What it does

Filterzyme is a structural-filtering pipeline whose input is a `pandas.DataFrame` of enzyme candidates plus substrate metadata, and whose output is a per-pose feature table suitable for ranking and downstream wet-lab triage. The README documents the input contract as:

| Column | Type | Required | Purpose |
|---|---|---|---|
| `Entry` | str | yes | Unique enzyme identifier |
| `Sequence` | str | yes | Amino-acid sequence |
| `substrate_name` | str | yes | Substrate label |
| `substrate_smiles` | str | yes | Substrate SMILES |
| `substrate_moiety` | str | yes | SMARTS pattern marking the reactive moiety |
| `cofactor_name` | str | optional | Cofactor label |
| `cofactor_smiles` | str | optional | Cofactor SMILES |
| `cofactor_moiety` | str | optional | SMARTS pattern for cofactor moiety |
| `vina_residues` | str / list | optional | User-specified active-site residues (overrides Squidly) |

None of these is validated at runtime; the absence of a required column surfaces several stages downstream as a confusing `KeyError`.

## Three stages

The pipeline (`pipeline_v2.py:351` â `Pipeline.run`) executes three top-level classes in sequence. Each class is also independently runnable, which is how the benchmarking scripts use them.

### Stage 1 â `Docking` (`pipeline_v2.py:42`)

1. **Active-site prediction.** `_catalytic_residue_prediction` (`pipeline_v2.py:78`) deduplicates by sequence, calls `enzymetk.ActiveSitePred` (Squidly LSTM over ESM2), merges results back, normalises `Squidly_CR_Position` and optional user-supplied `vina_residues` into a single pipe-delimited `catalytic_residues` string, and drops entries that have neither. `pipeline.py:71` uses the local `filterzyme.steps.predict_catalyticsite_step.ActiveSitePred` (subprocess to a separate conda env) instead â the two versions diverge here.
2. **Structure prediction (Chai-1).** `_run_chai` (`pipeline_v2.py:127`) calls `enzymetk.Chai` on `(Sequence, substrate_smiles, cofactor_smiles)`. Outputs land under `<output_dir>/chai/<Entry>/`.
3. **Structure prediction (Boltz-2).** `_run_boltz` (`pipeline_v2.py:138`) calls `enzymetk.Boltz` with the same triple. Outputs land under `<output_dir>/boltz/<Entry>/`.
4. **Physics-based docking (Vina).** `_run_vina` (`pipeline_v2.py:150`) calls the local `filterzyme.steps.dock_vina_step.Vina`, which wraps the external `docko` package. If a structure file is not provided, the wrapper attempts to fetch an AlphaFold2 structure (`dock_vina_step.py:44 get_alphafold_structure(...)`); on failure or when `metagenomic_enzymes == 1`, it falls back to the Chai or Boltz prediction (selected by `alternative_structure_for_vina`).
5. **Metric extraction.** `_extract_docking_quality_metrics` (`pipeline_v2.py:210`) parses Vina log `.txt`, Chai `.npz`, and Boltz `.json` confidence files into per-entry dicts via `DockingMetrics(Step)` at `extract_docking_metrics_step.py:125`.

### Stage 2 â `Superimposition` (`pipeline_v2.py:~220`)

1. **Prepare files.** `PrepareVina`, `PrepareChai`, `PrepareBoltz` (in `steps/preparevina_step.py:163`, `preparechai_step.py:155`, `prepareboltz_step.py:64`) normalise per-tool outputs into a flat per-entry directory tree under `superimposed_structures/`.
2. **Pairwise homologous superposition.** `_superimposition` (`pipeline_v2.py:248`) runs three pairwise `SuperimposeStructures` passes (vina-chai, vina-boltz, chai-boltz) backed by biotite's `superimpose_homologs` (see `superimposestructures_step.py:236`).
3. **Protein RMSD.** `_proteinRMSD` (`pipeline_v2.py:262`) calls `ProteinRMSD` (`computeproteinRMSD_step.py:177`) and returns a **tuple** `(df_proteinRMSD_pairwise, df_proteinRMSD)`. `pipeline.py:252` does the same but returns a single DataFrame â the two pipelines disagree on shape.
4. **Ligand RMSD + best-structure selection.** `_ligandRMSD` (`pipeline_v2.py:272`) calls `LigandRMSD` (`computeligandRMSD_step.py:370`), then in v2 calls `extract_docking_metrics` to enrich (line 278). In v1 it instead calls `add_metrics_to_best_structures` and reads pickles from `Path(self.output_dir).parent / 'docking/dockingmetrics.pkl'` (`pipeline.py:265` â brittle path assumption).

### Stage 3 â `GeometricFilters` (`pipeline_v2.py:284`)

1. **Geometric filtering.** Either `GeneralGeometricFiltering` (from `geometric_filtering_cofactor_MCS.py:435` in v2, `geometric_filtering_cofactor.py:450` in v1) or `EsteraseGeometricFiltering` (`geometric_filtering_esterase.py:418`), gated by the `esterase` int flag.
2. **Active-site volume.** `Fpocket(Step)` at `fpocket_step.py:232` â shells out to the `fpocket` binary.
3. **Ligand surface exposure.** `LigandSASA(Step)` at `ligandSASA_step.py:81` â calls FreeSASA.
4. **Protein-ligand interactions.** `PLIP(Step)` at `plip_step.py:81` â calls the PLIP Python API.

The final pickle is written to `<base_output_dir>/geometricfiltering/structural_features_final.pkl` (`pipeline_v2.py:309`).

## Operator pattern

`filterzyme/steps/step.py` defines the operator-overloading mini-DSL: `Step >> Step` returns a `Pipeline`, and `DataFrame << Pipeline` (or `<< Step`) triggers `.execute(df)`. The intent is to allow expression-style composition like:

```python
df_chai = df_squidly << (Chai(...) >> Save(...))
```

The file is messy. Lines 1â19 define `Pipeline`. Lines 21â49 are a triple-quoted block that contains two more `Pipeline` class definitions and the placeholder docstring `""" Execute some shit """` â harmless because it's a string, but signals abandoned refactoring. Lines 51â64 define a `Step` class with the operator dunders. Lines 66â74 define a **second** `Step` class that shadows the first; both expose `execute`/`__rshift__`/`__rlshift__`, so the operators still work, but the duplicate definition is a hazard for anyone extending `Step`. See `06_code_quality.md` for the fix.

## Module map

```
StructureZyme/
  README.md                          â README; example uses pipeline.py (v1)
  setup.py                           â has ppython_requires typo (line 50)
  environment.yml                    â lists filterzyme as a pip dep of its own env
  submit_pipeline.sh                 â calls a non-existent run_pipeline_on_multiple_substrates.py
  test_pipeline.py                   â imports the long-renamed `filtering_pipeline`
  examples/DEHP-MEHP.pkl             â the only example input
  benchmarking/
    martinez/{N1_Benchmark_martinez.ipynb, run_martinez.py}
    serine_hydrolases/{N2_Benchmark_serine_hydrolases.ipynb, run_serine_hydrolases.py}
  tests/                             â notebooks only, no pytest
  filterzyme/
    __init__.py                      â single line: __version__ = "0.0.6"
    pipeline.py                      â v1 entry, uses local Squidly subprocess
    pipeline_v2.py                   â v2 entry, uses enzymetk.ActiveSitePred (canonical)
    squidly_final_models/            â pre-trained Squidly weights + a checked-in .DS_Store
    steps/
      step.py                        â Step / Pipeline base (with shadowing)
      save_step.py                   â pickle-to-disk passthrough
      predict_catalyticsite_step.py  â local Squidly wrapper (subprocess to AS_inference env)
      predict_catalyticsite_run.py   â inner subprocess entry point (uses os.system + conda run)
      dock_vina_step.py              â wraps docko; tries AF2 fetch, falls back to Chai/Boltz
      extract_docking_metrics_step.py â parses Vina/Chai/Boltz scores into per-entry dicts
      preparevina_step.py            â flatten Vina outputs for superposition
      preparechai_step.py            â flatten Chai outputs
      prepareboltz_step.py           â flatten Boltz outputs
      superimposestructures_step.py  â biotite superimpose_homologs wrapper
      computeproteinRMSD_step.py     â pairwise protein RMSD + heatmap
      computeligandRMSD_step.py      â pairwise ligand RMSD + heatmap (4 helpers duplicated from utils/helpers.py)
      geometric_filtering_cofactor.py        â 595 LOC, has duplicated helpers
      geometric_filtering_cofactor_MCS.py    â 580 LOC, uses utils/helpers properly
      geometric_filtering_esterase.py        â 555 LOC, no helper duplication
      fpocket_step.py                â fpocket binary wrapper
      ligandSASA_step.py             â FreeSASA wrapper
      plip_step.py                   â PLIP wrapper
      cleanPDB_step.py               â PDB cleanup utility
      PLACER_step.py                 â DEAD: never imported
      PLACER_forChai_step.py         â DEAD: never imported
      PLACER_forVina_step.py         â DEAD: never imported
    utils/
      helpers.py                     â 558 LOC; logging, path generators, dedup helpers, broken add_metrics
```

## Dependency surface

External packages that must be present and reachable at the same time:

- `enzymetk` (pip) â provides `Chai`, `Boltz`, `ActiveSitePred` (v2 only). Origin not stated in this repo.
- `docko` (pip) â provides `dock`, `get_alphafold_structure`, `pdb_to_pdbqt_protein`, `clean_one_pdb`. Origin not stated.
- `chai_lab` (pip) â Chai-1 inference.
- Boltz-2 â pulled in by `enzymetk` (no direct import here).
- `biotite`, `rdkit`, `Bio.PDB`, `freesasa`, `plip` â core science stack.
- `fpocket`, `openbabel`, `pdbfixer`, `openmm` â binary / system deps via conda.
- A separate conda env named `AS_inference` is required by `predict_catalyticsite_run.py:19` for the local-Squidly path used by v1.
- AlphaFold2 cached structures are fetched on-demand by `docko.get_alphafold_structure` at `dock_vina_step.py:44`.

The dependency graph spans three Python environments in the worst case (the host env, `AS_inference`, and whatever conda env `chai_lab` installs into), invoked transitively via `subprocess` and `os.system`. This is the single biggest source of opaque failures in the codebase.
