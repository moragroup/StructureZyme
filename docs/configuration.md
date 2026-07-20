# Configuration Reference

StructureZyme is driven by a single YAML file describing a `RunConfig`. Generate
a template with:

```bash
structurezyme init --output run.yml
```

then edit it and run with `structurezyme run --config run.yml`. The config has
three top-level sections: `paths`, `runtime`, and `steps`. All three are
validated with Pydantic v2 (`structurezyme/config.py`).

Programmatically:

```python
from structurezyme.config import load_config
cfg = load_config("run.yml")   # -> RunConfig
```

## `paths` (`PathsConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `output_root` | `""` | Root directory for run outputs. Runs are written to `{output_root}/{user}/{run_id}/`. Required at run time (may be left empty and filled by a host profile — see [multi_user.md](multi_user.md)). |
| `boltz_cache_dir` | `""` | Directory for the shared Boltz model cache. Required at run time (may be filled by a host profile). |
| `input_csv` | `""` | Path to the input data file (`.csv` or pandas `.pkl`/`.pickle`) that seeds the run. Required at run time. See [Input data](#input-data-input_csv) below. |
| `squidly_weights_dir` | `null` | Directory holding Squidly model weights. Optional; when unset, Squidly uses its packaged default. |
| `placer_env_path` | `/mnt/labs/data/mora/software/PLACER/env` | Path to the PLACER conda environment (only used when the `placer` step is enabled). |

`output_root`, `boltz_cache_dir`, and `input_csv` are checked by
`RunConfig.validate_paths()` at the run boundary (inside `Runner`), which raises
if any is still empty. `output_root` and `boltz_cache_dir` default to `""` so a
host profile can fill them (host profiles do NOT fill `input_csv` — it is
per-run data you set yourself). See [multi_user.md](multi_user.md) for host
profiles.

## `runtime` (`RuntimeConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `user` | current OS user | Namespaces the run directory: `{output_root}/{user}/{run_id}/`. |
| `run_id` | timestamp `YYYYMMDD-HHMMSS` | Unique id for this run; the second path component under `user`. |
| `num_threads` | `1` | Default worker-thread count passed to steps that parallelize. |
| `force` | `[]` | List of step names to force-rerun even if a valid checkpoint exists. A **disabled** step is not run even if forced. |

## `steps` (`StepsConfig`)

Every step is a `StepConfig` with at least an `enabled` flag. `StepConfig` uses
`extra="allow"`, so each step may carry additional option keys beyond `enabled`;
the runner reads those via `RunConfig.step_options(name)`.

The 15 steps and their default enabled state:

| Step | Enabled by default | Role |
|------|--------------------|------|
| `squidly` | yes | Predict catalytic residues. |
| `chai` | yes | Chai structure prediction (writes `.cif`). |
| `boltz` | yes | Boltz structure prediction (writes `.cif`). |
| `vina` | **no** | AutoDock Vina docking. |
| `docking_metrics` | yes | Extract docking-quality metrics. |
| `prepare_files` | yes | Prepare/convert docked structures for downstream filters. |
| `fastrelax` | **no** | PyRosetta FastRelax of the top-K poses per engine. |
| `superimpose` | yes | Superimpose structures. |
| `protein_rmsd` | yes | Protein RMSD. |
| `ligand_rmsd` | yes | Ligand RMSD + best-pose selection. |
| `geometric_filter` | yes | Geometric filtering. |
| `fpocket` | yes | Pocket detection (fpocket). |
| `ligand_sasa` | yes | Ligand solvent-accessible surface area. |
| `plip` | yes | Protein–ligand interaction profiling (PLIP). |
| `placer` | **no** | PLACER ligand-pose prediction. |

`vina`, `fastrelax`, and `placer` are **off by default**; enable them with
`{enabled: true}` (plus any options below).

### Notable per-step options

These extra keys are read by `structurezyme/step_runners.py`. Any key not listed
is simply ignored by that step (but preserved in the written config).

**`squidly`**

| Option | Default | Meaning |
|--------|---------|---------|
| `skip_catalytic_residue_prediction` | `false` | Skip the prediction entirely. |
| `squidly_model_size` | `"3B"` | Squidly model size. |
| `squidly_as_threshold` | `null` | Active-site score threshold. |
| `squidly_num_threads` | `runtime.num_threads` | Thread override for this step. |

**`boltz`**

| Option | Default | Meaning |
|--------|---------|---------|
| `use_msa_server` | `true` | Add `--use_msa_server` to the Boltz invocation. |

**`vina`**

| Option | Default | Meaning |
|--------|---------|---------|
| `metagenomic_enzymes` | `0` | Set to `1` to use a fallback structure when AF2 structures are missing. |
| `alternative_structure_for_vina` | `"Boltz"` | Fallback engine (`"Boltz"` or `"Chai"`) for docking when structures are missing. |

**`fastrelax`**

| Option | Default | Meaning |
|--------|---------|---------|
| `fastrelax_top_k` | `2` | Number of top poses **per engine** (ranked by confidence) to relax. |
| `fastrelax_mode` | `"ligand_focused"` | Relaxation mode. |
| `fastrelax_drop_unrelaxed` | `true` | Drop poses that were not relaxed. |
| `fastrelax_shell_radius` | `8.0` | Relaxation shell radius (Å). |
| `fastrelax_constraint_weight` | `1.0` | Coordinate-constraint weight. |
| `fastrelax_scorefunction` | `"ref2015"` | Rosetta score function. |
| `ligand_resname` | `"LIG"` | Ligand residue name. |

**`ligand_rmsd`**

| Option | Default | Meaning |
|--------|---------|---------|
| `max_matches` | `1000` | Max substructure matches when computing ligand RMSD. |

**`geometric_filter`**

| Option | Default | Meaning |
|--------|---------|---------|
| `esterase` | `0` | Set to `1` for esterase-specific geometric criteria. |

**`placer`**

| Option | Default | Meaning |
|--------|---------|---------|
| `placer_predict_ligand` | `null` | Ligand to predict. |
| `placer_env_path` | `paths.placer_env_path` | Override the PLACER environment path. |
| `placer_nsamples` | `50` | Number of PLACER samples. |
| `placer_rerank` | `"prmsd"` | Rerank criterion. |

> **Note on "best N" / structure outputs.** Chai and Boltz write **all**
> generated model poses to disk as `.cif` (mmCIF) files; nothing is deleted.
> The `fastrelax_top_k` option (default 2) only controls how many poses **per
> engine** are carried into FastRelax — it is a downstream selection, not a save
> filter. When `fastrelax` is disabled (the default), no top-K pruning happens
> and downstream steps operate on the structures selected by `prepare_files`.

## Examples

### Minimal

```yaml
paths:
  output_root: /mnt/labs/data/mora/structurezyme_runs
  boltz_cache_dir: /mnt/labs/data/mora/boltz_cache
runtime:
  num_threads: 4
steps:
  vina: {enabled: false}
```

### With Vina docking

```yaml
paths:
  output_root: /mnt/labs/data/mora/structurezyme_runs
  boltz_cache_dir: /mnt/labs/data/mora/boltz_cache
runtime:
  num_threads: 8
steps:
  vina:
    enabled: true
    metagenomic_enzymes: 1
    alternative_structure_for_vina: Boltz
```

### With a cofactor

Provide `cofactor_smiles` in your input DataFrame; the docking steps pick it up
automatically. No special config key is required, but you typically pair it with
esterase-style geometric criteria:

```yaml
paths:
  output_root: /mnt/labs/data/mora/structurezyme_runs
  boltz_cache_dir: /mnt/labs/data/mora/boltz_cache
steps:
  geometric_filter:
    enabled: true
    esterase: 1
```

### With FastRelax and PLACER

```yaml
paths:
  output_root: /mnt/labs/data/mora/structurezyme_runs
  boltz_cache_dir: /mnt/labs/data/mora/boltz_cache
  placer_env_path: /mnt/labs/data/mora/software/PLACER/env
runtime:
  num_threads: 8
steps:
  fastrelax:
    enabled: true
    fastrelax_top_k: 2
    fastrelax_mode: ligand_focused
  placer:
    enabled: true
    placer_predict_ligand: LIG
    placer_nsamples: 50
    placer_rerank: prmsd
```

See also [pipeline_overview.md](pipeline_overview.md) for the step graph and
[resume_and_checkpoints.md](resume_and_checkpoints.md) for how enabled/disabled
steps interact with checkpoints and `--force`.

## Input data (`paths.input_csv`)

The `run` command reads its input from the file at `paths.input_csv` and seeds
`checkpoints/_input.pkl` on the first run (a resume never overwrites an existing
seed). The file may be a `.csv` or a pandas `.pkl`/`.pickle`. Use a pickle when
you need `None`/list-typed columns (e.g. `cofactor_moiety`) preserved exactly.

The input is **tabular only** — it never contains structure files. Module
*behavior* settings (fastrelax mode, `placer_predict_ligand`, thresholds, …)
live in `run.yml`, not in the input file.

Always-required columns:

| Column             | Meaning            |
|--------------------|--------------------|
| `Sequence`         | Protein sequence   |
| `substrate_smiles` | Substrate SMILES   |
| `Entry`            | UniProt accession  |

Required only when the module is enabled:

| Enabled module     | Extra required column |
|--------------------|-----------------------|
| `vina`             | `vina_residues`       |
| `geometric_filter` | `substrate_moiety`    |

Recommended (used opportunistically, may be empty/`None`): `substrate_name`,
`cofactor_smiles`, `cofactor_moiety`.

If an enabled module's required column is missing, `run` fails immediately with
a message naming the module and the missing column(s), before any GPU work.

### Bring-your-own-PDB single-module runs (future — not yet implemented)

Running a single downstream module (e.g. `fastrelax`, `placer`,
`geometric_filter`) directly on a PDB you already have — skipping Chai/Boltz
docking — is **not yet supported**. Those modules read structures from upstream
checkpoint columns (e.g. FastRelax needs the `*_files_for_superimposition`
columns), so a per-module "PDB → expected columns" adapter is required. This is
planned as a follow-up feature.


