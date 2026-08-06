# Pipeline Overview

Structurezyme runs as a **modular step graph**. Each step is a self-contained
module registered in `structurezyme/registry.py`. The `Runner`
(`structurezyme/runner.py`) walks the graph in topological order, checkpoints
every step to disk, and records progress in a `manifest.json` so that runs are
fully resumable.

## The step graph

There are **15 steps**. Their dependencies are declared in the `STEPS` registry
(`structurezyme/registry.py`) via `depends_on`, and executed in the order
returned by `ordered_steps()` (a topological sort of the graph):

```
squidly
  └─> chai
        └─> boltz
              ├─> vina
              └─> docking_metrics        (inputs: boltz, vina)
                    └─> prepare_files
                          ├─> fastrelax
                          └─> superimpose (depends_on: prepare_files, fastrelax)
                                └─> protein_rmsd
                                      └─> ligand_rmsd
                                            └─> geometric_filter
                                                  └─> fpocket
                                                        └─> ligand_sasa
                                                              └─> plip
                                                                    └─> placer
```

### Dependency / input table

| # | Step | `depends_on` | `inputs` |
|---|------|--------------|----------|
| 1 | `squidly` | — | — |
| 2 | `chai` | `squidly` | `squidly` |
| 3 | `boltz` | `chai` | `chai` |
| 4 | `vina` | `boltz` | `boltz` |
| 5 | `docking_metrics` | `boltz` | `boltz`, `vina` |
| 6 | `prepare_files` | `docking_metrics` | `docking_metrics` |
| 7 | `fastrelax` | `prepare_files` | `prepare_files` |
| 8 | `superimpose` | `prepare_files`, `fastrelax` | `prepare_files`, `fastrelax` |
| 9 | `protein_rmsd` | `superimpose` | `superimpose` |
| 10 | `ligand_rmsd` | `protein_rmsd` | `protein_rmsd` |
| 11 | `geometric_filter` | `ligand_rmsd` | `ligand_rmsd` |
| 12 | `fpocket` | `geometric_filter` | `geometric_filter` |
| 13 | `ligand_sasa` | `fpocket` | `fpocket` |
| 14 | `plip` | `ligand_sasa` | `ligand_sasa` |
| 15 | `placer` | `plip` | `plip` |

The linear order produced by `ordered_steps()` is:

```
squidly → chai → boltz → vina → docking_metrics → prepare_files →
fastrelax → superimpose → protein_rmsd → ligand_rmsd → geometric_filter →
fpocket → ligand_sasa → plip → placer
```

## Steps that are disabled by default

Every step is enabled unless overridden in the config
(`structurezyme/config.py`). Three steps ship **disabled** (`enabled=False`)
and must be turned on explicitly to run:

- `vina`
- `fastrelax`
- `placer`

All other steps are enabled by default.

## Docking engines and structure files

The two structure-prediction docking engines are **Chai** and **Boltz**. Both
write **mmCIF (`.cif`)** structure files, and *every* generated model pose is
kept on disk — nothing is deleted. The rank-0 model of each is located by the
helpers in `structurezyme/utils/helpers.py`:

- `generate_chai_structure_path` → `.../chai/<name>_0.cif`
- `generate_boltz_structure_path` → `.../boltz_results_<name>/predictions/<name>/<name>_model_0.cif`

`vina` (optional) provides an additional docking engine whose results feed into
`docking_metrics` alongside `boltz`.

## FastRelax step

`fastrelax` is **disabled by default**. When enabled it selects the top
`top_k` poses **per engine** (default `top_k = 2`) by confidence and relaxes
only those with PyRosetta's FastRelax mover. Ranking direction is
engine-dependent (`_select_top_k` in `structurezyme/steps/fastrelax_step.py`):

- **descending** for `chai` / `boltz` (higher confidence is better)
- **ascending** for `vina` (more negative affinity is better)

Poses that are not selected are **not deleted** — they are simply not carried
into relaxation.

`superimpose` depends on both `prepare_files` and `fastrelax`. Declaring
`fastrelax` in `depends_on` enforces ordering: when `fastrelax` is enabled,
`superimpose` consumes its relaxed frame; when `fastrelax` is disabled it is
skipped, and `superimpose` consumes the `prepare_files` frame instead.

## Ligand RMSD & best-pose selection

`ligand_rmsd` (`structurezyme/steps/computeligandRMSD_step.py`) computes
pairwise ligand RMSDs between all docked poses per entry and then calls
`select_best_docked_structures` to flag one or more "best" poses per entry.
Downstream steps that consume a single pose per entry (PLIP, ligand SASA,
fpocket, PLACER) read the `is_best` / `best_method` columns produced here;
`PLACER_step._select_one_per_entry` further reduces to exactly one row.

Four selection methods vote independently, each recorded in `best_method`:

1. **`inter_tool_weighted_avg`** — for each pose, average RMSD to every pose
   of every *other* tool, weighted by that tool's pose count. Pick the min.
2. **`inter_tool_min_per_tool`** — for each pose, take the *closest* pose per
   other tool and average those minima. Pick the min.
3. **`vina_avg_intra_tool`** — among vina poses only, pick the one with the
   lowest mean RMSD to the other vina poses. Requires ≥2 vina poses.
4. **`fused_rank`** — energy-aware fusion: rank poses by
   `inter_tool_min_per_tool` geometry and by `fastrelax_score` (lower energy =
   better), combine as summed dense ranks (lowest wins). When geometry is
   degenerate (single-pose-per-tool / <2 tools) energy dominates; when energy
   is unavailable it falls back to the pure-geometry `inter_tool_min_per_tool`
   pick. Emitted once per entry that has any selectable pose.

### Energy-aware selection (`fused_rank`)

The selector now fuses inter-tool geometric consensus with the Rosetta
`fastrelax_score` interaction energy via the `fused_rank` method. Energy is
passed per-Entry from `LigandRMSD.__execute` into
`select_best_docked_structures`, and PLACER's `_select_one_per_entry` treats a
`fused_rank` pick as authoritative (falling back to the geometric
`_method_count` vote only when `fused_rank` is absent, e.g. fastrelax
disabled).

Behavior:

- **Normal case** (>=2 tools, minority tool has >=2 poses, >=2 poses relaxed):
  `fused_score = geom_rank + energy_rank`; poses without energy get the worst
  energy rank so geometry still orders them.
- **Degenerate geometry** (<2 tools, or 2 tools with a single-pose minority):
  energy dominates, geometry breaks ties.
- **No energy** (fastrelax off / all relax failed): pure geometry
  (`inter_tool_min_per_tool`), identical to the previous behavior.

Score direction: `fastrelax_score` is ranked ascending (lower/more-negative
REU = better) for all engines.

**Future work:** 3-tool outlier trimming (best-2-of-3-tools consensus) is not
yet implemented; add only if a real 3-tool run shows geometry distortion.

## PLACER step

`placer` is the final step in the graph, depending on `plip`. It is
**disabled by default** and uses the PLACER environment configured via
`paths.placer_env_path` in the config.

## Checkpoints and resume

Every step writes a pickled DataFrame checkpoint to
`checkpoints/<step>.pkl` under the run directory (see
`RunLayout.checkpoint_path`). Progress is tracked in `manifest.json`, where
each step gets a `StepRecord` (`structurezyme/manifest.py`) with fields
including:

- `status` — one of `PENDING`, `RUNNING`, `OK`, `FAILED`,
  `SKIPPED_CHECKPOINT`, `SKIPPED_DISABLED`
- `input_hash` — hash of the existing input checkpoint files
- `config_hash` — hash of that step's config options
- `wall_time_s`, `mem_mb`, `started_at`, `finished_at`, `error`, `pkl_path`

### Skip / resume rules

`Runner._should_skip` decides, for each step, whether to run it:

- **`SKIPPED_DISABLED`** — the step is disabled in config. It is **never run**,
  even if forced (the enabled check happens *before* the force check).
- **`SKIPPED_CHECKPOINT`** — the step is enabled, its output checkpoint is
  present, its manifest record is `OK`, **and** both `config_hash` and
  `input_hash` match the current inputs/config. The existing `OK` record is
  preserved so resumability survives repeated runs.
- Otherwise the step **runs** (missing checkpoint, no record, or a changed
  config/input hash).

To re-run a specific step, add its name to `runtime.force` in the config; a
forced step re-runs even if a valid checkpoint exists — **except** a disabled
step, which is still skipped.
