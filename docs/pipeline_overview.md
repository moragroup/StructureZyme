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

Three selection methods vote independently, each recorded in `best_method`:

1. **`inter_tool_weighted_avg`** — for each pose, average RMSD to every pose
   of every *other* tool, weighted by that tool's pose count. Pick the min.
2. **`inter_tool_min_per_tool`** — for each pose, take the *closest* pose per
   other tool and average those minima. Pick the min.
3. **`vina_avg_intra_tool`** — among vina poses only, pick the one with the
   lowest mean RMSD to the other vina poses. Requires ≥2 vina poses.

### Known design gap: selector ignores `fastrelax_score`

The current selector uses **only inter-tool geometric consensus**. It does
not consult `fastrelax_score` (the Rosetta interaction energy computed by
the `fastrelax` step) or docking-engine confidence metrics. Two consequences
observed on real runs:

- When a run enables `chai` + `boltz` but not `vina`, both consensus methods
  reduce to "closest to the poses of the other tool." If one tool contributes
  many more poses than the other (e.g. 4 chai poses vs 1 boltz pose), the
  majority tool's poses cluster near their own mean and the pose closest to
  the minority tool's single point wins both methods — deterministically.
- A pose with a much better `fastrelax_score` can lose to a
  geometrically-central pose with a worse energy. On `fmo-fad-01` this
  produced `tool="chai"` on all 18 entries in `placer.pkl`, even for entries
  where the single boltz pose had a substantially lower (better)
  `fastrelax_score` than any chai pose.

**Status**: not a bug — the behavior is deterministic and matches the
docstring. But the selector should be redesigned to combine geometric
consensus with `fastrelax_score` (and possibly engine confidence) rather
than relying on consensus alone. Options range from adding a fourth
`fastrelax_min` method (small change; gives energy a vote via
`_method_count`) to a rank-fusion / weighted-score hybrid (larger change;
requires design discussion). Any redesign must handle the single-pose-per-
tool case explicitly, since intra-tool consensus is undefined there.

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
