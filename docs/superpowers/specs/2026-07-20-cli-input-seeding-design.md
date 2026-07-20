# CLI Input Seeding — Design

Date: 2026-07-20
Status: Approved (design), ready for implementation plan

## Problem

The config-driven CLI (`structurezyme run --config run.yml`) constructs
`Runner(cfg)` and runs immediately, but nothing supplies the input data. The
first step (`squidly`) reads `checkpoints/_input.pkl` via `_seed_input`
(`structurezyme/step_runners.py:57-60`), and only the **legacy**
`Pipeline(df=...)` adapter (`structurezyme/pipeline.py`) ever writes that file.
`RunConfig`/`PathsConfig` have no input-data field.

Result: a fresh `structurezyme run --config run.yml` crashes at squidly with a
missing `_input.pkl`. The config-driven CLI is not independently runnable,
which blocks the Phase F all-modules GPU run.

## Goal

Make the config-driven CLI independently runnable by declaring the input data
in the config and seeding `checkpoints/_input.pkl` at the run boundary, with
early validation of the columns the **enabled** modules require.

## Input data model

The input is **tabular only** (one row per protein/substrate). It never
contains structure files.

Base columns (always required, for squidly/chai/boltz):

| Column             | Meaning                                   |
|--------------------|-------------------------------------------|
| `Sequence`         | Protein sequence                          |
| `substrate_smiles` | Small-molecule / substrate SMILES         |
| `Entry`            | UniProt accession (used widely, e.g. AF2) |

Module-conditional columns (required only if that module is enabled):

| Enabled module     | Extra required column |
|--------------------|-----------------------|
| `vina`             | `vina_residues`       |
| `geometric_filter` | `substrate_moiety`    |

Other columns used opportunistically by some steps (`substrate_name`,
`cofactor_smiles`, `cofactor_moiety`) are recommended but not hard-required;
they may be empty / `None`. Module **behavior** knobs (fastrelax mode,
`placer_predict_ligand`, thresholds, etc.) live in `run.yml`
(`opts.get(...)`), NOT in the input file.

## Changes

### 1. Config (`structurezyme/config.py`)
- Add `PathsConfig.input_csv: str = ""` — path to the input file.
- `validate_paths()` also requires `input_csv` non-empty (alongside
  `output_root`, `boltz_cache_dir`).
- Add `input_csv` to the `init` template (`cli._template`).

### 2. Runner seeding (`structurezyme/runner.py`)
- Before running steps, `Runner.run()` seeds the input frame:
  - Resolve `checkpoints/_input.pkl` via `layout.checkpoint_path("_input")`.
  - **Resume-safe / idempotent:** if `_input.pkl` already exists, leave it
    untouched (downstream steps hashed against it; a resume must not clobber
    it). Only read `input_csv` and write the pickle when absent.
  - Load `input_csv`: `.csv` -> `pandas.read_csv`; `.pkl`/`.pickle` ->
    `pandas.read_pickle`. (Pickle preserves `None`/list columns such as
    `cofactor_moiety` that CSV round-trips poorly.)

### 3. Enabled-module column validation (`structurezyme/runner.py`)
- After loading the frame (fresh run) or the existing `_input.pkl` (resume),
  before running steps, validate required columns for the enabled modules and
  raise a clear error listing what is missing, e.g.
  `"vina enabled but input is missing required column(s): vina_residues"`.
- Structure-dependent modules (`fastrelax`, `placer`, `superimpose`) are NOT
  validated by input columns — their structures come from upstream checkpoints
  (`*_files_for_superimposition`), guarded by DAG/checkpoint presence.

### 4. Docs
- Add the input-column requirement tables above to `docs/configuration.md`
  (or `docs/getting_started.md`), plus the note that behavior knobs live in
  `run.yml`.
- Add a short "Bring-your-own-PDB single-module runs" section marked
  **Future / not yet implemented** (see below).

## Tests (`tests/`)
1. Fresh run seeds `_input.pkl` from a CSV.
2. Fresh run seeds `_input.pkl` from a `.pkl`.
3. Resume with an existing `_input.pkl` does NOT overwrite it.
4. Missing `input_csv` -> `validate_paths` raises.
5. `vina` enabled but `vina_residues` column missing -> raises with clear msg.
6. `geometric_filter` enabled but `substrate_moiety` missing -> raises.

## Out of scope (YAGNI)
- No CSV schema/dtype validation beyond required-column presence.
- No multi-file merge, no inline-in-YAML data.

## Future feature (documented, NOT built now)

**Bring-your-own-PDB single-module runs.** A user with an existing PDB
structure (e.g. from RCSB or AlphaFold DB) wants to run ONE downstream module
(fastrelax / placer / geometric_filter) directly on it, skipping the Chai/Boltz
docking pipeline. This is more involved: each module reads structures from
engine-specific checkpoint columns (e.g. FastRelax requires the
`*_files_for_superimposition` list-columns — `fastrelax_step.py:167-178`), so a
per-module "PDB -> expected columns" adapter must be defined. Captured here as
a follow-up; it will get its own brainstorm/spec later.
