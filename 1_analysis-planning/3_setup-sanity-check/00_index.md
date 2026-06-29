# Setup & Sanity-Check Plan

Goal (user's words): "get this repository running so I can do a sanity check, to determine what works, where it breaks and how to properly install all the other models in the future and implement them."

This folder is the executable plan for the **first run** of Filterzyme/StructureZyme on the HPC cluster. It is split into self-contained task files so multiple agents can each own one file.

## Environment facts (verified)

- **Cluster:** Slurm. Login node `login02`. No `module` system, no `conda`/`mamba`/`micromamba` anywhere on PATH, no GPU on the login node.
- **Slurm account:** `mora`. QOS available: `debug`, `normal`, `spot`.
- **Partitions:**
  - `cpu` — no GPU, 4-day limit.
  - `gpu` — RTX6000 (8/node, 24 GB each), 4-day limit. **← chosen target for sanity check.**
  - `h100` — H100 NVL (2/node, ~94 GB), 4-day limit.
  - `spot` — H100 + RTX6000, 10-day limit, preemptible.
- **Login node host:** Ubuntu 24.04, 16 cores, 62 GB RAM, `/usr/bin/python3` = 3.12 (do NOT use for the env; repo wants 3.11).
- **Home disk:** `/mnt/storage01/home/lherrmann` (`/tank/home`), 34 TB free. Safe for Miniforge + model weights.
- **Repo root:** `/mnt/storage01/home/lherrmann/StructureZyme`.

## Decisions locked in by the user

1. **Interactive GPU session** required first (user is on the login node). Target partition: **`gpu` (RTX6000, 24 GB)**.
2. **Environment manager: Conda/Miniforge** (NOT uv). Reason: repo needs bioconda-only binaries (`foldseek`, `mmseqs2`, `fpocket`, `openbabel`, `plip`) plus AutoDock Vina executable, which uv/PyPI cannot provide. Install Miniforge to `$HOME`, build the `filterpipeline` env from `environment.yml`.
3. **Sanity-check scope: staged** — install → imports → full end-to-end run on `examples/DEHP-MEHP.pkl`, documenting exactly which stage breaks and why.

## Canonical entry point

`filterzyme/pipeline.py` (the merged/consolidated pipeline; old `pipeline_v2.py` no longer exists, its content is now `pipeline.py`). `outdated/pipeline_v1.py` is retired. The README example and both `benchmarking/*/run_*.py` import `from filterzyme.pipeline import Pipeline`.

Sanity-check input: **`examples/DEHP-MEHP.pkl`** (210 KB, the only committed example). The benchmark input pickles (`martinez_input_df.pkl`, `serine_hydrolases_input_df.pkl`) are NOT in the repo.

## Task files (run order)

| # | File | Owner-able unit | Depends on |
|---|------|-----------------|------------|
| 1 | `01_interactive_session.md` | Get an interactive RTX6000 Slurm session | — |
| 2 | `02_install_miniforge.md` | Install Miniforge to `$HOME` | 1 |
| 3 | `03_build_env.md` | Build `filterpipeline` conda env + binaries | 2 |
| 4 | `04_install_python_pkgs.md` | torch (CUDA) + `pip install -e .` + enzymetk/docko/chai/boltz | 3 |
| 5 | `05_enzymetk_env_issue.md` | Resolve enzymetk `env_name` / tool-env requirement (BIGGEST RISK) | 4 |
| 6 | `06_import_sanity.md` | Import-level checks for every module | 4 |
| 7 | `07_staged_run.md` | Run Docking → Superimposition → GeometricFilters on the example | 5,6 |
| 8 | `08_report_template.md` | Fill in the "what works / where it breaks" report | 7 |
| 9 | `09_future_model_install.md` | Recipe for installing/adding future models | 7 |

## Known-bug cheat sheet (from `../05_bugs_by_severity.md`)

These are likely to surface during the staged run. Do NOT fix them as part of the sanity check (that is roadmap Phase 0/1) — just note when they fire:

- **P0-3** `helpers.add_metrics` raises `NameError` (`dict_columns`, `extract_vina_index` undefined). Imported by `pipeline.py`. The working sibling is `extract_docking_metrics`.
- **P1-3** `dock_vina_step.py:35-36` `int(r)+1` crashes on empty residue strings.
- **CQ-15** `generate_chai_structure_path` / `generate_boltz_structure_path` hard-code upstream on-disk layout; breaks if Chai/Boltz output dirs differ from expectation.
- **PLACER_step.py:8** `from step import Step` — bad import (dead code), only fires if something imports it.

## Out of scope for this sanity check

- Fixing pipeline bugs (roadmap Phase 0/1 in `../10_roadmap.md`).
- DiffDock / TRILL env.
- Multi-GPU, checkpointing, CLI, config schema.
