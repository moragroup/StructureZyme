# PI status note — Filterzyme sanity run (UPDATED 2026-06-26 afternoon)

**From:** LCH  **Date:** 2026-06-26  **Cluster node:** gpu03 (Blackwell sm_120)

## TL;DR

End-to-end sanity run of the lab Filterzyme pipeline on the new Blackwell GPU node. After rebuilding your env locally (yours is private to your account) and patching six environment issues, the pipeline now runs through **all of Docking and most of Superimposition** — past the protein-RMSD point you predicted would break. It then breaks at **ligand RMSD** with a `KeyError: 'docked_structure'`. This is a new finding.

## What I need from you

1. **Confirm:** do you run docko as an editable install from `/mnt/labs/data/mora/code/docko/` (lab version 0.1.5)? PyPI only has 0.1.3 which has a 5-arg `run_boltz_affinity` and is incompatible with current enzymetk. README implies pip-only; this should be documented.
2. **Conda-only packages** missing from the README — please confirm you also conda-installed: `pdbfixer`, `openbabel`, `plip`. (Not on PyPI as importable modules.)
3. **PyTorch on Blackwell:** I had to upgrade to torch 2.11.0+cu128 because the GPU is sm_120 and the chai_lab-pinned torch 2.6 only supports up to sm_90. Does your env still run on the new nodes?
4. **`Error selecting best docked structures: 'Entry'`** — does this ring a bell? It precedes the `docked_structure` KeyError and may indicate a column-rename further upstream.

## What works (Stages cleared end-to-end)

- Env builds and imports clean.
- CUDA kernels run on Blackwell.
- **Docking:** Chai ✓, Boltz ✓, Vina (via Chai structures) ✓, DockingMetrics ✓.
- **Superimposition:** Superimpose structures ✓, **protein RMSDs ✓** (past your predicted break point).

## Where it breaks (new finding)

- **Stage B.3 ligand RMSDs:** `KeyError: "Expected column 'docked_structure' not found."`
- Location: `filterzyme/utils/helpers.py:319` in `extract_docking_metrics`, called from `pipeline_v2.py:318` (`_ligandRMSD`).
- Preceding log line: `Error selecting best docked structures: 'Entry'`.
- Likely cause: column-rename / schema-drift between `select_best_docked_structures` (which silently fails on missing `Entry`) and `extract_docking_metrics` (which requires `docked_structure`).

## Environment we ended up with (divergences from your README)

| Item | Your README | What I had to do |
|------|-------------|------------------|
| enzymetk | pin to 0.0.8 | kept 0.1.0 (filterzyme pulls it; downgrade breaks) |
| torch | (implicit, chai_lab pin → 2.6) | upgraded to 2.11.0+cu128 for Blackwell |
| pdbfixer | not mentioned | `conda install -c conda-forge pdbfixer` |
| openbabel | not mentioned | `conda install -c conda-forge openbabel` |
| plip | not mentioned | `conda install -c conda-forge plip` |
| numpy | (implicit) | force-pinned `<2.2` (numba refuses 2.2+) |
| regex | docko pin 2024.9.11 | bumped to ≥2025.10.22 (enzymetk needs it) |
| docko | (assumed pip) | `pip install -e ~/docko_lab_LCH` (copy of your /mnt/labs/data/mora/code/docko) — version 0.1.5 |

## What is unchanged in your tree

- `/mnt/labs/data/mora/code/` is read-only from my side. Only my env (`~/envs/filterzyme/`), my docko copy (`~/docko_lab_LCH/`), and my workspace (`~/filterzyme-sanity/`) were created.
- Lab repo is at sha `d3bd867` ("Updated readme", 2026-06-25 15:20) with your existing uncommitted edits captured in `baseline_LCH.diff`.

## Soft warnings observed (not blocking, mention for completeness)

- `cuequivariance_ops_torch` failed to import inside Boltz (no cu128 wheel available). Boltz fell back to standard triangle multiplicative update — slower but functional.
