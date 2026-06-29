# Task 08 — "What works / where it breaks" report (fill in)

**Depends on:** Task 07. **Owner unit:** the deliverable the user asked for.

Fill this in as the staged run progresses. This is the answer to the user's question: *"determine what works, where it breaks."*

## Environment outcome

- Interactive session: partition `____`, GPU `____`, CUDA `____`, `nvidia-smi` OK? `____`
- Miniforge installed at: `____`
- `filterpipeline` env built? `____`  Python: `____`  openmm version actually installed: `____`
- Binaries on PATH: foldseek `__` mmseqs `__` fpocket `__` obabel `__` plip `__` vina `__`
- torch CUDA available? `____`  torch version: `____`
- enzymetk Chai/Boltz/ActiveSitePred run: in-process / named-env (`____`)

## Per-model status table

| Model | Stage | Status (works / partial / breaks / not-run) | Evidence / error | Notes |
|---|---|---|---|---|
| Squidly (ActiveSitePred) | Docking | | | ESM2 download + LSTM weights |
| Chai-1 | Docking | | | VRAM, weight download |
| Boltz-2 | Docking | | | optional in docko |
| AutoDock Vina (docko) | Docking | | | needs vina exe; P1-3 empty-residue bug |
| AF2 fetch (docko) | Docking | | | network egress needed |
| DockingMetrics | Docking | | | hard-coded Chai .npz / Boltz JSON keys |
| biotite superimposition | Superimposition | | | |
| protein/ligand RMSD | Superimposition | | | watch P0-3 add_metrics NameError |
| fpocket | GeometricFilters | | | CPU only |
| PLIP | GeometricFilters | | | ligand-chain heuristic |
| FreeSASA | GeometricFilters | | | CPU only |
| PLACER | (dead) | not integrated | import breaks by design | 3 colliding classes |

## Stage outcome

- Stage A Docking: `____`  (last pickle produced: `____`)
- Stage B Superimposition: `____`
- Stage C GeometricFilters: `____`  (`structural_features_final.pkl` produced? `____`)

## Bugs actually triggered

(List each known/new bug that fired, with file:line and the traceback head. Cross-ref `../05_bugs_by_severity.md`.)

## Recommended next steps

(e.g. apply roadmap Phase 0 hygiene fixes from `../10_roadmap.md` before a real run; whether RTX6000 suffices or h100 needed; which binary/env gaps to close.)
