# Filterzyme sanity-run report (LCH, 2026-06-26)

Run on cluster node gpu03 (NVIDIA RTX PRO 6000 Blackwell, sm_120, driver 610.43.02).
Branch: lab-sanity-run_LCH off origin/pbp. Lab repo sha d3bd867 ("Updated readme", 2026-06-25).

## Environment outcome

- Working dir: /mnt/labs/data/mora/code/Filterzyme (in place, read-only from our side)
- Conda env: $HOME/envs/filterzyme/ (self-built; Ariane's env is private to her account)
- Python: 3.11.15
- torch: 2.11.0+cu128, cuda.is_available()=True, kernel probe OK on sm_120
- GPU: NVIDIA RTX PRO 6000 Blackwell, driver 610.43.02
- Boltz cache: /mnt/labs/data/mora/code/Filterzyme/boltz_cache (9.1 GB, healthy)
- Output dir (ours): $HOME/filterzyme-sanity/PDE_2H_LCH/filterzyme_output_LCH/

## Env divergences from Ariane's README (8 patches, all reversible)

| # | What | Why |
|---|------|-----|
| 1 | torch 2.11.0+cu128 | Blackwell sm_120 needs cu128; chai_lab-pinned torch 2.6 only ships sm_50..sm_90 |
| 2 | enzymetk 0.1.0 (kept) | README says 0.0.8; filterzyme pulls 0.1.0 and downgrade would break it |
| 3 | conda install pdbfixer | OpenMM dep, not on PyPI, missing from README |
| 4 | conda install openbabel | C++ lib with bindings, conda-only, missing from README |
| 5 | conda install plip | Protein-Ligand Interaction Profiler, conda-only, missing from README |
| 6 | numpy <2.2 force-pin | numba refuses 2.2+; conda installs had dropped 2.4 alongside pip 1.26 |
| 7 | regex 2026.5.9 | enzymetk transitive needs >=2025.10.22, overrides docko's stale pin |
| 8 | docko 0.1.5 from ~/docko_lab_LCH | PyPI 0.1.3 has 5-arg run_boltz_affinity; lab has 6-arg, never released |

Full reversion paths in `../4_lab-sanity-run/03_env_state.md`.

## Per-stage status

| Stage | Step | Status | Output produced | Evidence |
|-------|------|--------|-----------------|----------|
| Docking | Chai | ✅ OK | chai.pkl (2 rows × 14 cols) | 83 s, ~9.7 GB GPU mem |
| Docking | Boltz | ✅ OK | boltz.pkl (2 × 15) | 43 s, with cuequivariance_ops_torch fallback (slower but functional) |
| Docking | Vina (via Chai) | ✅ OK | (folded into dockingmetrics) | configured alternative_structure_for_vina='Chai' |
| Docking | DockingMetrics | ✅ OK | dockingmetrics.pkl (2 × 41) | |
| Superimposition | Superimpose | ✅ OK | superimposedstructures.pkl (2 × 44), 13 superimposed PDBs | |
| Superimposition | **proteinRMSD** | ✅ OK | proteinRMSD.pkl (10 × 60), proteinRMSD_pairwise.pkl (14 × 61), 2 heatmaps | **past Ariane's predicted break point** |
| Superimposition | **ligandRMSD** | ❌ BREAK | ligandRMSD_prior.pkl (0×0, empty), 2 heatmaps | KeyError 'docked_structure' (see below) |
| GeometricFilters | fpocket / PLIP / FreeSASA | — | not reached | blocked by ligandRMSD |

## Bugs triggered

### NEW — ligandRMSD silent-empty + downstream KeyError (Stage B.3)

- **Visible exception:** `KeyError: "Expected column 'docked_structure' not found."` at `filterzyme/utils/helpers.py:319` in `extract_docking_metrics`.
- **Call chain:** `pipeline_v2.py:447 (Superimposition.run) → :261 → helpers.py:87 (log wrapper) → pipeline_v2.py:318 (_ligandRMSD calls extract_docking_metrics)`.
- **Real fault (upstream):** `LigandRMSD` step (in enzymetk) logs `Error selecting best docked structures: 'Entry'` and silently returns an empty DataFrame. Proof: `ligandRMSD_prior.pkl` exists on disk as 0 rows × 0 cols.
- **Contract violation:** `_ligandRMSD` does not validate LigandRMSD's output before passing it to `extract_docking_metrics`.
- **Not in** `../1_repo_analysis/05_bugs_by_severity.md` — add as new finding (severity ~P1, similar pattern to CQ-15 case-sensitivity).

### Side findings (not blocking)

- 4 per-entry Chai pickles (`docking/chai/PDE*/chai/PDE*.pkl`) are corrupted: `TypeError: object of type 'zip' has no len()` — a `zip` iterator was pickled instead of being materialised. Top-level chai.pkl is fine; per-entry artefacts are unused by downstream stages.
- `cuequivariance_ops_torch` import errors inside Boltz (no cu128 wheel exists). Boltz fell back to the slower standard triangle multiplicative update. Non-fatal.

## Recommended next steps

1. **Open a follow-up plan** to fix the ligandRMSD break. Two layers: (a) fix the `Entry` column lookup inside `LigandRMSD` (enzymetk), (b) make `_ligandRMSD` raise instead of silently passing an empty df downstream. Resume the pipeline from `superimposedstructures.pkl` to avoid re-running ~2 min of Chai+Boltz.
2. **Talk to Ariane** with `../4_lab-sanity-run/02_PI_status.md`. Confirm: docko editable from lab tree, conda extras (pdbfixer/openbabel/plip), torch cu128 compatibility on her end, and the `'Entry'` lookup error (does it ring a bell?).
3. **Update the README** in `/mnt/labs/data/mora/code/Filterzyme/` with the 8 patches once Ariane signs off — she alone has write access.
4. **Investigate cuequivariance_ops_torch cu128 wheel** for a faster Boltz path. Low priority; current fallback works.
5. **Hardware:** RTX PRO 6000 Blackwell suffices for the smoke test (~10 GB peak for Chai). No need to use the h100 partition for this workload.

## Cleanup recommendations (do NOT delete yet)

- `$HOME/envs/filterpipeline` — built before we knew Ariane's env exists; superseded by `$HOME/envs/filterzyme`. Safe to delete once Ariane confirms our setup. ~Several GB.
- `$HOME/enzyme-tk-agentic` — Plan-B enzymetk clone, never used. Safe to delete.
- `$HOME/.conda/pkgs` — keep (shared conda cache, harmless).
- `$HOME/StructureZyme/` (this repo) and `1_analysis-planning/` — keep.
- `$HOME/docko_lab_LCH/` — keep (editable install target for docko 0.1.5).
- `$HOME/filterzyme-sanity/` — keep (run snapshots, baseline, outputs, revert files).

## Summary in one sentence

After 8 documented env patches, the lab Filterzyme pipeline runs end-to-end through Docking and Superimposition (including protein RMSDs — past Ariane's predicted break point) on the Blackwell GPU node, then breaks at ligand RMSD due to a column-lookup failure in `LigandRMSD` that silently produces an empty DataFrame.
