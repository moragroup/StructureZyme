# 8. Top 5 Recommended Model Additions

These additions broaden Filterzyme's coverage along three axes: better docking scoring (GNINA, Smina), faster / more accurate pose sampling (DiffDock), modern structure prediction (AlphaFold3, ESMFold/OmegaFold). For each: rationale, integration sketch, dependencies, and rough cost.

## 1. GNINA â CNN-rescored Vina

**Why.** GNINA is a fork of Smina (which is itself a fork of Vina) that re-scores poses with a 3D CNN trained on PDBbind. CNNscore and CNNaffinity correlate substantially better with experimental affinities than the Vina scoring function does, especially for non-drug-like substrates. For an enzyme-substrate filter the extra discriminative power matters.

**Integration.** Drop-in replacement at the Vina layer. Add `filterzyme/steps/dock_gnina_step.py` modelled on `dock_vina_step.py` (interface: same `(label, residues, smiles, structure_path) â pose_dir + scores_dict`). Extend `extract_docking_metrics_step.py` to parse GNINA's `gnina.out` (line-prefixed `mode`, `affinity`, `intramol`, `CNNscore`, `CNNaffinity`). Add `docking_engine: Literal["vina","gnina","smina"]` to `VinaConfig` (see `07_top10_improvements.md` item 4).

**Dependencies.** GNINA binary (single statically-linked executable, CUDA-enabled). No new Python deps. Test fixtures: same as Vina.

**Cost.** ~2 days to wire and unit-test.

## 2. DiffDock â diffusion-based pose sampling

**Why.** DiffDock is a generative diffusion model over ligand poses. Unlike Vina/GNINA it does not require a known pocket; given a holo or apo structure plus a ligand SMILES, it samples plausible poses in seconds on a GPU. Useful for entries where Squidly's active-site prediction is uncertain and for substrates whose binding mode is novel relative to the training distribution.

**Integration.** Add `filterzyme/steps/dock_diffdock_step.py`. The model takes `(receptor_pdb, ligand_smiles)` and produces `N` ranked SDF poses plus a confidence score. Wire into the Docking stage as an alternative engine: `pipeline_v2.py` `Docking._run_dock(...)` dispatches on `config.docking_engine`. Outputs flow through the same `PrepareVina`-equivalent step.

**Dependencies.** PyTorch, ESM, the DiffDock weights (~1 GB), `torch_geometric`. Significant install footprint; ship as an optional extra: `pip install filterzyme[diffdock]`.

**Cost.** ~4 days (the install + IO formatting is most of the work; the math is upstream).

## 3. Smina â Vina with custom scoring functions and atomtypes

**Why.** Smina is a fork of Vina with extended scoring (custom Vinardo, Ad4 forcefield) and arbitrary atom-type definitions. Drop-in for cases where Vina's default scoring under-resolves enzyme-specific catalytic geometry. Lowest-effort addition of the five.

**Integration.** Same as GNINA, simpler. The Vina binary and Smina binary share command-line conventions. Add `dock_smina_step.py` (likely ~30 LOC over the Vina version) and a scoring-function selector.

**Dependencies.** Single binary, no Python deps.

**Cost.** ~1 day.

## 4. AlphaFold3 (or OpenFold-AF3) â all-atom prediction with ligands

**Why.** AF3 (and the public OpenFold/Boltz-2 reimplementations) handle protein + ligand + cofactor + ion + metal in one forward pass. Filterzyme already uses Chai-1 and Boltz-2 for exactly this, but AF3-class models are state-of-the-art and ought to be option in the consensus. Adding AF3 specifically gives a third opinion that can break ties between Chai and Boltz.

**Integration.** Add `enzymetk.AF3` (assuming `enzymetk` is the right home for upstream-tool wrappers; see open question in `12_assumptions_open_questions.md`). Wire into `Docking._run_af3` mirroring `_run_chai` and `_run_boltz`. Extend `DockingMetrics.execute` to parse the AF3 confidence JSON. Best-structure selection in `Superimposition` becomes a 3- or 4-way pairwise rather than 2-way.

**Dependencies.** Weights license (AF3 weights are gated for non-commercial use; OpenFold-AF3 is open). DeepMind's `alphafold3` package or OpenFold equivalent. GPU memory >24 GB recommended.

**Cost.** ~5 days. Weights/licensing dominates the timeline.

## 5. ESMFold (or OmegaFold) â fast apo prediction for AF2-misses

**Why.** When `docko.get_alphafold_structure` misses (no UniProt entry, novel metagenomic sequence), the current code falls back to Chai or Boltz as the Vina receptor (gated by `alternative_structure_for_vina`). Both are expensive. ESMFold runs single-sequence prediction in seconds without an MSA, at the cost of some accuracy. For Vina-receptor-only purposes (where the ligand is going to be re-docked anyway) it is a strong cost/quality tradeoff.

**Integration.** Add `filterzyme/steps/esmfold_step.py` invoked from `Vina.__execute`'s AF2-fallback branch (currently around `dock_vina_step.py:44`). Tertiary fallback order: AF2 cache â AF2 fetch â ESMFold â Chai/Boltz.

**Dependencies.** `fair-esm` and the ESMFold weights (~3.5 GB). GPU helpful but not strictly required.

**Cost.** ~2 days.

## Honorable mentions

| Model | Why considered | Why not in top 5 |
|---|---|---|
| **RoseTTAFold-AA** | All-atom predictor competitive with AF3 | Overlaps with AF3; choose one |
| **Equibind** | Equivariant pose prediction, very fast | Largely subsumed by DiffDock and less accurate on average |
| **RFAA (RoseTTAFold All-Atom)** | Same family as RFAA, broader support | Same as RFAA; same overlap with AF3 |
| **Vina-GPU 2.0** | Vina with CUDA backend, ~10x speedup | Drop-in for Vina; not a new model, more a perf upgrade. Worth doing as a config flag later. |

## Selection logic

If the team can do only one: **GNINA** (1â2 days, immediate scoring upgrade, no new install class).

If two: **GNINA + ESMFold** (covers scoring upgrade and AF2-miss fallback, both <3 days).

If a sprint: all five, in roadmap Phase 3.
