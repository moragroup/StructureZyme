StructureZyme — Project Overview & Remaining Work
Where things stand
- Canonical checkout: /mnt/storage01/home/lherrmann/structurezyme (lowercase), branch lab-sanity-run_LCH, in sync with origin (github.com/moragroup/StructureZyme). The capital-S StructureZyme dir is an older checkout — ignore it.
- Health: Clean working tree. Test suite: 197 passed, 3 skipped (the 3 skips are GPU-gated PLACER smoke tests). No broken code.
- Last sanity run (fmo18, 18 enzymes): ran all 15 steps end-to-end through placer on Jul 22 (job 98795). The earlier ligand_rmsd KeyError('tool1') crash was fixed and the resumed run finished.
- Architecture: The big "unification" refactor (filterzyme → structurezyme, YAML-config + CLI + resumable modular runner, 15-step graph) is done and merged (PR #2). CLI input-seeding is done. This delivered pre-refactor wishlist items F-3, F-4, F-6, F-7, F-8.
The step graph (all working)
squidly → chai → boltz → vina → docking_metrics → prepare_files → fastrelax → superimpose → protein_rmsd → ligand_rmsd → geometric_filter → fpocket → ligand_sasa → plip → placer
(vina, fastrelax, placer ship disabled-by-default.)
What was postponed — the actual to-do list
1. Best-pose selector ignores fastrelax_score (documented design gap) — highest impact scientifically
docs/pipeline_overview.md:122-147. The ligand_rmsd selector picks the "best" pose using only inter-tool geometric consensus, ignoring Rosetta fastrelax_score and engine confidence. On fmo-fad-01 this deterministically picked tool="chai" on all 18 entries even where the single boltz pose had a much better energy. Needs a redesign to fuse geometry + energy, and must handle the single-pose-per-tool case. This is the one real "doesn't work properly" item.

The Jul 23 planning session (for the halogenase pyridine+tryptophan run) hit this wall: the schema allows only one substrate_smiles per Entry. Two substrates per enzyme requires either a schema change or a per-substrate row convention. This is why that run never got submitted.
3. Bring-your-own-PDB single-module runs — documented as "future / not yet implemented"
docs/configuration.md:245. You can't run a single downstream module (e.g. fastrelax, placer, geometric_filter) directly on an existing PDB, skipping Chai/Boltz. Needs a per-module "PDB → expected checkpoint columns" adapter.
4. Squidly upstream workaround — works, but is a maintenance debt
docs/known-issues.md + structurezyme/steps/squidly_step.py:237 (TODO(squidly-upstream)). We recompute catalytic residues locally because upstream drops --mean-prob/--mean-var. Removable once upstream is patched.

Unaddressed wishlist items (optional, F-list): F-1 multi-GPU, F-2 unified list-valued cofactor handling, F-5 per-entry errors.csv report, F-9 HTML report. None block finishing; they're enhancements.
My read on "to finish the project"
The pipeline runs end-to-end. The blockers to finishing are #2 (multi-substrate) — needed to actually submit your halogenase run — and #1 (pose selector) — needed for the results to be scientifically trustworthy. #3–#5 are nice-to-haves / cleanup.
Which of these do you want to tackle, and in what order? A couple of things I'd want to confirm before we start any implementation (via brainstorming):
- Is the halogenase two-substrate run the concrete goal that defines "finished"? If so, #2 is the critical path.
- For the pose selector (#1), do you have a scientific preference — e.g. "energy should always win ties" vs. a weighted rank-fusion?