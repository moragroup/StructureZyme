# StructureZyme — Lab-Handoff Readiness (Design)

**Date:** 2026-08-11
**Owner:** lherrmann (LCH)
**Canonical checkout:** `/mnt/storage01/home/lherrmann/structurezyme` (lowercase),
branch `lab-sanity-run_LCH`, in sync with `github.com/moragroup/StructureZyme`.

## Goal

Get StructureZyme ready to move into a shared lab folder where other lab users
can create the environment and run the pipeline. This is a **handoff-hygiene and
reproducibility** effort, not a redesign — the architecture (per-user output
dirs, host profiles, file locking) is already multi-user-ready. The remaining
work is completing the `filterzyme -> structurezyme` rename, making the test
suite green on a fresh checkout, unifying install docs, removing the
personal-account dependency for `squidly`, and general repo hygiene.

**Out of scope:** the two documented *scientific* gaps (best-pose selector
energy-awareness — partially addressed by the merged `fused_rank` work — and the
single-substrate schema limitation). These are results-quality concerns, not
"other users can't run it" concerns, and do not block the folder move.

## Context / Findings

- **Canonical repo is settled.** `~/structurezyme` (lowercase, 238 commits on
  `lab-sanity-run_LCH`) already contains the CamelCase `~/StructureZyme` HEAD
  (147 commits) as an ancestor, and is **0 ahead / 0 behind** its GitHub remote.
  No merge is needed. The `85` / `238` GitHub numbers are simply the divergence
  of `lab-sanity-run_LCH` from `main` (238 ahead, 85 behind). The CamelCase
  checkout is a stale subset.
- **Code rename is essentially done.** `filterzyme/__init__.py` is now an
  intentional deprecation shim (lazy `__getattr__` + `MetaPathFinder`)
  forwarding to `structurezyme`; kept until 0.2.0. No source in `structurezyme/`
  imports `filterzyme`.
- **Real conda env is still `~/envs/filterzyme`.** `environment.yml` already
  declares `name: structurezyme`, but the working env on disk is `filterzyme`,
  and `tools/*.sbatch` hardcode `/mnt/storage01/home/lherrmann/envs/filterzyme/bin`.
- **Test suite is NOT green on a fresh checkout:** 6 failures
  (`tools/test_pipeline.py` stale x4; `tests/test_placer_step.py` brittle x2),
  contradicting the "all green" note in `what_needs_to_be_done.md`.
- **`squidly` pinned to a personal fork** (`github.com/HerrLuca99/Squidly@022fa40`)
  for a single one-commit CLI-forwarding patch — a single point of failure for a
  shared install.

## Decisions (confirmed with user)

1. Canonical = lowercase `~/structurezyme` on `lab-sanity-run_LCH`. No GitHub
   merge. All work on a new branch `chore/lab-handoff`; **no push/merge/GitHub
   action without explicit approval.**
2. Conda env: **recreate `structurezyme` fresh from `environment.yml`** (proves
   the shipped spec works for a new user; catches drift). Verify before deleting
   the old `filterzyme` env.
3. Fix scope: **all handoff blockers** (Critical + Important) plus cleanup.
4. `squidly`: **move fork to the `moragroup` org** and repoint `environment.yml`
   to `github.com/moragroup/Squidly@022fa40` (lightest fix; user does the GitHub
   fork, plan leaves a clear placeholder). No vendoring.
5. sbatch env activation: **overridable** `STRUCTUREZYME_ENV` with a shared lab
   default, not a hardcoded personal path.
6. `benchmarking/yang_filterzyme/`: **leave as-is** (cosmetic historical name);
   add a historical note to `benchmarking/`.
7. `1_analysis-planning/`: **keep in repo** as project history.
8. Stale `~/StructureZyme` and old `~/envs/filterzyme`: **delete only after** the
   new setup verifies green.
9. Lab folder move: **prepare only, do not move.** Recommend a destination under
   `/mnt/labs/data/mora/...` (matches host-profile defaults); user does the move.
 10. Delivery: everything committed on `chore/lab-handoff` with a summary; user
     reviews the diff and decides when to merge/push.

## Workstream 1 — Canonical repo & consolidation

- Commit the currently-dirty tree first so work starts from a defined revision:
  - `structurezyme/steps/PLACER_step.py` (docstring-only change; verified no
    logic change).
  - Untracked `what_needs_to_be_done.md` and
    `docs/superpowers/plans/2026-07-17-structurezyme-unification.md` — commit
    both (user approved keeping project history in-repo).
- Create branch `chore/lab-handoff` off `lab-sanity-run_LCH`. All subsequent
  work lands there.
- **No push, merge, rebase, or GitHub action without explicit user approval.**
- Delete stale `~/StructureZyme` and old `~/envs/filterzyme` **only after**
  final verification passes (Workstream 5).

## Workstream 2 — Complete the filterzyme -> structurezyme rename

### Conda env (recreate from environment.yml)
- Create a fresh `structurezyme` env from `environment.yml` **after** the
  Workstream 4 file edits (install docs, enzymetk/cuequivariance pins) are done.
- squidly source: if the `moragroup` org fork exists at env-build time, use the
  repointed URL; otherwise build against the still-working personal-fork pin and
  repoint later (the repoint is a one-line change that does not require an env
  rebuild). The env-recreate is therefore **not blocked** on the user's org fork.
- Verify: imports, `structurezyme --help`, full test suite green — **before**
  deleting the old `filterzyme` env.

### SLURM scripts (`tools/*.sbatch`) — critical
Replace hardcoded `/mnt/storage01/home/lherrmann/envs/filterzyme/bin` with an
overridable, non-personal activation in every sbatch:
```bash
STRUCTUREZYME_ENV="${STRUCTUREZYME_ENV:-/mnt/labs/data/mora/envs/structurezyme}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$STRUCTUREZYME_ENV"
```
Affected (from review): `tools/gpu_run/run_allmodules.sbatch`,
`run_multirow.sbatch`, `resume_multirow.sbatch`, `tools/fmo18/run_fmo18.sbatch`,
`tools/smoke/*`. Enumerate all `tools/**/*.sbatch` during implementation.

### Docs, .gitignore, benchmarks
- `.gitignore`: add `*structurezyme_output/` patterns alongside existing
  `*filterzyme_output/` (keep old for back-compat).
- Reconcile remaining `filterzyme` env references in docs.
- Benchmark scripts' `base_output_dir="filterzyme_output"` -> update string
  (cosmetic; part of the "everything incl. cleanup" scope).
- `benchmarking/yang_filterzyme/`: **leave folder name**; add historical note.

### Deprecation shim
`filterzyme/__init__.py` unchanged (intentional back-compat, removed in 0.2.0).

## Workstream 3 — Fix the test suite (green on fresh checkout)

Target: 0 failures in the fresh `structurezyme` env; the 3 GPU-gated PLACER
smoke tests remain skipped on a non-GPU node (expected).

### (a) `tools/test_pipeline.py` — 4 stale failures
- Asserts `pipeline.squidly_model_size == "3B"`, an attribute the current
  `Pipeline` no longer exposes; uses old `filterzyme_test_output` naming.
- Get collected by `pytest` from repo root.
- Fix: add `pyproject.toml`/`pytest.ini` with `testpaths = ["tests"]` so only
  the real suite runs; **and** read `tools/test_pipeline.py` to decide
  update-vs-delete (if fully obsolete, delete; if it has salvageable cases,
  fix them).

### (b) `tests/test_placer_step.py` — 2 brittle failures
- Expect `FileNotFoundError` from `Path("/nonexistent/...").exists()`, but on
  this shared filesystem stat-ing a top-level nonexistent path raises
  `PermissionError`, so it fails for real lab users.
- Fix, two parts:
  1. Make the PLACER existence check (`PLACER_step.py:183`) treat
     `OSError`/`PermissionError` as "not found" (more robust for real users).
  2. Point the tests at a path under pytest `tmp_path` instead of `/nonexistent`.

### Verification
Run full suite in the fresh env; paste the actual pytest summary
(verification-before-completion). No "green" claim without evidence.

## Workstream 4 — Installability & reproducibility

### (a) squidly — remove personal-account dependency
Root cause (confirmed against `docs/known-issues.md`): upstream
`squidly==0.1.0` `squidly/__main__.py::run` builds the inner
`python squidly.py ...` command without appending `--mean_prob`/`--mean_var`, so
the ensemble worker always filters at defaults (0.6/0.225) regardless of the
outer CLI — yielding empty residue strings for proteins whose probabilities
never exceed 0.6 (e.g. flavin monooxygenases). Fork commit `022fa40` appends the
threshold args to every `cmd` build.
- Action: repoint `environment.yml` from `github.com/HerrLuca99/Squidly@022fa40`
  to `github.com/moragroup/Squidly@022fa40`.
- **User does the org fork.** Plan leaves a clear placeholder + a step; update
  the `known-issues.md` fork URL to match.

### (b) Unify install instructions
- Standardize on `pip install -e .` everywhere; remove the outdated
  `setup.py sdist bdist_wheel ... --use-deprecated=legacy-resolver` and the
  hardcoded `structurezyme-0.1.0.tar.gz` from `README.md`.
- Reconcile `enzymetk` pin to the actually-working version (inspect
  `~/envs/filterzyme` via `pip freeze`/`conda list`; review indicates `0.1.0`)
  and pin consistently in `README.md` + `environment.yml`.
- Remove `pip install squidly` (plain) from `docs/getting_started.md` — it pulls
  unpatched upstream and reintroduces the bug. env.yml is the only squidly
  source.

### (c) cuequivariance_torch
In `setup.py` install_requires but absent from `environment.yml`; CUDA-specific,
can fail on non-GPU login nodes. Add it to `environment.yml` consistently with
the other GPU deps (chai_lab, torch), or move to an extras group + document the
GPU requirement. Decide based on how the existing GPU deps are declared.

## Workstream 5 — Repo hygiene & finalize (no move)

### Cleanup
- `test_placer/` (218 KB, near-duplicate of `benchmarking/metallohydrolases/`,
  tracked, not gitignored): confirm it's scratch, then remove.
- `examples/DEHP-MEHP.pkl` (116 KB, force-added despite `*.pkl` ignore): keep;
  add a one-line note in `examples/` describing what it is.
- Cosmetic `filterzyme`/personal-path leftovers in comments (e.g.
  `fpocket_step.py:20` `/home/helen/...`): tidy.
- `1_analysis-planning/`: keep as-is (project history).
- `benchmarking/`: add a short historical note that scripts contain dead
  personal paths (`/nvme2/helen/...`, `/mnt/labs/data/mora/code/Filterzyme/...`,
  `conda activate filterzyme`) and need editing before use.

### Finalize (in place, no move)
- Full test suite green in the fresh env (paste summary).
- Sanity: `structurezyme --help`; `structurezyme submit --dry-run` renders.
- Everything committed on `chore/lab-handoff`; produce a change summary.
- **No push/merge/GitHub without approval.**
- After all green: delete stale `~/StructureZyme` and old `~/envs/filterzyme`.
- Recommend lab destination under `/mnt/labs/data/mora/...` (matches host-profile
  defaults); **user performs the actual move.**

## Risks & open items

- **squidly org fork is a user action** (GitHub perms). Until done,
  `environment.yml` still references the personal fork; env recreation should
  use the working spec meanwhile. Track as a required follow-up.
- **GPU required** for a real end-to-end verify (squidly/ESM2, chai, boltz).
  On a non-GPU node, verification is limited to install + unit tests + dry-runs;
  the GPU smoke tests stay skipped. Full pipeline re-run is a separate step.
- **enzymetk / cuequivariance_torch pins** must reflect what actually works in
  the rebuilt env; confirm empirically, don't guess.

## Success criteria

1. Fresh `structurezyme` env builds from `environment.yml`.
2. `pip install -e .` succeeds; `structurezyme --help` works.
3. Test suite: 0 failures (GPU smoke tests skipped) — evidence pasted.
4. No hardcoded personal paths / `filterzyme` env references in
   `tools/*.sbatch`, `environment.yml`, or install docs.
5. Install docs internally consistent (one method, matching pins).
6. Clean committed tree on `chore/lab-handoff`; stale checkout + old env removed;
   repo ready for the user to copy into the lab folder.
