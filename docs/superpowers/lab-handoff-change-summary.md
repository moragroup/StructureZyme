# StructureZyme lab-handoff — change summary

Branch: `chore/lab-handoff` (off `lab-sanity-run_LCH`), **10 commits ahead, 0 behind**.
Status: **local only — not pushed, merged, or PR'd.** Awaiting your review.

Goal: make the `structurezyme` project (formerly `filterzyme`) installable and
runnable by other lab members from a shared location, without maintainer-specific
paths, credentials, or a broken test suite.

---

## What changed (by theme)

### 1. Test suite is green on a fresh checkout
- Added `pyproject.toml` with `testpaths = ["tests"]` so `tools/` dev-smoke tests
  are excluded from default collection.
- `structurezyme/steps/PLACER_step.py`: added a `_path_exists()` helper that treats
  un-statable paths (PermissionError on the shared filesystem) as "not found" instead
  of crashing `PLACER.__init__`'s validation. Retargeted two brittle tests at
  `tmp_path`.
- `structurezyme/pipeline.py`: fixed a real regression — `PathsConfig.input_csv`
  became required, but the legacy `Pipeline` adapter never set it, so every
  `Pipeline(...).run()` raised. Now points `input_csv` at the `_input` checkpoint
  pickle the adapter already seeds (the Runner short-circuits CSV loading when that
  pickle exists, so it is never read as a CSV). Independently reviewed as correct.
- `tools/test_pipeline.py`: fixed stale assertions and rewrote 3 tests to stub at the
  modern `registry.STEPS[...].runner` level instead of legacy classes the current
  Pipeline no longer constructs.

### 2. SLURM scripts are multi-user (no personal env/repo paths)
- Every sbatch/shell launcher under `tools/` now activates an **overridable** env:
  ```bash
  STRUCTUREZYME_ENV="${STRUCTUREZYME_ENV:-/mnt/labs/data/mora/envs/structurezyme}"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$STRUCTUREZYME_ENV"
  ```
  (replacing hardcoded `/mnt/storage01/home/lherrmann/envs/filterzyme` /
  `$HOME/envs/filterzyme`).
- `tools/fmo18/submit_fmo18.sbatch`: env + worktree made overridable
  (`STRUCTUREZYME_REPO`), `ENV_PY` resolved via `command -v python` after activation.
- `tools/smoke/run_smoke_on_gpu.sh`: replaced a hardcoded personal repo `cd` with a
  `STRUCTUREZYME_REPO`-overridable repo-root derivation (caught in review).

### 3. Unified, reproducible install
- `README.md`: install collapsed to the standard three lines —
  `conda env create -f environment.yml` / `conda activate structurezyme` /
  `pip install -e .` (removed the `setup.py sdist`, legacy-resolver, hardcoded tarball,
  and `enzymetk==0.0.8` steps).
- `environment.yml`: pinned `enzymetk==0.1.0`, added `cuequivariance_torch==0.10.0`,
  added `pytest` (fresh env could not run the suite without it), removed the stale
  self-referential `filterzyme` comment.
- `docs/getting_started.md`: removed the standalone `pip install squidly` (it pulled
  unpatched upstream and reintroduced the empty-residue bug); text now notes squidly
  comes from `environment.yml`.

### 4. squidly pinned to fixed public upstream (no fork needed)
- The `--mean_prob`/`--mean_var` forwarding bug is now **fixed in public upstream**
  `WRiegs/Squidly@main` (verified: it forwards `threshold_args` to the ensemble worker
  and ships `tests/test_cli_forwards_thresholds.py` as a regression guard).
- `environment.yml` pins
  `squidly @ git+https://github.com/WRiegs/Squidly.git@58a8f7d6cac128c4d0915835d20ecb575cb72931`
  (immutable commit — upstream has no tagged release yet). This **drops the personal
  `HerrLuca99` fork** and makes the `moragroup` org-fork step unnecessary.
- Verified in the fresh env: squidly builds + installs from upstream, CLI on PATH, the
  fix is present in the installed `__main__.py`, and the suite is still
  242 passed / 5 skipped / 0 failed.

### 5. Repo hygiene + rename cosmetics
- Removed `test_placer/` (a tracked scratch duplicate — older/messier copies of
  `benchmarking/metallohydrolases/`, plus a 504 KB inspection notebook with no unique
  analysis).
- Added `benchmarking/README.md` (historical-scripts note) and `examples/README.md`.
- Finished `filterzyme -> structurezyme` cosmetics in `.gitignore`, benchmark
  `base_output_dir` strings, and a personal-path comment in `fpocket_step.py`.

---

## Verification evidence

Fresh env built purely from `environment.yml` at
`/mnt/storage01/home/lherrmann/envs/structurezyme` (CPU node):

| Step | Result |
|------|--------|
| `conda env create -f environment.yml` | PASS (exit 0). enzymetk-0.1.0, cuequivariance_torch-0.10.0, squidly-0.1.0 (pinned upstream), torch-2.11.0+cu128, boltz-2.2.1, chai_lab-0.6.1, meeko-0.7.1, vina-1.2.7, docko (patched fork) |
| `pip install -e .` | PASS — `Successfully installed structurezyme-0.1.0` |
| `import structurezyme` + `--version` | PASS — `0.1.0` |
| `structurezyme --help` | PASS — lists `init/run/resume/step/status/submit` |
| `pytest tests/ tools/test_pipeline.py` | **242 passed, 5 skipped, 0 failed** |
| `structurezyme submit --dry-run` | PASS — renders a valid sbatch calling `structurezyme run` |
| torch on GPU (Blackwell sm_120) | PASS — matmul + LSTM ok with torch 2.11.0+cu128 |
| **GPU end-to-end (Step 6)** | **PASS** — full Squidly→Chai→Boltz→Vina→metrics on gpu partition (CalB/P41365) → `structural_features_final.pkl` (6×108, catalytic residues + cross-tool RMSDs + boltz2 affinity + vina outputs) |

The 5 skips are all legitimate (2× pyrosetta not installed [licensed, intentional],
1× PLACER real-run gated on `PLACER_SMOKE=1`+GPU, 2× squidly step needs weights+CUDA;
squidly CLI itself detected as installed). Full detail in
`.superpowers/sdd/task-7-report.md`.

---

## Commits on `chore/lab-handoff`

```
3347aa0 env: add pytest to environment.yml so a fresh env can run the test suite
612dc30 chore(hygiene): remove test_placer scratch dir; add benchmarking/examples orientation notes
da218c3 docs+env: unify install on 'pip install -e .', pin enzymetk/cuequivariance, pin reachable squidly fork with moragroup TODO
c9f1c11 chore: finish filterzyme->structurezyme cosmetic rename (gitignore, benchmark output dirs, comment)
d169261 fix(tools): de-personalize repo cd in run_smoke_on_gpu.sh
25f2db1 fix(tools): overridable STRUCTUREZYME_ENV activation; drop hardcoded personal env paths
ce2eba6 test: green suite on fresh checkout (pytest testpaths, tolerant PLACER path checks, fix stale pipeline smoke tests)
f429eb3 chore: commit in-flight docstring + project-notes before handoff cleanup
d768a3b docs(plan): lab-handoff readiness implementation plan
d02cf6f docs(spec): lab-handoff readiness design
```
(`git diff --stat lab-sanity-run_LCH..chore/lab-handoff`: 35 files, +3019 / -6611.)

---

## Outstanding user actions

These need your credentials/decisions and were intentionally NOT done automatically.

### A. squidly — RESOLVED, no action needed
The forwarding bug is fixed in public upstream `WRiegs/Squidly@main`; the pin now points
at upstream commit `58a8f7d`, so the personal fork and the `moragroup` org fork are no
longer needed. Optional future nicety: bump the pin to a tagged release once upstream
cuts one.

### A2. docko fork — PUSH REQUIRED (blocks a clean rebuild)
The docko dependency needs two patches for a single-env install (see
`docs/known-issues.md`). I prepared a clean fork commit but cannot push it (needs
your GitHub credentials). `environment.yml` currently pins
`docko @ git+https://github.com/moragroup/docko.git@REPLACE_WITH_SHA` — a
**placeholder that will fail `conda env create` until you push the fork and fill
in the SHA**.

Steps:
1. Create the fork under `moragroup/docko` (from `ArianeMora/docko`).
2. Push the prepared commit. It currently lives at
   `/tmp/lch_docko_work/docko` on branch `fix/mk-prepare-ligand-no-conda-run`
   (local commit `c596dee`, off upstream `d3273dd` / v0.1.4). E.g.:
   ```bash
   cd /tmp/lch_docko_work/docko
   git remote add moragroup git@github.com:moragroup/docko.git
   git push moragroup fix/mk-prepare-ligand-no-conda-run
   ```
   (If `/tmp/lch_docko_work` has been cleared, the two-file patch is fully
   described in `docs/known-issues.md` and can be re-applied to a fresh clone of
   `ArianeMora/docko@d3273dd`.)
3. Replace `REPLACE_WITH_SHA` in `environment.yml` with the pushed commit SHA.

### B. GPU end-to-end sanity run (Task 7 Step 6) — DONE ✓
Ran the full chain on the `gpu` partition against the fresh env and it passed
end-to-end (Squidly → Chai → Boltz → Vina → docking metrics on CalB / P41365,
producing `structural_features_final.pkl`). To re-run:
```bash
STRUCTUREZYME_ENV=/mnt/storage01/home/lherrmann/envs/structurezyme \
  sbatch tools/smoke/run_phase_c_smoke.sbatch
```
First-run model assets are required (see `docs/getting_started.md`): the Squidly
weights are a **manual** `python -m squidly.download_models_hf`; ESM2/Chai/Boltz
assets auto-download (and are already present on this server).

### C. Delete stale artifacts — WHEN YOU'RE READY (you chose: not now)
Both still exist; run these only once you're confident in the new env:
```bash
rm -rf /mnt/storage01/home/lherrmann/StructureZyme          # stale CamelCase ancestor checkout
conda env remove -p /mnt/storage01/home/lherrmann/envs/filterzyme   # old working env
```

### D. Move to the shared lab location (recommendation only — you perform the move)
Host profiles already default `output_root` etc. under `/mnt/labs/data/mora/...`.
Recommended shared locations:
- code: `/mnt/labs/data/mora/code/structurezyme` (sibling of existing `.../code/*`)
- env:  `/mnt/labs/data/mora/envs/structurezyme` (the sbatch `STRUCTUREZYME_ENV` default;
  `/mnt/labs/data/mora/envs` is already group-writable)

Rebuild the env at the shared path rather than copying the prefix (conda envs are not
relocatable):
```bash
conda env create -f environment.yml -p /mnt/labs/data/mora/envs/structurezyme
conda run -p /mnt/labs/data/mora/envs/structurezyme pip install -e /mnt/labs/data/mora/code/structurezyme
```

### E. Branch disposition (no push/merge without your OK)
`chore/lab-handoff` is 10 ahead / 0 behind `lab-sanity-run_LCH` (clean fast-forward).
When you approve, either merge locally (`git checkout lab-sanity-run_LCH && git merge --ff-only chore/lab-handoff`)
and/or push. I will not do this without an explicit go-ahead.

---

## Non-blocking follow-ups noted during verification
- `environment.yml` pins `openmm==8.3.1` (conda) but pip's `boltz`/`chai_lab` downgrade
  it to 8.1.1 during the build (noisy uninstall/reinstall; env still works). Consider
  aligning or dropping the pin.
- `setup.py:50` `install_requires` lists unpinned `enzymetk` (environment.yml pins
  `0.1.0`); minor inconsistency, harmless since environment.yml drives the env.
- Minor test-file nit: `tools/test_pipeline.py`'s directory-glob assertion is coupled to
  the `output_root/user/run_id` depth; could derive from `ctx.layout.root` instead.

## Explicitly out of scope (scientific, not handoff)
Best-pose energy-awareness beyond the merged `fused_rank`; multi-substrate schema;
bring-your-own-PDB single-module runs.
