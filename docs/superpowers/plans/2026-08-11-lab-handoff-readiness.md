# StructureZyme Lab-Handoff Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `structurezyme` repo installable, testable, and runnable by other lab users from a shared folder, completing the `filterzyme -> structurezyme` rename and removing personal-path / personal-account dependencies.

**Architecture:** Repo already restructured into the `structurezyme` package with a `filterzyme` deprecation shim, per-user output dirs, host profiles, and file locking. This plan does not change architecture; it fixes handoff-hygiene, reproducibility, and test gaps, then verifies on a freshly recreated conda env.

**Tech Stack:** Python 3.11, conda/mamba, pytest, SLURM (sbatch), setuptools.

## Global Constraints

- Canonical checkout: `/mnt/storage01/home/lherrmann/structurezyme` (lowercase), branch `chore/lab-handoff` (already created off `lab-sanity-run_LCH`).
- **No push, merge, rebase, pull, or any GitHub action without explicit user approval.** Commits to the local `chore/lab-handoff` branch only.
- Conda env is `structurezyme`; recreate fresh from `environment.yml`. Working versions to pin: `enzymetk==0.1.0`, `cuequivariance_torch==0.10.0`.
- squidly fork target: `git+https://github.com/moragroup/Squidly.git@022fa40` (user performs the org fork; until then the personal-fork pin `HerrLuca99/Squidly@022fa40` remains functional).
- sbatch env activation must be overridable, defaulting to a shared lab path, never a hardcoded personal path:
  ```bash
  STRUCTUREZYME_ENV="${STRUCTUREZYME_ENV:-/mnt/labs/data/mora/envs/structurezyme}"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$STRUCTUREZYME_ENV"
  ```
- Deprecation shim `filterzyme/__init__.py` is intentional; do NOT remove.
- `benchmarking/yang_filterzyme/` folder name stays; only add a historical note.
- `1_analysis-planning/` stays in the repo.
- GPU is required only for a real end-to-end pipeline verify (squidly/ESM2, chai, boltz); unit tests, install checks, and dry-runs do not need one. Ask the user for a GPU node when the end-to-end verify step is reached.
- Verification-before-completion: never claim "green"/"done" without pasting the actual command output.

## File Structure

- `pyproject.toml` — **create**; add `[tool.pytest.ini_options] testpaths = ["tests"]` so the real suite (not `tools/`) is collected by default.
- `structurezyme/steps/PLACER_step.py` — **modify**; make the two `.exists()` validation checks tolerate `OSError`/`PermissionError`.
- `tests/test_placer_step.py` — **modify**; point the two "missing path" tests at `tmp_path` instead of top-level `/nonexistent`.
- `tools/test_pipeline.py` — **modify**; fix the stale `squidly_model_size` attribute assertion + old output-dir naming (dev smoke test).
- `tools/**/*.sbatch`, `tools/smoke/run_smoke_on_gpu.sh` — **modify**; overridable env activation.
- `environment.yml` — **modify**; pin `enzymetk==0.1.0`, add `cuequivariance_torch==0.10.0`, squidly org URL, drop stale comment.
- `README.md`, `docs/getting_started.md` — **modify**; unify install on `pip install -e .`, fix pins, remove `pip install squidly`.
- `docs/known-issues.md` — **modify**; update squidly fork URL to the org.
- `.gitignore` — **modify**; add `*structurezyme_output/` patterns.
- `benchmarking/README.md` — **create**; historical-note.
- `examples/README.md` — **create**; one-line note on `DEHP-MEHP.pkl`.
- `structurezyme/steps/fpocket_step.py` — **modify**; tidy a personal-path comment.
- benchmark `run_*.py` — **modify**; `base_output_dir` string rename (cosmetic).
- `test_placer/` — **remove** (scratch duplicate) after confirming.

---

### Task 1: Commit the dirty working tree (clean starting revision)

**Files:**
- Modify (commit): `structurezyme/steps/PLACER_step.py` (already-modified, docstring-only)
- Add (commit): `what_needs_to_be_done.md`, `docs/superpowers/plans/2026-07-17-structurezyme-unification.md`

**Interfaces:**
- Consumes: nothing.
- Produces: a clean `git status` on branch `chore/lab-handoff`.

- [ ] **Step 1: Confirm you are on the handoff branch**

Run: `git rev-parse --abbrev-ref HEAD`
Expected: `chore/lab-handoff`

- [ ] **Step 2: Verify the PLACER_step.py change is docstring-only**

Run: `git diff structurezyme/steps/PLACER_step.py`
Expected: only a docstring wording change ("scaffold" -> "implemented"), no logic change. If logic changed, STOP and report.

- [ ] **Step 3: Stage and commit the pre-existing changes**

```bash
git add structurezyme/steps/PLACER_step.py what_needs_to_be_done.md docs/superpowers/plans/2026-07-17-structurezyme-unification.md
git commit -m "chore: commit in-flight docstring + project-notes before handoff cleanup"
```

- [ ] **Step 4: Verify the tree is clean**

Run: `git status -s`
Expected: empty output (the design spec from the previous step is already committed).

---

### Task 2: Make the test suite green on a fresh checkout

**Files:**
- Create: `pyproject.toml`
- Modify: `structurezyme/steps/PLACER_step.py:182-196`
- Modify: `tests/test_placer_step.py:94-116`
- Modify: `tools/test_pipeline.py:62` (and old output-dir strings at lines 59, 98)
- Test: `tests/test_placer_step.py`, `tools/test_pipeline.py`

**Interfaces:**
- Consumes: current `PLACER.__init__` signature (validates `placer_script_path`, `placer_env_path`).
- Produces: `_placer_path_exists`-style tolerant checks; unchanged public `PLACER` API.

- [ ] **Step 1: Add pyproject.toml restricting default test collection to tests/**

Create `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Verify default collection now excludes tools/**

Run: `/mnt/storage01/home/lherrmann/envs/filterzyme/bin/python -m pytest --collect-only -q 2>/dev/null | tail -5`
Expected: only `tests/...` items listed; no `tools/test_pipeline.py` items.

- [ ] **Step 3: Write/confirm the failing PLACER tests use tmp_path (not /nonexistent)**

In `tests/test_placer_step.py`, replace the hardcoded `/nonexistent` paths so the "not found" branch is exercised via a path under `tmp_path` that truly does not exist and is statable:

```python
def test_placer_init_raises_on_missing_script(tmp_path):
    env_root = _make_fake_env(tmp_path)
    missing_script = tmp_path / "run_PLACER.py"  # does not exist, but statable
    with pytest.raises(FileNotFoundError, match="PLACER script not found"):
        PLACER(
            preparedfiles_dir=tmp_path,
            output_dir=tmp_path,
            predict_ligand="LIG",
            placer_script_path=str(missing_script),
            placer_env_path=str(env_root),
        )


def test_placer_init_raises_on_missing_env(tmp_path):
    fake_script = tmp_path / "fake_run_PLACER.py"
    fake_script.touch()
    missing_env = tmp_path / "no_env"  # no bin/python under it
    with pytest.raises(FileNotFoundError, match="PLACER env python not found"):
        PLACER(
            preparedfiles_dir=tmp_path,
            output_dir=tmp_path,
            predict_ligand="LIG",
            placer_script_path=str(fake_script),
            placer_env_path=str(missing_env),
        )
```

- [ ] **Step 4: Run the two tests to confirm they fail for the RIGHT reason**

Run: `/mnt/storage01/home/lherrmann/envs/filterzyme/bin/python -m pytest tests/test_placer_step.py::test_placer_init_raises_on_missing_script tests/test_placer_step.py::test_placer_init_raises_on_missing_env -v`
Expected: PASS now if `tmp_path` paths are statable on this FS. If they still error with `PermissionError` instead of `FileNotFoundError`, proceed to Step 5 (the real fix is in the step code).

- [ ] **Step 5: Make PLACER validation tolerate OSError/PermissionError**

In `structurezyme/steps/PLACER_step.py`, replace the two `.exists()` guards (currently lines 182-196) with a helper that treats an un-statable path as "not found":

```python
def _path_exists(p: Path) -> bool:
    """True if the path exists; treat un-statable paths (PermissionError on
    shared filesystems) as 'does not exist' rather than propagating OSError."""
    try:
        return p.exists()
    except OSError:
        return False
```

Then in `__init__`:

```python
        script = Path(placer_script_path)
        if not _path_exists(script):
            raise FileNotFoundError(
                f"PLACER script not found at {placer_script_path}. "
                "Set placer_script_path explicitly or install PLACER."
            )
        self.placer_script_path = script

        env_python = Path(placer_env_path) / "bin" / "python"
        if not _path_exists(env_python):
            raise FileNotFoundError(
                f"PLACER env python not found at {env_python}. "
                "Set placer_env_path explicitly or install PLACER."
            )
        self.placer_env_python = env_python
```

- [ ] **Step 6: Fix the stale tools/test_pipeline.py assertion + output-dir names**

In `tools/test_pipeline.py`, line 62 asserts an attribute the current `Pipeline` no longer stores. `Pipeline.__init__` accepts `squidly_model_size` but does not set `self.squidly_model_size`. Replace the attribute assertion with a construction-only check, and update the two `filterzyme_test_*` output-dir strings:

```python
    # line 59
        base_output_dir="/tmp/structurezyme_test_output",
    )
    assert pipeline is not None
    # (removed stale `assert pipeline.squidly_model_size == "3B"`; the
    #  attribute is no longer stored on the deprecated Pipeline shim.)
```
```python
    # line 98
            base_output_dir="/tmp/structurezyme_test_placer_missing_ligand",
```
Also skim the other 3 currently-failing `tools/test_pipeline.py` tests (`test_pipeline_run_placer_false_skips_placer`, `test_pipeline_run_placer_true_invokes_execute`, `test_pipeline_forwards_fastrelax_kwargs_to_superimposition`) and update any assertion that references a removed attribute; keep signature-based checks (they still pass).

- [ ] **Step 7: Run BOTH suites and confirm zero failures**

Run: `/mnt/storage01/home/lherrmann/envs/filterzyme/bin/python -m pytest tests/ tools/test_pipeline.py -q 2>&1 | tail -5`
Expected: `0 failed`; skips only for GPU-gated smoke tests. Paste the summary.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml structurezyme/steps/PLACER_step.py tests/test_placer_step.py tools/test_pipeline.py
git commit -m "test: green suite on fresh checkout (pytest testpaths, tolerant PLACER path checks, fix stale pipeline smoke test)"
```

---

### Task 3: Parameterize the conda-env activation in all SLURM scripts

**Files:**
- Modify: `tools/gpu_run/run_allmodules.sbatch:24`
- Modify: `tools/gpu_run/run_multirow.sbatch:23`
- Modify: `tools/gpu_run/resume_multirow.sbatch:27`
- Modify: `tools/fmo18/run_fmo18.sbatch:27` (and comment at line 20)
- Modify: `tools/fmo18/submit_fmo18.sbatch:11,17` (and the personal `WORKTREE` at line 12)
- Modify: `tools/smoke/run_phase_c_smoke.sbatch:13-14`
- Modify: `tools/smoke/run_smoke_on_gpu.sh:4`
- Modify: `tools/gpu_run/CHECKLIST.md:22`

**Interfaces:**
- Consumes: nothing.
- Produces: sbatch scripts that any user can submit without editing.

- [ ] **Step 1: Enumerate every hardcoded personal env path**

Run: `grep -rn "home/lherrmann/envs/filterzyme\|\$HOME/envs/filterzyme" tools/`
Expected: matches in the files listed above. Handle each.

- [ ] **Step 2: Replace `export PATH=/mnt/storage01/home/lherrmann/envs/filterzyme/bin:$PATH` in the simple sbatch scripts**

In `run_allmodules.sbatch`, `run_multirow.sbatch`, `resume_multirow.sbatch`, `run_fmo18.sbatch`, `run_phase_c_smoke.sbatch`, and `run_smoke_on_gpu.sh`, replace that single line with:

```bash
STRUCTUREZYME_ENV="${STRUCTUREZYME_ENV:-/mnt/labs/data/mora/envs/structurezyme}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$STRUCTUREZYME_ENV"
```

- [ ] **Step 3: Fix `submit_fmo18.sbatch` (has personal env AND worktree path)**

Replace lines 11-17 so both the env and the worktree are overridable and default to non-personal paths:

```bash
STRUCTUREZYME_ENV="${STRUCTUREZYME_ENV:-/mnt/labs/data/mora/envs/structurezyme}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$STRUCTUREZYME_ENV"
ENV_PY="$(command -v python)"
# Repo root: override with STRUCTUREZYME_REPO if not submitting from the checkout.
WORKTREE="${STRUCTUREZYME_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
```
Leave the rest of the script (the `"$ENV_PY" -m structurezyme.cli run ...`) intact.

- [ ] **Step 4: Update comments referencing the personal env**

In `run_fmo18.sbatch:20` and `tools/gpu_run/CHECKLIST.md:22`, change "env installed at /mnt/storage01/home/lherrmann/envs/filterzyme" to "env `structurezyme` (default `/mnt/labs/data/mora/envs/structurezyme`, override with `$STRUCTUREZYME_ENV`)".

- [ ] **Step 5: Verify no personal env paths remain in tools/**

Run: `grep -rn "home/lherrmann/envs/filterzyme\|\$HOME/envs/filterzyme" tools/ || echo CLEAN`
Expected: `CLEAN`.

- [ ] **Step 6: Syntax-check the sbatch scripts (bash -n)**

Run: `for f in tools/gpu_run/*.sbatch tools/fmo18/*.sbatch tools/smoke/*.sbatch tools/smoke/*.sh; do bash -n "$f" && echo "OK $f"; done`
Expected: `OK` for each.

- [ ] **Step 7: Commit**

```bash
git add tools/
git commit -m "fix(tools): overridable STRUCTUREZYME_ENV activation; drop hardcoded personal env paths"
```

---

### Task 4: Finish rename cosmetics (.gitignore, benchmark output strings, comment)

**Files:**
- Modify: `.gitignore:29-31`
- Modify: benchmark `run_*.py` with `base_output_dir="filterzyme_output"` (PBP_binding/run_PBP.py:26, PDE_2H/run_PDE_2H.py, martinez/run_martinez.py, serine_hydrolases/run_serine_hydrolases.py, and any others found)
- Modify: `structurezyme/steps/fpocket_step.py:20`

**Interfaces:**
- Consumes: nothing.
- Produces: no functional change; cosmetic consistency.

- [ ] **Step 1: Extend .gitignore with structurezyme_output patterns (keep old)**

After the existing `*filterzyme_output*` lines, add:

```
*structurezyme_output/
*structurezyme_output_yang_benchmark/
*structurezyme_output_test/
```

- [ ] **Step 2: List benchmark scripts that hardcode the old output dir**

Run: `grep -rn 'base_output_dir *= *"filterzyme_output' benchmarking/`
Expected (7 lines across `run_*.py`): `martinez/run_martinez.py:16`, `PDE_2H/run_PDE_2H.py:25`, `metallohydrolases/run_metallohydrolases.py:17`, `PBP_binding/run_PBP.py:26`, `serine_hydrolases/run_serine_hydrolases.py:17`, `yang_filterzyme/run_yang_benchmark.py:16` (the `_yang_benchmark` variant). Do NOT edit `test_placer/` — it is removed in Task 6.

- [ ] **Step 3: Rename the output-dir string in each**

In each file change `base_output_dir = "filterzyme_output"` to `base_output_dir = "structurezyme_output"`, and in `yang_filterzyme/run_yang_benchmark.py` change `"filterzyme_output_yang_benchmark"` to `"structurezyme_output_yang_benchmark"`. (These scripts have other dead personal paths documented in Task 6's `benchmarking/README.md` note; do not chase those here.)

- [ ] **Step 4: Tidy the personal-path comment in fpocket_step.py:20**

Replace the `/home/helen/...` example path in the comment with a generic placeholder (e.g. `<fpocket_output_dir>`), leaving code unchanged.

- [ ] **Step 5: Confirm no functional references to a `filterzyme` python module remain outside the shim**

Run: `grep -rn "import filterzyme\|from filterzyme" --include=*.py . | grep -v "filterzyme/__init__.py"`
Expected: only test files that intentionally test the shim (e.g. `tests/test_deprecation_shim.py`). No production `structurezyme/` code.

- [ ] **Step 6: Commit**

```bash
git add .gitignore benchmarking/ structurezyme/steps/fpocket_step.py
git commit -m "chore: finish filterzyme->structurezyme cosmetic rename (gitignore, benchmark output dirs, comment)"
```

---

### Task 5: Unify installability & reproducibility

**Files:**
- Modify: `environment.yml:22-23,36`
- Modify: `README.md:28-34`
- Modify: `docs/getting_started.md:54-60`
- Modify: `docs/known-issues.md` (squidly fork URL, ~line 22)

**Interfaces:**
- Consumes: nothing.
- Produces: one consistent install method; a self-consistent dependency set.

- [ ] **Step 1: Pin enzymetk and add cuequivariance_torch in environment.yml**

In `environment.yml` under the `pip:` block: change `- enzymetk` to `- enzymetk==0.1.0`, and add `- cuequivariance_torch==0.10.0` (place it near the other GPU/torch deps). Remove the stale `# - filterzyme  # removed...` comment line (line 23).

- [ ] **Step 2: Repoint the squidly pin to the moragroup org**

In `environment.yml`, change the squidly line to:
```yaml
    - squidly @ git+https://github.com/moragroup/Squidly.git@022fa40  # catalytic-residue prediction CLI; requires GPU for ESM2 inference
```
Keep the surrounding explanatory comment block. **NOTE:** the org fork is a user action (see Task 8). If the org repo does not yet exist at env-build time, temporarily keep `HerrLuca99/Squidly@022fa40` and flag it — the repoint is a one-line change that does not require an env rebuild.

- [ ] **Step 3: Fix README installation block**

Replace `README.md` lines 28-34 with:
```bash
conda env create -f environment.yml
conda activate structurezyme
pip install -e .
```
(Removes the `setup.py sdist bdist_wheel` / `--use-deprecated=legacy-resolver` / hardcoded `structurezyme-0.1.0.tar.gz` / `enzymetk==0.0.8` lines. enzymetk is now pinned in environment.yml.)

- [ ] **Step 4: Remove the `pip install squidly` line from getting_started**

In `docs/getting_started.md`, delete line 59 (`pip install squidly`) — it pulls unpatched upstream and reintroduces the empty-residue bug. Keep the weight-download command that follows, but note squidly is installed by `environment.yml` (the pinned fork). Adjust the surrounding text so it no longer instructs a separate `pip install squidly`.

- [ ] **Step 5: Update the squidly fork URL in known-issues.md**

In `docs/known-issues.md`, change the fork URL from `github.com/HerrLuca99/Squidly` to `github.com/moragroup/Squidly` (commit `022fa40`, same branch), matching environment.yml. Keep the follow-up note about repointing to real upstream once merged.

- [ ] **Step 6: Sanity-check docs consistency**

Run: `grep -rn "setup.py sdist\|enzymetk==0.0.8\|pip install squidly\|HerrLuca99" README.md docs/ environment.yml || echo CONSISTENT`
Expected: `CONSISTENT` (no stale install instructions remain).

- [ ] **Step 7: Commit**

```bash
git add environment.yml README.md docs/getting_started.md docs/known-issues.md
git commit -m "docs+env: unify install on 'pip install -e .', pin enzymetk/cuequivariance, repoint squidly to moragroup org"
```

---

### Task 6: Repo hygiene (scratch removal + notes)

**Files:**
- Remove: `test_placer/` (tracked scratch duplicate of `benchmarking/metallohydrolases/`)
- Create: `benchmarking/README.md`
- Create: `examples/README.md`

**Interfaces:**
- Consumes: nothing.
- Produces: a leaner tree with orientation notes for new users.

- [ ] **Step 1: Confirm test_placer/ is a scratch duplicate**

Run: `git ls-files test_placer/ && diff -rq test_placer/metallohydrolases benchmarking/metallohydrolases 2>&1 | head`
Expected: it is tracked and largely duplicates `benchmarking/metallohydrolases/`. If it contains anything unique/important, STOP and ask before removing.

- [ ] **Step 2: Remove test_placer/ from git**

```bash
git rm -r test_placer/
```

- [ ] **Step 3: Add benchmarking/README.md historical note**

Create `benchmarking/README.md`:
```markdown
# Benchmarking scripts (historical)

These per-target scripts (`martinez/`, `metallohydrolases/`, `PBP_binding/`,
`PDE_2H/`, `serine_hydrolases/`, `yang_filterzyme/`, `yang_paper/`) are the
original benchmark runs used during development. They contain **dead,
machine-specific paths** (e.g. `/nvme2/helen/...`, old `Filterzyme` paths,
`conda activate filterzyme`) and are **not** guaranteed to run out of the box.

To reuse one: edit the hardcoded input/output paths and env name for your
environment, and prefer the current CLI (`structurezyme run --config ...`) over
the legacy `Pipeline` API where possible.
```

- [ ] **Step 4: Add examples/README.md note**

Create `examples/README.md`:
```markdown
# Examples

- `DEHP-MEHP.pkl` — a small example input DataFrame (DEHP -> MEHP substrate
  case) kept as a ready-made pipeline input for quick manual testing.
```

- [ ] **Step 5: Verify tree**

Run: `git status -s && ls test_placer 2>&1 | head -1`
Expected: `test_placer/` deletion staged; directory gone from working tree.

- [ ] **Step 6: Commit**

```bash
git add -A benchmarking/README.md examples/README.md
git commit -m "chore(hygiene): remove test_placer scratch dir; add benchmarking/examples orientation notes"
```

---

### Task 7: Recreate the `structurezyme` conda env and verify

**Files:** none (environment + verification only).

**Interfaces:**
- Consumes: finalized `environment.yml`, `setup.py`, `pyproject.toml`.
- Produces: a working `structurezyme` conda env proving the shipped spec.

> This task validates that a fresh lab user can build and use the project. It does NOT need a GPU for install + unit tests + dry-runs. A GPU is only needed for the optional end-to-end pipeline run (Step 6) — ask the user for a GPU node when you reach it.

- [ ] **Step 1: Create the env from environment.yml**

Run: `conda env create -f environment.yml -n structurezyme`
Expected: env solves and installs. If the squidly `moragroup` URL 404s (org fork not done yet), temporarily use the `HerrLuca99` URL, note it, and continue.

- [ ] **Step 2: Editable-install the package**

Run: `conda run -n structurezyme pip install -e .`
Expected: `Successfully installed structurezyme-0.1.0`.

- [ ] **Step 3: Import + CLI smoke check**

Run: `conda run -n structurezyme python -c "import structurezyme, structurezyme.pipeline; print(structurezyme.__version__)"` then `conda run -n structurezyme structurezyme --help`
Expected: prints `0.1.0`; `--help` lists `init/run/resume/status/submit`.

- [ ] **Step 4: Run the full unit-test suite in the NEW env**

Run: `conda run -n structurezyme python -m pytest tests/ tools/test_pipeline.py -q 2>&1 | tail -8`
Expected: `0 failed`, only GPU-gated skips. **Paste the summary line.** Do not proceed until green.

- [ ] **Step 5: Dry-run the SLURM submit path (no GPU needed)**

Run: `conda run -n structurezyme structurezyme init --output /tmp/sz_run.yml && conda run -n structurezyme structurezyme submit --config /tmp/sz_run.yml --host default --dry-run 2>&1 | head -30`
Expected: a rendered sbatch script printed to stdout, referencing `structurezyme run` — no submission.

- [ ] **Step 6 (GPU, optional but recommended): end-to-end sanity run**

Ask the user for a GPU node. Then run the smallest available end-to-end harness (e.g. `tools/smoke/`), overriding the env:
`STRUCTUREZYME_ENV=<path-to-structurezyme-env> sbatch tools/gpu_run/run_allmodules.sbatch` (or the smoke variant).
Expected: pipeline runs through the enabled steps and writes a manifest. Paste the manifest/log tail. If no GPU is available, mark this step deferred and record it as a follow-up.

- [ ] **Step 7: Record the verification results in the plan/summary**

No commit (env is external to git). Capture the pytest summary + CLI/dry-run output into the final change summary for the user.

---

### Task 8: User-action follow-ups & prepare-for-move (no move, no GitHub)

These require user credentials/decisions and are NOT performed autonomously.

- [ ] **Step 1: squidly org fork (USER)**

User forks `github.com/HerrLuca99/Squidly` (or the branch `fix/forward-mean-prob-mean-var-cli`, commit `022fa40`) into `github.com/moragroup/Squidly`. Once done, confirm `environment.yml` + `known-issues.md` already point there (Task 5). If the org fork uses a different commit/branch, update the pin accordingly and re-run Task 7 Step 1-4.

- [ ] **Step 2: Delete stale checkout + old env (only after Task 7 is green)**

After the new env verifies green and the user confirms:
```bash
rm -rf /mnt/storage01/home/lherrmann/StructureZyme
conda env remove -n filterzyme   # or: conda env remove -p /mnt/storage01/home/lherrmann/envs/filterzyme
```
Do this ONLY on explicit user confirmation.

- [ ] **Step 3: Recommend the lab destination (do NOT move)**

Host profiles already default `output_root` etc. to `/mnt/labs/data/mora/...`. Recommend placing the shared repo at a sibling code location under `/mnt/labs/data/mora/` (e.g. `/mnt/labs/data/mora/code/structurezyme`) and the shared env at `/mnt/labs/data/mora/envs/structurezyme` (the sbatch default). Present this to the user; the user performs the actual copy/move.

- [ ] **Step 4: Produce the change summary for user review**

Summarize all commits on `chore/lab-handoff`, the verification evidence, and the outstanding user actions (org fork, GPU e2e if deferred, the move). **Do not push, merge, or open a PR without explicit approval.**

---

## Self-Review

Coverage vs spec:
- Rename completeness -> Tasks 3 (env/sbatch), 4 (cosmetics), 5 (env.yml). ✅
- Multi-user readiness -> already solved in-repo; sbatch personal-path leak fixed in Task 3. ✅
- Installability/reproducibility -> Task 5 (docs/pins/squidly) + Task 7 (fresh-env proof). ✅
- Repo hygiene -> Task 6. ✅
- Test suite green -> Task 2 + verified in Task 7. ✅
- Squidly personal-account risk -> Task 5 + Task 8 Step 1. ✅
- No move / no GitHub without approval -> Global Constraints + Task 8. ✅

Out of scope (documented, not planned): best-pose energy-awareness beyond the merged `fused_rank`, multi-substrate schema, bring-your-own-PDB single-module runs.

Placeholder scan: no "TBD/TODO/handle edge cases"; every code/command step has concrete content. Type/name consistency: `_path_exists` helper defined in Task 2 Step 5 and used there; `STRUCTUREZYME_ENV`/`STRUCTUREZYME_REPO` names consistent across Task 3; env name `structurezyme` and pins (`enzymetk==0.1.0`, `cuequivariance_torch==0.10.0`) consistent across Tasks 5 and 7.
