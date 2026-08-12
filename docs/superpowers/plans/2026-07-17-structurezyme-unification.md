# StructureZyme Unification & Modular Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the Filterzyme project to StructureZyme uniformly and turn its docking/filtering pipeline into a config-driven, resumable, multi-user system where any module can be included or excluded and intermediate outputs are reused after a crash or when re-running a previously skipped step.

**Architecture:** A single canonical `structurezyme` Python package exposes a step registry plus a `Runner` that executes steps in dependency order. Each step reads/writes a pickle checkpoint under a per-user, per-run directory; a `manifest.json` records status and input/config hashes so a step is skipped when its checkpoint is valid. A Pydantic `RunConfig` (loaded from YAML) declares which steps are enabled and their options; a CLI drives `run`, `resume`, `status`, `step`, `init`, and Slurm `submit`.

**Tech Stack:** Python >=3.10, pandas, Pydantic v2, PyYAML, filelock, pytest; existing scientific deps (enzymetk, Chai, Boltz, AutoDock Vina/docko, Squidly, PLACER, Fpocket, PLIP, freesasa, RDKit, Biotite/BioPython).

## Global Constraints

- Python floor: `python_requires=">=3.10"` (fix the existing `ppython_requires` typo).
- Package name is exactly `structurezyme`; distribution name is exactly `structurezyme`; version `0.1.0`.
- All GitHub URLs point to `https://github.com/moragroup/StructureZyme`.
- Conda environment name is exactly `structurezyme`.
- Keep a deprecation shim package `filterzyme` that re-exports `structurezyme` and raises `DeprecationWarning`; scheduled for removal in `0.2.0`.
- Preserve existing on-disk sub-directory names (`docking/chai/`, `docking/boltz/`, `superimposition/preparedfiles_for_superimposition/`, etc.) so prior benchmarking outputs stay consumable.
- Never write outputs to the current working directory; all run artifacts live under `{output_root}/{user}/{run_id}/`.
- Log file is named `structurezyme.log` and lives in the run's `logs/` directory.
- Work in a fresh clone of `moragroup/StructureZyme` at branch `lab-sanity-run_LCH`, on a new branch `refactor/structurezyme-modular`.
- TDD: every code task writes a failing test first. Commit after each task.

---

## File Structure

New / changed files, each with a single responsibility:

- `structurezyme/` — renamed package (was `filterzyme/`). Canonical code.
- `structurezyme/__init__.py` — version string only.
- `structurezyme/pipeline.py` — was `pipeline_v2.py`; keeps `Docking`, `Superimposition`, `GeometricFilters` phase classes and a backward-compatible `Pipeline` adapter.
- `structurezyme/config.py` — Pydantic `RunConfig`, `StepsConfig`, `PathsConfig`, `RuntimeConfig`; `load_config`/`write`.
- `structurezyme/registry.py` — `StepSpec` dataclass and `STEPS` ordered registry mapping step name -> inputs/output/runner/deps.
- `structurezyme/runner.py` — `RunContext`, `Manifest`, `Runner` (checkpoint discovery, hashing, resume).
- `structurezyme/hashing.py` — canonical hashing of config sub-trees and pickle inputs.
- `structurezyme/paths.py` — run directory layout helpers.
- `structurezyme/hosts.py` + `structurezyme/hosts.yml` — per-host default paths (Boltz cache, Squidly weights, PLACER env).
- `structurezyme/cli.py` — argparse CLI: `run`, `resume`, `status`, `step`, `init`, `submit`.
- `structurezyme/templates/slurm.sbatch.j2` — Slurm job template.
- `structurezyme/steps/*_step.py` — existing steps, refactored to a uniform runner signature.
- `filterzyme/__init__.py` — deprecation shim re-exporting `structurezyme`.
- `tests/…` — pytest unit + smoke tests (new).
- `tools/` — relocated root-level runner scripts.
- `docs/…` — rewritten documentation.

---

## Phase 0: Baseline

### Task 0: Clone, branch, and verify the newest pipeline runs

**Files:**
- Create: (working tree) `/mnt/labs/data/mora/code/StructureZyme`

**Interfaces:**
- Produces: a clean checkout on branch `refactor/structurezyme-modular` and a confirmed-working baseline before any refactor.

- [ ] **Step 1: Clone the newest branch**

```bash
git clone -b lab-sanity-run_LCH git@github.com:moragroup/StructureZyme.git /mnt/labs/data/mora/code/StructureZyme
cd /mnt/labs/data/mora/code/StructureZyme
```

- [ ] **Step 2: Create the working branch**

```bash
git switch -c refactor/structurezyme-modular
```

- [ ] **Step 3: Create the conda environment**

```bash
conda env create -f environment.yml || conda env update -f environment.yml
conda activate filterpipeline2
python setup.py sdist bdist_wheel
pip install dist/filterzyme-0.0.6.tar.gz --use-deprecated=legacy-resolver
pip install enzymetk==0.0.8
```

- [ ] **Step 4: Run the quickstart example as a baseline smoke test**

Run: `python docs/examples/00_quickstart.py --help`
Expected: argparse help prints without import errors. If a GPU is available, run the minimal example to completion and confirm `structural_features_final.pkl` is produced.

- [ ] **Step 5: Record the baseline**

```bash
git log --oneline -1 > /tmp/opencode/structurezyme_baseline_commit.txt
```

Expected: the current HEAD commit is captured for reference. No commit needed (read-only verification).

---

## Phase A: Rename Filterzyme -> StructureZyme

### Task A1: Rename the Python package directory

**Files:**
- Modify (rename): `filterzyme/` -> `structurezyme/`

**Interfaces:**
- Produces: package importable as `structurezyme` (imports still broken until A2).

- [ ] **Step 1: Git-move the package**

```bash
git mv filterzyme structurezyme
```

- [ ] **Step 2: Delete the legacy v1 pipeline**

```bash
git rm structurezyme/pipeline.py
```

- [ ] **Step 3: Rename v2 to the canonical name**

```bash
git mv structurezyme/pipeline_v2.py structurezyme/pipeline.py
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: rename filterzyme package to structurezyme and drop pipeline v1"
```

### Task A2: Update all internal imports and references

**Files:**
- Modify: every `.py` under `structurezyme/`, `docs/examples/*.py`, `run_phase_c_smoke.py`, `test_pipeline.py`, `tests/*`, `benchmarking/**/*.py`

**Interfaces:**
- Consumes: renamed `structurezyme` package from A1.
- Produces: all modules import `from structurezyme...` and `import structurezyme`.

- [ ] **Step 1: Find remaining references**

Run: `grep -rIn "filterzyme" --include="*.py" .`
Expected: a list of files still importing/naming `filterzyme`.

- [ ] **Step 2: Rewrite Python imports**

```bash
grep -rIl "filterzyme" --include="*.py" . | xargs sed -i 's/\bfilterzyme\b/structurezyme/g'
```

- [ ] **Step 3: Verify the package imports**

Run: `python -c "import structurezyme; from structurezyme.pipeline import Pipeline; print('ok')"`
Expected: prints `ok` with no ImportError.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: update all imports to structurezyme"
```

### Task A3: Update packaging metadata

**Files:**
- Modify: `setup.py`
- Modify: `structurezyme/__init__.py`

**Interfaces:**
- Produces: distribution `structurezyme==0.1.0` with entry point `structurezyme = structurezyme.cli:main`.

- [ ] **Step 1: Set the version**

In `structurezyme/__init__.py` set:

```python
__version__ = "0.1.0"
```

- [ ] **Step 2: Rewrite setup.py fields**

In `setup.py`: set `name='structurezyme'`; change `url` and `project_urls` to `https://github.com/moragroup/StructureZyme`; fix `ppython_requires` to `python_requires=">=3.10"`; add:

```python
      entry_points={
          'console_scripts': [
              'structurezyme = structurezyme.cli:main'
          ]
      },
```

and ensure `install_requires` includes `'pydantic>=2'`, `'pyyaml'`, `'filelock'`.

- [ ] **Step 3: Verify metadata parses**

Run: `python setup.py --name --version`
Expected: prints `structurezyme` then `0.1.0`.

- [ ] **Step 4: Commit**

```bash
git add setup.py structurezyme/__init__.py
git commit -m "build: rename distribution to structurezyme 0.1.0 and add CLI entry point"
```

### Task A4: Rename environment and log filename

**Files:**
- Modify: `environment.yml`
- Modify: `structurezyme/utils/helpers.py` (the `log_usage` default `log_file`)

**Interfaces:**
- Produces: env `structurezyme`; default usage log named `structurezyme_usage.log`.

- [ ] **Step 1: Rename env and enable pip deps**

In `environment.yml` change `name: filterpipeline2` to `name: structurezyme` and uncomment the `pip:` block, adding `pydantic`, `pyyaml`, `filelock`.

- [ ] **Step 2: Rename the usage log default**

In `structurezyme/utils/helpers.py`, change the `log_usage` signature default `log_file: str = "filterzyme_usage.log"` to `log_file: str = "structurezyme_usage.log"`.

- [ ] **Step 3: Verify**

Run: `grep -rn "filterpipeline2\|filterzyme_usage" environment.yml structurezyme/`
Expected: no matches.

- [ ] **Step 4: Commit**

```bash
git add environment.yml structurezyme/utils/helpers.py
git commit -m "refactor: rename conda env and usage log to structurezyme"
```

### Task A5: Add the deprecation shim package

**Files:**
- Create: `filterzyme/__init__.py`
- Test: `tests/test_deprecation_shim.py`

**Interfaces:**
- Produces: `import filterzyme` works, re-exports `structurezyme`, and emits `DeprecationWarning`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_deprecation_shim.py
import warnings

def test_filterzyme_shim_warns_and_reexports():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import filterzyme
        from filterzyme.pipeline import Pipeline  # noqa: F401
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    import structurezyme
    assert filterzyme.__version__ == structurezyme.__version__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_deprecation_shim.py -v`
Expected: FAIL (no `filterzyme` package).

- [ ] **Step 3: Create the shim**

```python
# filterzyme/__init__.py
import warnings as _warnings
from structurezyme import *  # noqa: F401,F403
from structurezyme import __version__  # noqa: F401

_warnings.warn(
    "`filterzyme` is deprecated; import `structurezyme` instead. "
    "The `filterzyme` alias will be removed in 0.2.0.",
    DeprecationWarning,
    stacklevel=2,
)

import importlib as _importlib
import sys as _sys

def __getattr__(name):
    module = _importlib.import_module(f"structurezyme.{name}")
    _sys.modules[f"filterzyme.{name}"] = module
    return module
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_deprecation_shim.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add filterzyme/__init__.py tests/test_deprecation_shim.py
git commit -m "feat: add filterzyme deprecation shim re-exporting structurezyme"
```

---

## Phase D: Repo Hygiene (run right after rename)

### Task D1: Untrack build artifacts, vendored binaries, and stray files

**Files:**
- Modify: `.gitignore`
- Remove from tracking (keep on disk where useful): `openbabel-3.1.1/`, `openbabel-3.1.1-source.tar.bz2`, `=2024.03`, `rrrr/`, `23/`, `af3/`, `mol/`, `dist/`, `build/`, `filterzyme.egg-info/`, `boltz_cache/`, `boltz_results_example/`, `nohup.out`, `.kilo/`, `benchmarking/**/logs/`, `benchmarking/**/*_output_old/`

**Interfaces:**
- Produces: a clean `git status` and a repo ~30 MB smaller.

- [ ] **Step 1: Append ignore rules**

Add to `.gitignore`:

```gitignore
# Build artifacts
build/
dist/
*.egg-info/

# Vendored / large binaries (install via package manager instead)
openbabel-3.1.1/
openbabel-3.1.1-source.tar.bz2

# Stale bundled Squidly weights (unused: weights come from the `squidly`
# pip package's own HuggingFace download, see Task D2)
structurezyme/squidly_final_models/

# Caches and run outputs
boltz_cache/
boltz_results_example/
*.log
nohup.out
=2024.03

# Editor / tooling
.kilo/

# Stray scratch dirs
rrrr/
23/
af3/
mol/

# Benchmarking outputs
benchmarking/**/logs/
benchmarking/**/*_output_old/
```

- [ ] **Step 2: Remove from the index (retain files on disk)**

```bash
git rm -r --cached --ignore-unmatch build dist filterzyme.egg-info openbabel-3.1.1 \
  openbabel-3.1.1-source.tar.bz2 "=2024.03" rrrr 23 af3 mol boltz_cache \
  boltz_results_example nohup.out .kilo 2>/dev/null || true
# The whole squidly_final_models/ tree is stale and unused by the new
# squidly_step.py (which shells out to the `squidly` CLI). Untrack it.
git rm -r --cached --ignore-unmatch structurezyme/squidly_final_models 2>/dev/null || true
```

- [ ] **Step 3: Verify the tree is clean**

Run: `git status --short`
Expected: only `.gitignore` modification and deletions of the untracked-now-ignored paths; no stray binaries staged.

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: stop tracking build artifacts, vendored openbabel, and scratch dirs"
```

### Task D2: Verify and document Squidly weight provenance (NO custom downloader)

**Context / decision:** StructureZyme does **not** load `.pth` files directly and must **not** ship its own weight downloader. The refactored `structurezyme/steps/squidly_step.py` shells out to the `squidly` CLI via `enzymetk.predict_catalyticsite_step.ActiveSitePred`. The `squidly` CLI locates its own model ensemble inside the installed package at `site-packages/squidly/models/{3B,15B}/` (files `CataloDB_esm2_t36_3B_UR50D_{CR_*,LSTM_*}` and the 15B equivalents). Those files are populated by the `squidly` package's own `download_models_hf.py`, which runs `snapshot_download(repo_id="WillRieger/Squidly", ...)` — the exact command already documented in the README. Therefore weights come from the `squidly` pip package + its HuggingFace download; the bundled `filterzyme/squidly_final_models/*.pth` are stale and unused (untracked in D1). This task only verifies provenance and documents the one-time setup command; it creates NO new Python module.

**Files:**
- Modify: `docs/getting_started.md` (weights setup section)
- Test: `tests/test_no_custom_weight_downloader.py`

**Interfaces:**
- Produces: a documented, verifiable weight-setup step; a guard test ensuring no stale bundled-weights path or bespoke downloader is reintroduced.

- [ ] **Step 1: Write the guard test**

```python
# tests/test_no_custom_weight_downloader.py
import shutil
from pathlib import Path
import structurezyme

def test_no_bespoke_weight_downloader_module():
    pkg_dir = Path(structurezyme.__file__).parent
    assert not (pkg_dir / "download_weights.py").exists(), (
        "Weights come from the `squidly` package's own HF download; "
        "do not add a bespoke downloader."
    )

def test_squidly_step_does_not_reference_bundled_pth():
    pkg_dir = Path(structurezyme.__file__).parent
    src = (pkg_dir / "steps" / "squidly_step.py").read_text()
    assert "squidly_final_models" not in src
    assert ".pth" not in src

def test_squidly_cli_available_message_is_documented():
    # The step must fail loudly (clear message) if the CLI is missing,
    # rather than silently trying to load local weights.
    src = (Path(structurezyme.__file__).parent / "steps" / "squidly_step.py").read_text()
    assert "shutil.which(\"squidly\")" in src or "shutil.which('squidly')" in src
```

- [ ] **Step 2: Run test to verify current state**

Run: `pytest tests/test_no_custom_weight_downloader.py -v`
Expected: the first two assertions PASS after D1 removes `squidly_final_models/`; the third PASSES because `squidly_step.py` already guards on `shutil.which("squidly")`. If `test_squidly_step_does_not_reference_bundled_pth` FAILS, it means the stale path leaked back in — fix the step, do not add weights.

- [ ] **Step 3: Document the one-time weights setup in getting_started.md**

Add a "Catalytic-residue model weights (Squidly)" section stating:

```bash
# Weights are shipped and downloaded by the `squidly` package itself
# (HuggingFace repo WillRieger/Squidly). Run once after installing squidly:
pip install squidly
python -c "import squidly, os; os.system(f'python {os.path.dirname(squidly.__file__)}/download_models_hf.py')"

# Verify the ensemble landed in site-packages/squidly/models/{3B,15B}/:
python -c "import squidly, os, glob; d=os.path.join(os.path.dirname(squidly.__file__),'models'); print(sorted(glob.glob(d+'/*/*.p*')))"
```

Note explicitly that StructureZyme requires the `squidly` CLI on `$PATH` and does not manage weights itself; ESM2 inference needs a GPU.

- [ ] **Step 4: Verify the documented command resolves the weights on this host**

Run: `python -c "import squidly, os, glob; d=os.path.join(os.path.dirname(squidly.__file__),'models'); print(len(glob.glob(d+'/*/*.p*')), 'weight files')"`
Expected: prints a nonzero count (e.g. `20 weight files`) confirming the pip-package weights exist; no download needed if already present.

- [ ] **Step 5: Commit**

```bash
git add docs/getting_started.md tests/test_no_custom_weight_downloader.py
git commit -m "docs: document squidly-package weight provenance; guard against bespoke downloader"
```

### Task D3: Relocate root-level runner scripts into tools/

**Files:**
- Modify (rename): `run_phase_c_smoke.py` -> `tools/smoke/run_phase_c_smoke.py`
- Modify (rename): `run_phase_c_smoke.sbatch` -> `tools/smoke/run_phase_c_smoke.sbatch`
- Modify (rename): `_run_smoke_on_gpu.sh` -> `tools/smoke/run_smoke_on_gpu.sh`
- Modify (rename): `test_pipeline.py` -> `tools/test_pipeline.py`

**Interfaces:**
- Produces: a clean repo root; runner scripts grouped under `tools/`.

- [ ] **Step 1: Move the scripts**

```bash
mkdir -p tools/smoke
git mv run_phase_c_smoke.py tools/smoke/run_phase_c_smoke.py
git mv run_phase_c_smoke.sbatch tools/smoke/run_phase_c_smoke.sbatch
git mv _run_smoke_on_gpu.sh tools/smoke/run_smoke_on_gpu.sh
git mv test_pipeline.py tools/test_pipeline.py
```

- [ ] **Step 2: Fix any internal path references**

Run: `grep -rn "run_phase_c_smoke\|_run_smoke_on_gpu\|test_pipeline" . --include="*.py" --include="*.sh" --include="*.sbatch" --include="*.md"`
Expected: update any references to the new `tools/` paths.

- [ ] **Step 3: Verify scripts still resolve imports**

Run: `python -c "import ast; ast.parse(open('tools/smoke/run_phase_c_smoke.py').read()); print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: relocate smoke/runner scripts into tools/"
```

---

## Phase B: Modular, Resumable Pipeline

### Task B1: Canonical hashing helpers

**Files:**
- Create: `structurezyme/hashing.py`
- Test: `tests/test_hashing.py`

**Interfaces:**
- Produces:
  - `hash_obj(obj) -> str` — sha256 hex of any JSON-serializable object via canonical (sorted-key) JSON.
  - `hash_file(path) -> str` — sha256 hex of a file's bytes (streamed).
  - `hash_files(paths: list) -> str` — order-independent combined hash of multiple files.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hashing.py
from structurezyme.hashing import hash_obj, hash_file, hash_files

def test_hash_obj_is_key_order_independent():
    assert hash_obj({"a": 1, "b": 2}) == hash_obj({"b": 2, "a": 1})

def test_hash_obj_changes_with_value():
    assert hash_obj({"a": 1}) != hash_obj({"a": 2})

def test_hash_file_and_files(tmp_path):
    p1 = tmp_path / "a.txt"; p1.write_text("hello")
    p2 = tmp_path / "b.txt"; p2.write_text("world")
    assert hash_file(p1) == hash_file(p1)
    assert hash_files([p1, p2]) == hash_files([p2, p1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hashing.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement hashing**

```python
# structurezyme/hashing.py
import hashlib
import json
from pathlib import Path

def hash_obj(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def hash_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def hash_files(paths) -> str:
    digests = sorted(hash_file(p) for p in paths if Path(p).is_file())
    return hash_obj(digests)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hashing.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add structurezyme/hashing.py tests/test_hashing.py
git commit -m "feat: add canonical hashing helpers for checkpoint invalidation"
```

### Task B2: Run directory path helpers

**Files:**
- Create: `structurezyme/paths.py`
- Test: `tests/test_paths.py`

**Interfaces:**
- Produces:
  - `run_dir(output_root, user, run_id) -> Path`
  - `RunLayout` dataclass with attributes: `root`, `config_path`, `manifest_path`, `logs_dir`, `log_path`, `checkpoints_dir`; method `step_dir(name) -> Path` and `checkpoint_path(name) -> Path`.
  - `RunLayout.create() -> None` (mkdirs `logs_dir` and `checkpoints_dir`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paths.py
from pathlib import Path
from structurezyme.paths import run_dir, RunLayout

def test_run_dir_layout(tmp_path):
    rd = run_dir(tmp_path, "alice", "run1")
    assert rd == tmp_path / "alice" / "run1"

def test_runlayout_paths_and_create(tmp_path):
    layout = RunLayout(run_dir(tmp_path, "alice", "run1"))
    layout.create()
    assert layout.logs_dir.is_dir()
    assert layout.checkpoints_dir.is_dir()
    assert layout.manifest_path == layout.root / "manifest.json"
    assert layout.log_path == layout.logs_dir / "structurezyme.log"
    assert layout.checkpoint_path("chai") == layout.checkpoints_dir / "chai.pkl"
    assert layout.step_dir("docking") == layout.root / "docking"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_paths.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement paths**

```python
# structurezyme/paths.py
from dataclasses import dataclass
from pathlib import Path

def run_dir(output_root, user: str, run_id: str) -> Path:
    return Path(output_root) / user / run_id

@dataclass
class RunLayout:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)

    @property
    def config_path(self) -> Path:
        return self.root / "config.yml"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def log_path(self) -> Path:
        return self.logs_dir / "structurezyme.log"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    def step_dir(self, name: str) -> Path:
        return self.root / name

    def checkpoint_path(self, name: str) -> Path:
        return self.checkpoints_dir / f"{name}.pkl"

    def create(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_paths.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add structurezyme/paths.py tests/test_paths.py
git commit -m "feat: add per-user run directory layout helpers"
```

### Task B3: Pydantic RunConfig and YAML load/write

**Files:**
- Create: `structurezyme/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `StepConfig` (base): `enabled: bool = True`, plus arbitrary extra options (`model_config = ConfigDict(extra="allow")`).
  - `StepsConfig`: one `StepConfig`-typed field per step: `squidly, chai, boltz, vina, docking_metrics, prepare_files, fastrelax, superimpose, protein_rmsd, ligand_rmsd, geometric_filter, fpocket, ligand_sasa, plip, placer`.
  - `PathsConfig`: `output_root: str`, `boltz_cache_dir: str`, `squidly_weights_dir: str | None = None`, `placer_env_path: str = "/mnt/labs/data/mora/software/PLACER/env"`.
  - `RuntimeConfig`: `user: str` (default `getpass.getuser()`), `run_id: str` (default timestamp `%Y%m%d-%H%M%S`), `num_threads: int = 1`, `force: list[str] = []`.
  - `RunConfig`: `paths: PathsConfig`, `runtime: RuntimeConfig = RuntimeConfig()`, `steps: StepsConfig = StepsConfig()`; method `step_options(name) -> dict` (the enabled+options for a step) and `is_enabled(name) -> bool`.
  - `load_config(path) -> RunConfig`; `RunConfig.write(path) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from structurezyme.config import RunConfig, load_config

def _minimal():
    return RunConfig(paths={"output_root": "/tmp/out", "boltz_cache_dir": "/tmp/cache"})

def test_defaults_enable_all_steps():
    cfg = _minimal()
    assert cfg.is_enabled("chai") is True
    assert cfg.runtime.num_threads == 1
    assert cfg.runtime.run_id  # non-empty default

def test_disable_step():
    cfg = RunConfig(paths={"output_root": "/tmp/out", "boltz_cache_dir": "/tmp/cache"},
                    steps={"vina": {"enabled": False}})
    assert cfg.is_enabled("vina") is False

def test_step_options_roundtrip(tmp_path):
    cfg = RunConfig(paths={"output_root": "/tmp/out", "boltz_cache_dir": "/tmp/cache"},
                    steps={"boltz": {"enabled": True, "use_msa_server": False}})
    p = tmp_path / "run.yml"
    cfg.write(p)
    loaded = load_config(p)
    assert loaded.step_options("boltz")["use_msa_server"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement config**

```python
# structurezyme/config.py
import getpass
from datetime import datetime
from pathlib import Path
import yaml
from pydantic import BaseModel, ConfigDict, Field

class StepConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = True

class StepsConfig(BaseModel):
    squidly: StepConfig = StepConfig()
    chai: StepConfig = StepConfig()
    boltz: StepConfig = StepConfig()
    vina: StepConfig = StepConfig(enabled=False)
    docking_metrics: StepConfig = StepConfig()
    prepare_files: StepConfig = StepConfig()
    fastrelax: StepConfig = StepConfig(enabled=False)
    superimpose: StepConfig = StepConfig()
    protein_rmsd: StepConfig = StepConfig()
    ligand_rmsd: StepConfig = StepConfig()
    geometric_filter: StepConfig = StepConfig()
    fpocket: StepConfig = StepConfig()
    ligand_sasa: StepConfig = StepConfig()
    plip: StepConfig = StepConfig()
    placer: StepConfig = StepConfig(enabled=False)

class PathsConfig(BaseModel):
    output_root: str
    boltz_cache_dir: str
    squidly_weights_dir: str | None = None
    placer_env_path: str = "/mnt/labs/data/mora/software/PLACER/env"

class RuntimeConfig(BaseModel):
    user: str = Field(default_factory=getpass.getuser)
    run_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S"))
    num_threads: int = 1
    force: list[str] = Field(default_factory=list)

class RunConfig(BaseModel):
    paths: PathsConfig
    runtime: RuntimeConfig = RuntimeConfig()
    steps: StepsConfig = StepsConfig()

    def _step(self, name: str) -> StepConfig:
        return getattr(self.steps, name)

    def is_enabled(self, name: str) -> bool:
        return self._step(name).enabled

    def step_options(self, name: str) -> dict:
        return self._step(name).model_dump()

    def write(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.model_dump(), fh, sort_keys=False)

def load_config(path) -> RunConfig:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return RunConfig(**data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add structurezyme/config.py tests/test_config.py
git commit -m "feat: add Pydantic RunConfig with YAML load/write"
```

### Task B4: Manifest read/write and step records

**Files:**
- Create: `structurezyme/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `StepRecord` (Pydantic): `status: str` (one of `PENDING/RUNNING/OK/FAILED/SKIPPED_DISABLED/SKIPPED_CHECKPOINT`), `pkl_path: str | None`, `input_hash: str | None`, `config_hash: str | None`, `wall_time_s: float | None`, `mem_mb: float | None`, `started_at: str | None`, `finished_at: str | None`, `error: str | None`.
  - `Manifest`: `steps: dict[str, StepRecord]`; classmethod `load(path) -> Manifest` (returns empty if missing); method `save(path) -> None` (atomic write via temp file + replace); method `get(name) -> StepRecord | None`; method `set(name, record) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest.py
from structurezyme.manifest import Manifest, StepRecord

def test_load_missing_returns_empty(tmp_path):
    m = Manifest.load(tmp_path / "manifest.json")
    assert m.steps == {}

def test_set_save_load_roundtrip(tmp_path):
    p = tmp_path / "manifest.json"
    m = Manifest.load(p)
    m.set("chai", StepRecord(status="OK", input_hash="abc", config_hash="def"))
    m.save(p)
    m2 = Manifest.load(p)
    rec = m2.get("chai")
    assert rec.status == "OK"
    assert rec.input_hash == "abc"
    assert rec.config_hash == "def"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_manifest.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement manifest**

```python
# structurezyme/manifest.py
import json
import os
from pathlib import Path
from pydantic import BaseModel, Field

class StepRecord(BaseModel):
    status: str = "PENDING"
    pkl_path: str | None = None
    input_hash: str | None = None
    config_hash: str | None = None
    wall_time_s: float | None = None
    mem_mb: float | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None

class Manifest(BaseModel):
    steps: dict[str, StepRecord] = Field(default_factory=dict)

    @classmethod
    def load(cls, path) -> "Manifest":
        path = Path(path)
        if not path.is_file():
            return cls()
        with open(path) as fh:
            return cls(**json.load(fh))

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(self.model_dump(), fh, indent=2)
        os.replace(tmp, path)

    def get(self, name: str) -> StepRecord | None:
        return self.steps.get(name)

    def set(self, name: str, record: StepRecord) -> None:
        self.steps[name] = record
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_manifest.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add structurezyme/manifest.py tests/test_manifest.py
git commit -m "feat: add manifest with atomic save and step records"
```

### Task B5: Step registry (StepSpec + STEPS graph)

**Files:**
- Create: `structurezyme/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: nothing at import time (runner callables are wired in B7; here they may be `None` placeholders replaced by real functions, OR imported lazily inside `runner`). For this task, `StepSpec.runner` is typed as `Callable | None`.
- Produces:
  - `StepSpec` dataclass: `name: str`, `inputs: list[str]` (names of upstream steps whose checkpoints feed this one), `output: str` (checkpoint name, equals `name`), `depends_on: list[str]`, `runner: Callable | None`.
  - `STEPS: dict[str, StepSpec]` in dependency order.
  - `ordered_steps() -> list[str]` — topologically sorted names (raises on cycle).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
from structurezyme.registry import STEPS, ordered_steps

def test_all_steps_present():
    expected = {"squidly","chai","boltz","vina","docking_metrics","prepare_files",
                "fastrelax","superimpose","protein_rmsd","ligand_rmsd",
                "geometric_filter","fpocket","ligand_sasa","plip","placer"}
    assert set(STEPS) == expected

def test_topological_order_respects_dependencies():
    order = ordered_steps()
    idx = {name: i for i, name in enumerate(order)}
    for name, spec in STEPS.items():
        for dep in spec.depends_on:
            assert idx[dep] < idx[name], f"{dep} must precede {name}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement registry**

```python
# structurezyme/registry.py
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class StepSpec:
    name: str
    depends_on: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    runner: Callable | None = None

    @property
    def output(self) -> str:
        return self.name

def _spec(name, depends_on=None, inputs=None):
    return StepSpec(name=name, depends_on=depends_on or [], inputs=inputs or [])

STEPS: dict[str, StepSpec] = {
    "squidly":          _spec("squidly"),
    "chai":             _spec("chai", ["squidly"], ["squidly"]),
    "boltz":            _spec("boltz", ["chai"], ["chai"]),
    "vina":             _spec("vina", ["boltz"], ["boltz"]),
    "docking_metrics":  _spec("docking_metrics", ["boltz"], ["boltz", "vina"]),
    "prepare_files":    _spec("prepare_files", ["docking_metrics"], ["docking_metrics"]),
    "fastrelax":        _spec("fastrelax", ["prepare_files"], ["prepare_files"]),
    "superimpose":      _spec("superimpose", ["prepare_files"], ["prepare_files"]),
    "protein_rmsd":     _spec("protein_rmsd", ["superimpose"], ["superimpose"]),
    "ligand_rmsd":      _spec("ligand_rmsd", ["protein_rmsd"], ["protein_rmsd"]),
    "geometric_filter": _spec("geometric_filter", ["ligand_rmsd"], ["ligand_rmsd"]),
    "fpocket":          _spec("fpocket", ["geometric_filter"], ["geometric_filter"]),
    "ligand_sasa":      _spec("ligand_sasa", ["fpocket"], ["fpocket"]),
    "plip":             _spec("plip", ["ligand_sasa"], ["ligand_sasa"]),
    "placer":           _spec("placer", ["plip"], ["plip"]),
}

def ordered_steps() -> list[str]:
    visited, temp, order = set(), set(), []
    def visit(n):
        if n in visited:
            return
        if n in temp:
            raise ValueError(f"Cycle detected at {n}")
        temp.add(n)
        for dep in STEPS[n].depends_on:
            visit(dep)
        temp.discard(n)
        visited.add(n)
        order.append(n)
    for name in STEPS:
        visit(name)
    return order
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_registry.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add structurezyme/registry.py tests/test_registry.py
git commit -m "feat: add step registry with dependency graph and topo sort"
```

### Task B6: Runner with checkpoint discovery and resume

**Files:**
- Create: `structurezyme/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `RunConfig` (B3), `RunLayout` (B2), `Manifest`/`StepRecord` (B4), `STEPS`/`ordered_steps` (B5), `hash_obj`/`hash_files` (B1).
- Produces:
  - `RunContext` dataclass: `config: RunConfig`, `layout: RunLayout`, `manifest: Manifest`, `logger: logging.Logger`; convenience `checkpoint_path(name)`, `step_dir(name)`, `input_frames(spec) -> list[pd.DataFrame]`.
  - `Runner(config: RunConfig)`: method `run() -> None` iterating `ordered_steps()`; helper `_should_skip(name, spec) -> tuple[bool, str]` returning `(skip, reason)`.
  - Step runner contract: `runner(ctx: RunContext, spec: StepSpec) -> pd.DataFrame` — the produced frame is pickled to `ctx.checkpoint_path(spec.output)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner.py
import pandas as pd
from structurezyme.config import RunConfig
from structurezyme.runner import Runner
from structurezyme import registry

def _cfg(tmp_path):
    return RunConfig(
        paths={"output_root": str(tmp_path), "boltz_cache_dir": str(tmp_path / "cache")},
        runtime={"user": "tester", "run_id": "r1"},
    )

def test_disabled_step_is_skipped_and_recorded(tmp_path, monkeypatch):
    calls = []
    def fake_runner(ctx, spec):
        calls.append(spec.name)
        return pd.DataFrame({"Entry": [spec.name]})
    for spec in registry.STEPS.values():
        monkeypatch.setattr(spec, "runner", fake_runner, raising=False)
    cfg = _cfg(tmp_path)
    cfg.steps.vina.enabled = False
    Runner(cfg).run()
    assert "vina" not in calls
    m = Runner(cfg).manifest
    assert m.get("vina").status == "SKIPPED_DISABLED"

def test_checkpoint_hit_skips_second_run(tmp_path, monkeypatch):
    calls = []
    def fake_runner(ctx, spec):
        calls.append(spec.name)
        return pd.DataFrame({"Entry": [spec.name]})
    for spec in registry.STEPS.values():
        monkeypatch.setattr(spec, "runner", fake_runner, raising=False)
    cfg = _cfg(tmp_path)
    Runner(cfg).run()
    first = len(calls)
    calls.clear()
    Runner(cfg).run()  # second run: everything cached
    assert calls == [], "no step should re-run on a clean cache hit"
    assert first > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runner.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement runner**

```python
# structurezyme/runner.py
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pandas as pd

from .config import RunConfig
from .paths import run_dir, RunLayout
from .manifest import Manifest, StepRecord
from .registry import STEPS, StepSpec, ordered_steps
from .hashing import hash_obj, hash_files

@dataclass
class RunContext:
    config: RunConfig
    layout: RunLayout
    manifest: Manifest
    logger: logging.Logger

    def checkpoint_path(self, name: str) -> Path:
        return self.layout.checkpoint_path(name)

    def step_dir(self, name: str) -> Path:
        d = self.layout.step_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def input_frames(self, spec: StepSpec) -> list[pd.DataFrame]:
        frames = []
        for dep in spec.inputs:
            p = self.checkpoint_path(dep)
            if p.is_file():
                frames.append(pd.read_pickle(p))
        return frames

def _make_logger(layout: RunLayout) -> logging.Logger:
    logger = logging.getLogger(f"structurezyme.run.{layout.root.name}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(layout.log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(fh)
        logger.addHandler(logging.StreamHandler())
    return logger

class Runner:
    def __init__(self, config: RunConfig):
        self.config = config
        self.layout = RunLayout(run_dir(config.paths.output_root,
                                        config.runtime.user,
                                        config.runtime.run_id))
        self.layout.create()
        self.manifest = Manifest.load(self.layout.manifest_path)
        self.logger = _make_logger(self.layout)
        self.config.write(self.layout.config_path)

    def _existing_input_paths(self, spec: StepSpec):
        return [self.layout.checkpoint_path(d) for d in spec.inputs
                if self.layout.checkpoint_path(d).is_file()]

    def _should_skip(self, name: str, spec: StepSpec):
        if not self.config.is_enabled(name):
            return True, "SKIPPED_DISABLED"
        if name in self.config.runtime.force:
            return False, ""
        rec = self.manifest.get(name)
        out = self.layout.checkpoint_path(spec.output)
        if rec is None or not out.is_file():
            return False, ""
        cfg_hash = hash_obj(self.config.step_options(name))
        in_hash = hash_files(self._existing_input_paths(spec))
        if rec.status == "OK" and rec.config_hash == cfg_hash and rec.input_hash == in_hash:
            return True, "SKIPPED_CHECKPOINT"
        return False, ""

    def run(self) -> None:
        for name in ordered_steps():
            spec = STEPS[name]
            skip, reason = self._should_skip(name, spec)
            if skip:
                self.logger.info(f"{reason}: {name}")
                self.manifest.set(name, StepRecord(
                    status=reason,
                    pkl_path=str(self.layout.checkpoint_path(spec.output)),
                    config_hash=hash_obj(self.config.step_options(name)),
                    input_hash=hash_files(self._existing_input_paths(spec)),
                ))
                self.manifest.save(self.layout.manifest_path)
                continue

            self.logger.info(f"RUN: {name}")
            started = datetime.now().isoformat()
            t0 = time.time()
            self.manifest.set(name, StepRecord(status="RUNNING", started_at=started))
            self.manifest.save(self.layout.manifest_path)
            ctx = RunContext(self.config, self.layout, self.manifest, self.logger)
            try:
                df = spec.runner(ctx, spec)
                out = self.layout.checkpoint_path(spec.output)
                df.to_pickle(out)
                self.manifest.set(name, StepRecord(
                    status="OK",
                    pkl_path=str(out),
                    config_hash=hash_obj(self.config.step_options(name)),
                    input_hash=hash_files(self._existing_input_paths(spec)),
                    wall_time_s=time.time() - t0,
                    started_at=started,
                    finished_at=datetime.now().isoformat(),
                ))
                self.manifest.save(self.layout.manifest_path)
            except Exception as exc:
                self.manifest.set(name, StepRecord(
                    status="FAILED", started_at=started,
                    finished_at=datetime.now().isoformat(), error=repr(exc)))
                self.manifest.save(self.layout.manifest_path)
                self.logger.error(f"FAILED: {name}: {exc!r}")
                raise
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_runner.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add structurezyme/runner.py tests/test_runner.py
git commit -m "feat: add Runner with checkpoint discovery, hashing, and resume"
```

### Task B7: Wire existing phase logic into step runner callables

**Files:**
- Create: `structurezyme/step_runners.py`
- Modify: `structurezyme/registry.py` (attach runners)
- Modify: `structurezyme/pipeline.py` (delegate to runners; keep phase classes as thin wrappers)
- Test: `tests/test_step_runners_smoke.py`

**Interfaces:**
- Consumes: `RunContext` (B6); the existing step classes in `structurezyme/steps/` and the phase logic currently inside `Docking`, `Superimposition`, `GeometricFilters` in `pipeline.py`.
- Produces: one module-level function per registry step, each `def run_<step>(ctx, spec) -> pd.DataFrame`, wired into `STEPS[name].runner`. Each function:
  1. loads its input frame(s) via `ctx.input_frames(spec)` (falls back to the seed input DataFrame for `squidly`/first step, read from `ctx.checkpoint_path("_input")`),
  2. runs the same operations the phase classes ran (Chai, Boltz, Vina, DockingMetrics, PrepareChai/Boltz/Vina, FastRelax, SuperimposeStructures, ProteinRMSD, LigandRMSD, GeneralGeometricFiltering/EsteraseGeometricFiltering, Fpocket, LigandSASA, PLIP, PLACER), writing to the preserved sub-directories under `ctx.step_dir(...)`,
  3. returns the resulting DataFrame.

- [ ] **Step 1: Write the failing test (structural, no heavy deps)**

```python
# tests/test_step_runners_smoke.py
from structurezyme import registry

def test_every_step_has_a_runner_attached():
    for name, spec in registry.STEPS.items():
        assert callable(spec.runner), f"{name} has no runner wired"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_step_runners_smoke.py -v`
Expected: FAIL (runners are `None`).

- [ ] **Step 3: Implement runners and wire them**

Create `structurezyme/step_runners.py` with one function per step. Port the body of each existing private method from `pipeline.py` (`_catalytic_residue_prediction`->`run_squidly`, `_run_chai`->`run_chai`, `_run_boltz`->`run_boltz`, `_run_vina`->`run_vina`, `_extract_docking_quality_metrics`->`run_docking_metrics`, `_prepare_files_for_superimposition`->`run_prepare_files`, `_run_fastrelax`->`run_fastrelax`, `_superimposition`->`run_superimpose`, `_proteinRMSD`->`run_protein_rmsd`, `_ligandRMSD`->`run_ligand_rmsd`, `_run_geometric_filtering`->`run_geometric_filter`, `_active_site_volume`->`run_fpocket`, `_ligand_surface_exposure`->`run_ligand_sasa`, `_plip_interactions`->`run_plip`, PLACER block->`run_placer`). Each reads inputs via `ctx`, writes outputs under `ctx.step_dir(...)`, and returns the DataFrame. At the bottom of `registry.py` add:

```python
from . import step_runners as _sr  # noqa: E402
_RUNNERS = {
    "squidly": _sr.run_squidly, "chai": _sr.run_chai, "boltz": _sr.run_boltz,
    "vina": _sr.run_vina, "docking_metrics": _sr.run_docking_metrics,
    "prepare_files": _sr.run_prepare_files, "fastrelax": _sr.run_fastrelax,
    "superimpose": _sr.run_superimpose, "protein_rmsd": _sr.run_protein_rmsd,
    "ligand_rmsd": _sr.run_ligand_rmsd, "geometric_filter": _sr.run_geometric_filter,
    "fpocket": _sr.run_fpocket, "ligand_sasa": _sr.run_ligand_sasa,
    "plip": _sr.run_plip, "placer": _sr.run_placer,
}
for _name, _fn in _RUNNERS.items():
    STEPS[_name].runner = _fn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_step_runners_smoke.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add structurezyme/step_runners.py structurezyme/registry.py structurezyme/pipeline.py tests/test_step_runners_smoke.py
git commit -m "feat: wire existing phase logic into modular step runners"
```

### Task B8: Backward-compatible Pipeline adapter

**Files:**
- Modify: `structurezyme/pipeline.py`
- Test: `tests/test_pipeline_adapter.py`

**Interfaces:**
- Consumes: `RunConfig` (B3), `Runner` (B6).
- Produces: the existing `Pipeline(df, boltz_cache_dir=..., ...)` constructor still works; internally builds a `RunConfig`, seeds the input DataFrame to `checkpoints/_input.pkl`, and calls `Runner(cfg).run()`. Emits a `DeprecationWarning` steering users to `RunConfig` + `Runner`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_adapter.py
import warnings
import pandas as pd
from structurezyme.pipeline import Pipeline

def test_pipeline_kwargs_build_runconfig(tmp_path):
    df = pd.DataFrame({"Entry": ["e1"], "Sequence": ["MKT"], "substrate_smiles": ["CC"],
                       "substrate_name": ["x"], "substrate_moiety": ["[C]"]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = Pipeline(df=df, boltz_cache_dir=str(tmp_path / "cache"),
                     base_output_dir=str(tmp_path / "out"), run_vina=False)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert p.config.is_enabled("vina") is False
    assert p.config.is_enabled("chai") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_adapter.py -v`
Expected: FAIL (no `.config` attribute / no warning).

- [ ] **Step 3: Implement the adapter**

In `structurezyme/pipeline.py`, add a `Pipeline.__init__` that maps legacy kwargs onto a `RunConfig`:
- `base_output_dir` -> `paths.output_root`
- `boltz_cache_dir` -> `paths.boltz_cache_dir`
- `run_vina` -> `steps.vina.enabled`
- `skip_catalytic_residue_prediction` -> `steps.squidly.enabled = not skip`
- `run_fastrelax` -> `steps.fastrelax.enabled`; `run_placer` -> `steps.placer.enabled`
- `esterase`, `max_matches`, `num_threads`, `metagenomic_enzymes`, `alternative_structure_for_vina`, `use_msa_server`, and the `squidly_*`/`fastrelax_*`/`placer_*` options -> the corresponding step options.
Store the resulting config on `self.config`, seed the DataFrame to `checkpoints/_input.pkl`, and implement `run()` to call `Runner(self.config).run()`. Emit the `DeprecationWarning` in `__init__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_adapter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add structurezyme/pipeline.py tests/test_pipeline_adapter.py
git commit -m "feat: make legacy Pipeline a thin RunConfig+Runner adapter"
```

---

## Phase C: Multi-User, CLI, and Slurm

### Task C1: Host profiles for shared caches

**Files:**
- Create: `structurezyme/hosts.yml`
- Create: `structurezyme/hosts.py`
- Test: `tests/test_hosts.py`

**Interfaces:**
- Produces:
  - `load_host_profile(name: str | None = None) -> dict` — resolves a host profile from `hosts.yml`; `name` defaults to `$STRUCTUREZYME_HOST` then `"default"`.
  - `apply_host_defaults(cfg: RunConfig, name: str | None = None) -> RunConfig` — fills unset `paths.boltz_cache_dir`, `paths.squidly_weights_dir`, `paths.placer_env_path`, `paths.output_root` from the profile without overriding explicit values.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hosts.py
from structurezyme.config import RunConfig
from structurezyme.hosts import load_host_profile, apply_host_defaults

def test_default_profile_has_keys():
    prof = load_host_profile("default")
    assert "boltz_cache_dir" in prof

def test_apply_does_not_override_explicit(tmp_path):
    cfg = RunConfig(paths={"output_root": "/explicit", "boltz_cache_dir": "/explicit/cache"})
    out = apply_host_defaults(cfg, "default")
    assert out.paths.output_root == "/explicit"
    assert out.paths.boltz_cache_dir == "/explicit/cache"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hosts.py -v`
Expected: FAIL (modules missing).

- [ ] **Step 3: Implement hosts**

```yaml
# structurezyme/hosts.yml
default:
  output_root: /mnt/labs/data/mora/structurezyme_runs
  boltz_cache_dir: /mnt/labs/data/mora/boltz_cache
  squidly_weights_dir: /mnt/labs/data/mora/squidly_weights
  placer_env_path: /mnt/labs/data/mora/software/PLACER/env
```

```python
# structurezyme/hosts.py
import os
from importlib import resources
import yaml
from .config import RunConfig

def _all_profiles() -> dict:
    with resources.files("structurezyme").joinpath("hosts.yml").open() as fh:
        return yaml.safe_load(fh)

def load_host_profile(name: str | None = None) -> dict:
    name = name or os.environ.get("STRUCTUREZYME_HOST", "default")
    profiles = _all_profiles()
    if name not in profiles:
        raise KeyError(f"Unknown host profile {name!r}; have {list(profiles)}")
    return profiles[name]

_SENTINEL_PLACER = "/mnt/labs/data/mora/software/PLACER/env"

def apply_host_defaults(cfg: RunConfig, name: str | None = None) -> RunConfig:
    prof = load_host_profile(name)
    p = cfg.paths
    if not p.output_root:
        p.output_root = prof.get("output_root", p.output_root)
    if not p.boltz_cache_dir:
        p.boltz_cache_dir = prof.get("boltz_cache_dir", p.boltz_cache_dir)
    if p.squidly_weights_dir is None:
        p.squidly_weights_dir = prof.get("squidly_weights_dir")
    if p.placer_env_path == _SENTINEL_PLACER and "placer_env_path" in prof:
        p.placer_env_path = prof["placer_env_path"]
    return cfg
```

Note: make `output_root` and `boltz_cache_dir` optional (default `""`) in `PathsConfig` so host defaults can fill them; add a `RunConfig` validator that errors if they remain empty after host application.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hosts.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add structurezyme/hosts.yml structurezyme/hosts.py tests/test_hosts.py
git commit -m "feat: add host profiles for shared cache/weight paths"
```

### Task C2: File-locked shared cache access

**Files:**
- Create: `structurezyme/locks.py`
- Test: `tests/test_locks.py`

**Interfaces:**
- Produces: `shared_lock(path) -> ContextManager` — a `filelock.FileLock` on `<path>.lock`, used to guard concurrent writes to the Boltz cache and Squidly weights.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_locks.py
from structurezyme.locks import shared_lock

def test_shared_lock_acquires_and_releases(tmp_path):
    target = tmp_path / "cache"
    with shared_lock(target) as lock:
        assert lock.is_locked
    assert not lock.is_locked
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_locks.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement locks**

```python
# structurezyme/locks.py
from contextlib import contextmanager
from pathlib import Path
from filelock import FileLock

@contextmanager
def shared_lock(path, timeout: float = -1):
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_path), timeout=timeout)
    with lock:
        yield lock
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_locks.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add structurezyme/locks.py tests/test_locks.py
git commit -m "feat: add filelock-based shared cache guard"
```

### Task C3: CLI (run, resume, status, step, init)

**Files:**
- Create: `structurezyme/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `load_config` (B3), `Runner` (B6), `apply_host_defaults` (C1), `Manifest` (B4), `RunLayout`/`run_dir` (B2).
- Produces: `main(argv: list[str] | None = None) -> int` with subcommands:
  - `init --output PATH` — writes a template `run.yml`.
  - `run --config PATH [--host NAME] [--force STEP[,STEP...]]` — full run.
  - `resume --run-dir PATH` — reloads `config.yml` from the run dir and re-runs (checkpoints skip completed steps).
  - `step NAME --run-dir PATH` — force a single step (adds it to `runtime.force`, runs only up to and including it).
  - `status --run-dir PATH` — prints a manifest table.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
from pathlib import Path
from structurezyme.cli import main

def test_init_writes_template(tmp_path):
    out = tmp_path / "run.yml"
    rc = main(["init", "--output", str(out)])
    assert rc == 0
    assert out.is_file()
    text = out.read_text()
    assert "output_root" in text and "steps" in text

def test_status_on_missing_run_dir_returns_nonzero(tmp_path):
    rc = main(["status", "--run-dir", str(tmp_path / "nope")])
    assert rc != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement the CLI**

```python
# structurezyme/cli.py
import argparse
import sys
from pathlib import Path
from .config import RunConfig, PathsConfig, load_config
from .hosts import apply_host_defaults
from .manifest import Manifest
from .runner import Runner

def _template() -> RunConfig:
    return RunConfig(paths=PathsConfig(output_root="", boltz_cache_dir=""))

def cmd_init(args) -> int:
    _template().write(args.output)
    print(f"Wrote template config to {args.output}")
    return 0

def cmd_run(args) -> int:
    cfg = load_config(args.config)
    cfg = apply_host_defaults(cfg, args.host)
    if args.force:
        cfg.runtime.force = args.force.split(",")
    Runner(cfg).run()
    return 0

def cmd_resume(args) -> int:
    cfg = load_config(Path(args.run_dir) / "config.yml")
    Runner(cfg).run()
    return 0

def cmd_step(args) -> int:
    cfg = load_config(Path(args.run_dir) / "config.yml")
    cfg.runtime.force = list(set(cfg.runtime.force) | {args.name})
    Runner(cfg).run()
    return 0

def cmd_status(args) -> int:
    manifest_path = Path(args.run_dir) / "manifest.json"
    if not manifest_path.is_file():
        print(f"No manifest at {manifest_path}", file=sys.stderr)
        return 1
    m = Manifest.load(manifest_path)
    for name, rec in m.steps.items():
        wt = f"{rec.wall_time_s:.1f}s" if rec.wall_time_s else "-"
        print(f"{name:20s} {rec.status:20s} {wt}")
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="structurezyme")
    sub = p.add_subparsers(dest="command", required=True)
    pi = sub.add_parser("init"); pi.add_argument("--output", required=True); pi.set_defaults(func=cmd_init)
    pr = sub.add_parser("run"); pr.add_argument("--config", required=True)
    pr.add_argument("--host", default=None); pr.add_argument("--force", default=None); pr.set_defaults(func=cmd_run)
    ps = sub.add_parser("resume"); ps.add_argument("--run-dir", required=True); ps.set_defaults(func=cmd_resume)
    pst = sub.add_parser("step"); pst.add_argument("name"); pst.add_argument("--run-dir", required=True); pst.set_defaults(func=cmd_step)
    pstat = sub.add_parser("status"); pstat.add_argument("--run-dir", required=True); pstat.set_defaults(func=cmd_status)
    return p

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add structurezyme/cli.py tests/test_cli.py
git commit -m "feat: add structurezyme CLI (init/run/resume/step/status)"
```

### Task C4: Slurm submit command and template

**Files:**
- Create: `structurezyme/templates/slurm.sbatch.j2`
- Modify: `structurezyme/cli.py` (add `submit`)
- Test: `tests/test_submit.py`

**Interfaces:**
- Consumes: `load_config` (B3).
- Produces: `render_sbatch(config_path, host, job_name, partition, gpus, time_limit) -> str` and a `submit` subcommand with `--config`, `--host`, `--partition`, `--gpus`, `--time`, `--dry-run` (prints script instead of calling `sbatch`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_submit.py
from structurezyme.cli import render_sbatch

def test_render_sbatch_contains_run_command(tmp_path):
    cfg = tmp_path / "run.yml"; cfg.write_text("paths:\n  output_root: /o\n  boltz_cache_dir: /c\n")
    script = render_sbatch(str(cfg), host="default", job_name="sz", partition="gpu", gpus=1, time_limit="24:00:00")
    assert "structurezyme run --config" in script
    assert "--partition=gpu" in script
    assert "--gres=gpu:1" in script
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_submit.py -v`
Expected: FAIL (`render_sbatch` missing).

- [ ] **Step 3: Implement template and submit**

```
# structurezyme/templates/slurm.sbatch.j2
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --time={time_limit}
#SBATCH --output=%x-%j.out

set -euo pipefail
structurezyme run --config {config_path} --host {host}
```

Add to `structurezyme/cli.py`:

```python
from importlib import resources

def render_sbatch(config_path, host, job_name, partition, gpus, time_limit) -> str:
    tmpl = resources.files("structurezyme.templates").joinpath("slurm.sbatch.j2").read_text()
    return tmpl.format(job_name=job_name, partition=partition, gpus=gpus,
                       time_limit=time_limit, config_path=config_path, host=host)

def cmd_submit(args) -> int:
    script = render_sbatch(args.config, args.host or "default", args.job_name,
                           args.partition, args.gpus, args.time)
    if args.dry_run:
        print(script)
        return 0
    import subprocess, tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".sbatch", delete=False) as fh:
        fh.write(script); path = fh.name
    try:
        subprocess.run(["sbatch", path], check=True)
    finally:
        os.unlink(path)
    return 0
```

Register in `build_parser()`:

```python
    psub = sub.add_parser("submit")
    psub.add_argument("--config", required=True)
    psub.add_argument("--host", default=None)
    psub.add_argument("--job-name", default="structurezyme")
    psub.add_argument("--partition", default="gpu")
    psub.add_argument("--gpus", type=int, default=1)
    psub.add_argument("--time", default="24:00:00")
    psub.add_argument("--dry-run", action="store_true")
    psub.set_defaults(func=cmd_submit)
```

Also ensure `setup.py` ships templates via `package_data={"structurezyme": ["hosts.yml", "templates/*.j2"]}` and `include_package_data=True`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_submit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add structurezyme/templates/slurm.sbatch.j2 structurezyme/cli.py setup.py tests/test_submit.py
git commit -m "feat: add Slurm submit command and job template"
```

---

## Phase E: Documentation

### Task E1: Rewrite README under StructureZyme name

**Files:**
- Modify: `README.md`

**Interfaces:**
- Produces: a StructureZyme-branded README whose install + quickstart + CLI examples work verbatim.

- [ ] **Step 1: Replace the title and intro**

Change the H1 to `# StructureZyme` and update the description to name StructureZyme.

- [ ] **Step 2: Fix install commands**

Update to:

```bash
conda env create -f environment.yml
conda activate structurezyme
python setup.py sdist bdist_wheel
pip install dist/structurezyme-0.1.0.tar.gz --use-deprecated=legacy-resolver
pip install enzymetk==0.0.8
```

- [ ] **Step 3: Add a CLI-first quickstart section**

```bash
# 1. Generate a config template
structurezyme init --output run.yml
# 2. Edit run.yml (set output_root, boltz_cache_dir, enable/disable steps)
# 3. Run
structurezyme run --config run.yml
# 4. Inspect progress / resume after a crash
structurezyme status --run-dir /path/to/output_root/<user>/<run_id>
structurezyme resume --run-dir /path/to/output_root/<user>/<run_id>
```

- [ ] **Step 4: Update all GitHub URLs and doc links**

Run: `grep -n "filterzyme\|Filterzyme\|HelenSchmid" README.md`
Expected after edits: no matches (except an optional migration note about the deprecated `filterzyme` import).

- [ ] **Step 5: Verify links resolve**

Run: `grep -oE "docs/[a-zA-Z_./]+\.md" README.md | sort -u | while read f; do test -f "$f" && echo "OK $f" || echo "MISSING $f"; done`
Expected: every referenced doc prints `OK`.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for StructureZyme with CLI-first quickstart"
```

### Task E2: Rewrite pipeline overview around the modular step graph

**Files:**
- Modify: `docs/pipeline_overview.md`

**Interfaces:**
- Produces: overview describing the step registry, checkpoints, resume, and the FastRelax + PLACER steps.

- [ ] **Step 1: Replace the architecture diagram**

Add a step-graph diagram covering all 15 steps (squidly, chai, boltz, vina, docking_metrics, prepare_files, fastrelax, superimpose, protein_rmsd, ligand_rmsd, geometric_filter, fpocket, ligand_sasa, plip, placer) with their dependencies.

- [ ] **Step 2: Document checkpoints and resume**

Add a section explaining that each step writes `checkpoints/<step>.pkl`, that `manifest.json` records `status`/`input_hash`/`config_hash`, and that a step is skipped when enabled + checkpoint present + hashes match.

- [ ] **Step 3: Replace all "Filterzyme" mentions**

Run: `grep -n "filterzyme\|Filterzyme" docs/pipeline_overview.md`
Expected after edits: no matches.

- [ ] **Step 4: Commit**

```bash
git add docs/pipeline_overview.md
git commit -m "docs: rewrite pipeline overview around modular step graph"
```

### Task E3: Add configuration reference

**Files:**
- Create: `docs/configuration.md`

**Interfaces:**
- Produces: full `RunConfig` field reference and example YAMLs (minimal, with Vina, with cofactor, with FastRelax+PLACER).

- [ ] **Step 1: Document the config schema**

List every field of `PathsConfig`, `RuntimeConfig`, and each `StepsConfig` entry with its default and meaning (mirror `structurezyme/config.py`).

- [ ] **Step 2: Add a minimal example**

```yaml
paths:
  output_root: /mnt/labs/data/mora/structurezyme_runs
  boltz_cache_dir: /mnt/labs/data/mora/boltz_cache
runtime:
  num_threads: 4
steps:
  vina: {enabled: false}
```

- [ ] **Step 3: Add Vina + cofactor + FastRelax/PLACER examples**

Provide three more YAML blocks toggling `vina`, cofactor columns, and `fastrelax`/`placer` (with `placer.predict_ligand`).

- [ ] **Step 4: Commit**

```bash
git add docs/configuration.md
git commit -m "docs: add RunConfig configuration reference with examples"
```

### Task E4: Add resume/checkpoints and multi-user docs

**Files:**
- Create: `docs/resume_and_checkpoints.md`
- Create: `docs/multi_user.md`

**Interfaces:**
- Produces: operator docs for recovery and shared-host usage.

- [ ] **Step 1: Write resume_and_checkpoints.md**

Cover: manifest anatomy, the three skip statuses, `--force step,step`, rerunning a single step with `structurezyme step <name> --run-dir ...`, and corruption recovery (delete a checkpoint + resume).

- [ ] **Step 2: Write multi_user.md**

Cover: `{output_root}/{user}/{run_id}/` layout, `hosts.yml` profiles + `--host`/`$STRUCTUREZYME_HOST`, shared caches with filelock, and `structurezyme submit --slurm`.

- [ ] **Step 3: Commit**

```bash
git add docs/resume_and_checkpoints.md docs/multi_user.md
git commit -m "docs: add resume/checkpoint and multi-user guides"
```

### Task E5: Update API reference and getting-started

**Files:**
- Modify: `docs/api_reference.md`
- Modify: `docs/getting_started.md`

**Interfaces:**
- Produces: API docs for `config`, `registry`, `runner`, `cli`, `hosts`; getting-started referencing the env `structurezyme` and CLI.

- [ ] **Step 1: Document the new modules**

Add sections for `RunConfig`, `Runner`, `STEPS`/`StepSpec`, and the CLI subcommands.

- [ ] **Step 2: Purge remaining Filterzyme references across docs**

Run: `grep -rn "filterzyme\|Filterzyme\|filterpipeline2" docs/`
Expected after edits: no matches (except intentional migration notes).

- [ ] **Step 3: Commit**

```bash
git add docs/api_reference.md docs/getting_started.md
git commit -m "docs: update API reference and getting-started for StructureZyme"
```

---

## Phase F: Integration & Final Verification

### Task F1: End-to-end resumability integration test with fake steps

**Files:**
- Test: `tests/test_integration_resume.py`

**Interfaces:**
- Consumes: `RunConfig`, `Runner`, `registry.STEPS`, `Manifest`.
- Produces: proof that (a) a crash mid-run leaves a resumable state and (b) resume completes without re-running successful steps, and (c) `--force` re-runs the targeted step.

- [ ] **Step 1: Write the integration test**

```python
# tests/test_integration_resume.py
import pandas as pd
import pytest
from structurezyme.config import RunConfig
from structurezyme.runner import Runner
from structurezyme import registry

def _cfg(tmp_path):
    return RunConfig(
        paths={"output_root": str(tmp_path), "boltz_cache_dir": str(tmp_path / "c")},
        runtime={"user": "t", "run_id": "r"},
    )

def test_crash_then_resume(tmp_path, monkeypatch):
    order = registry.ordered_steps()
    crash_at = order[3]
    state = {"crash": True}

    def make(name):
        def _run(ctx, spec):
            if spec.name == crash_at and state["crash"]:
                raise RuntimeError("boom")
            return pd.DataFrame({"Entry": [spec.name]})
        return _run
    for name, spec in registry.STEPS.items():
        monkeypatch.setattr(spec, "runner", make(name), raising=False)

    with pytest.raises(RuntimeError):
        Runner(_cfg(tmp_path)).run()

    r = Runner(_cfg(tmp_path))
    assert r.manifest.get(crash_at).status == "FAILED"
    assert r.manifest.get(order[0]).status == "OK"

    state["crash"] = False
    ran = []
    def make2(name):
        def _run(ctx, spec):
            ran.append(spec.name)
            return pd.DataFrame({"Entry": [spec.name]})
        return _run
    for name, spec in registry.STEPS.items():
        monkeypatch.setattr(spec, "runner", make2(name), raising=False)
    Runner(_cfg(tmp_path)).run()
    assert order[0] not in ran           # cached, not re-run
    assert crash_at in ran               # resumed
    assert order[-1] in ran              # completed to the end

def test_force_reruns_target(tmp_path, monkeypatch):
    def make(name):
        def _run(ctx, spec):
            return pd.DataFrame({"Entry": [spec.name]})
        return _run
    for name, spec in registry.STEPS.items():
        monkeypatch.setattr(spec, "runner", make(name), raising=False)
    Runner(_cfg(tmp_path)).run()

    ran = []
    def make2(name):
        def _run(ctx, spec):
            ran.append(spec.name)
            return pd.DataFrame({"Entry": [spec.name]})
        return _run
    for name, spec in registry.STEPS.items():
        monkeypatch.setattr(spec, "runner", make2(name), raising=False)
    cfg = _cfg(tmp_path)
    cfg.runtime.force = ["chai"]
    Runner(cfg).run()
    assert "chai" in ran
```

- [ ] **Step 2: Run test to verify it fails (if any wiring is incomplete) or passes**

Run: `pytest tests/test_integration_resume.py -v`
Expected: PASS once Runner (B6) is correct. If it fails, fix Runner before proceeding.

- [ ] **Step 3: Run the full unit suite**

Run: `pytest tests/ -v`
Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration_resume.py
git commit -m "test: add end-to-end resume and force integration tests"
```

### Task F2: Full-suite green + real quickstart re-verification

**Files:** none (verification only)

- [ ] **Step 1: Confirm no Filterzyme leftovers anywhere**

Run: `grep -rIn "filterzyme\|Filterzyme\|filterpipeline2" --exclude-dir=.git . | grep -v "deprecat"`
Expected: no unexpected matches (only intentional deprecation shim/migration notes).

- [ ] **Step 2: Re-run the real quickstart (GPU permitting)**

Run: `structurezyme init --output /tmp/opencode/run.yml` then edit paths and `structurezyme run --config /tmp/opencode/run.yml`
Expected: run completes; `checkpoints/plip.pkl` and `geometricfiltering/structural_features_final.pkl` exist; `manifest.json` shows all enabled steps `OK`.

- [ ] **Step 3: Verify resume is a no-op on a completed run**

Run: `structurezyme resume --run-dir <run_dir>`
Expected: log shows `SKIPPED_CHECKPOINT` for every step; nothing re-runs.

- [ ] **Step 4: Final commit / open PR**

```bash
git push -u origin refactor/structurezyme-modular
gh pr create --base lab-sanity-run_LCH --title "StructureZyme unification + modular resumable pipeline" --body "See docs/superpowers/plans/2026-07-17-structurezyme-unification.md"
```

---

## Self-Review

**Spec coverage:**
- Uniform Filterzyme->StructureZyme rename — Phase A (A1-A5) + docs Phase E.
- Multiple users — Phase C (C1 hosts, C2 locks) + per-user layout (B2) + CLI (C3).
- Include/exclude modules — `StepsConfig.enabled` (B3), `_should_skip` (B6).
- Intermediate outputs saved; resume after crash; re-run a previously excluded step without repeating everything — checkpoints + manifest hashing (B1, B4, B6) + `--force`/`step` (C3) + integration test (F1).
- "What else is missing" items from the analysis (CLI, config file, manifest, logging location, packaging drift, repo hygiene, stale weights removed + provenance documented, tests) — Phase A3/A4, Phase D (D1-D3), Phase B, Phase C, Phase F.
- Squidly weight provenance clarified: weights ship with the `squidly` pip package and are fetched by its own `download_models_hf.py` (HF repo `WillRieger/Squidly`); StructureZyme adds no downloader — Task D2.

**Placeholder scan:** Task B7 intentionally ports existing method bodies rather than re-listing hundreds of lines; every other code step contains complete code. B7's mapping table names each source method and its target function explicitly.

**Type consistency:** `StepSpec.runner(ctx, spec) -> pd.DataFrame` is used consistently across B5, B6, B7, and the F1 tests. `RunConfig.is_enabled/step_options`, `Manifest.get/set/save/load`, `RunLayout.checkpoint_path/step_dir`, and `apply_host_defaults` signatures match across all consuming tasks.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-17-structurezyme-unification.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
