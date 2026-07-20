# CLI Input Seeding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the config-driven CLI independently runnable by declaring input data in the config and seeding `checkpoints/_input.pkl` at the run boundary, with early validation of the columns the enabled modules require.

**Architecture:** Add `PathsConfig.input_csv`; `validate_paths()` requires it. `Runner.run()` seeds `checkpoints/_input.pkl` once from `input_csv` (CSV or pickle), resume-safe (existing pickle never clobbered), then validates that enabled modules' required input columns are present before executing steps.

**Tech Stack:** Python 3.11, pydantic, pandas, pytest.

## Global Constraints

- Env: `source /mnt/nfs/vol8t/software/software/miniforge/25.9.1/etc/profile.d/conda.sh && conda activate /mnt/storage01/home/lherrmann/envs/filterzyme`
- Run tests with `-p no:cacheprovider`; grep the summary line to avoid heavy-dep import noise.
- Git identity: Luca Herrmann <luca.herrmann98@gmail.com>.
- Worktree: `/mnt/storage01/home/lherrmann/structurezyme-worktrees/refactor-structurezyme-modular`, branch `refactor/structurezyme-modular`.
- Do NOT modify the legacy `Pipeline` adapter's own `_input.pkl` seeding (pipeline.py); this plan adds a parallel seeding path in `Runner`.

## File Structure

- `structurezyme/config.py` — add `PathsConfig.input_csv`; extend `validate_paths()`.
- `structurezyme/runner.py` — add module→required-columns map, a `_load_input_frame` helper, seeding + validation in `Runner.run()`.
- `structurezyme/cli.py` — add `input_csv=""` to the `init` template.
- `tests/test_seed.py` — new test module for seeding + column validation.
- `docs/configuration.md` — input-column requirement tables + BYO-PDB future note.

---

### Task 1: Add `input_csv` to config and `validate_paths`

**Files:**
- Modify: `structurezyme/config.py` (PathsConfig ~line 29-33; validate_paths ~line 55-56)
- Test: `tests/test_seed.py` (create)

**Interfaces:**
- Consumes: `RunConfig`, `PathsConfig` (existing).
- Produces: `PathsConfig.input_csv: str` (default `""`); `RunConfig.validate_paths()` now also requires `input_csv` non-empty.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_seed.py
import pytest
from structurezyme.config import RunConfig, PathsConfig


def _cfg(**paths):
    base = dict(output_root="/tmp/o", boltz_cache_dir="/tmp/b", input_csv="/tmp/in.csv")
    base.update(paths)
    return RunConfig(paths=PathsConfig(**base))


def test_validate_paths_requires_input_csv():
    cfg = _cfg(input_csv="")
    with pytest.raises(ValueError) as e:
        cfg.validate_paths()
    assert "input_csv" in str(e.value)


def test_validate_paths_ok_with_input_csv():
    assert _cfg().validate_paths() is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_seed.py -q -p no:cacheprovider`
Expected: FAIL — `PathsConfig` has no `input_csv` (or validate_paths does not check it).

- [ ] **Step 3: Add the field and extend validation**

In `structurezyme/config.py`, add to `PathsConfig`:

```python
    input_csv: str = ""
```

In `validate_paths()`, change the required list:

```python
        missing = [n for n in ("output_root", "boltz_cache_dir", "input_csv")
                   if not getattr(self.paths, n)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_seed.py -q -p no:cacheprovider`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add structurezyme/config.py tests/test_seed.py
git commit -m "feat: add PathsConfig.input_csv required by validate_paths"
```

---

### Task 2: Input-frame loader + module→columns map

**Files:**
- Modify: `structurezyme/runner.py` (module scope, after imports ~line 13)
- Test: `tests/test_seed.py`

**Interfaces:**
- Consumes: `pandas` (already imported in runner.py).
- Produces (module-level in `structurezyme/runner.py`):
  - `REQUIRED_INPUT_COLUMNS: dict[str, list[str]]` — always-required under key `"_base"`, plus per-module keys.
  - `def _load_input_frame(path: str) -> pd.DataFrame` — reads `.pkl`/`.pickle` via `read_pickle`, else `read_csv`.
  - `def missing_input_columns(df, enabled: set[str]) -> dict[str, list[str]]` — maps each requirement group (that applies) to its missing columns.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_seed.py
import pandas as pd
from structurezyme.runner import (
    _load_input_frame, missing_input_columns, REQUIRED_INPUT_COLUMNS,
)


def test_load_input_frame_csv(tmp_path):
    p = tmp_path / "in.csv"
    pd.DataFrame({"Sequence": ["M"], "substrate_smiles": ["C"], "Entry": ["P1"]}).to_csv(p, index=False)
    df = _load_input_frame(str(p))
    assert list(df.columns) == ["Sequence", "substrate_smiles", "Entry"]


def test_load_input_frame_pickle(tmp_path):
    p = tmp_path / "in.pkl"
    pd.DataFrame({"Sequence": ["M"], "cofactor_moiety": [None]}).to_pickle(p)
    df = _load_input_frame(str(p))
    assert df["cofactor_moiety"].iloc[0] is None


def test_missing_columns_base_only():
    df = pd.DataFrame({"Sequence": ["M"], "substrate_smiles": ["C"], "Entry": ["P1"]})
    assert missing_input_columns(df, enabled=set()) == {}


def test_missing_columns_vina():
    df = pd.DataFrame({"Sequence": ["M"], "substrate_smiles": ["C"], "Entry": ["P1"]})
    miss = missing_input_columns(df, enabled={"vina"})
    assert miss.get("vina") == ["vina_residues"]


def test_missing_columns_geometric_filter():
    df = pd.DataFrame({"Sequence": ["M"], "substrate_smiles": ["C"], "Entry": ["P1"]})
    miss = missing_input_columns(df, enabled={"geometric_filter"})
    assert miss.get("geometric_filter") == ["substrate_moiety"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_seed.py -q -p no:cacheprovider`
Expected: FAIL — `_load_input_frame`/`missing_input_columns`/`REQUIRED_INPUT_COLUMNS` do not exist.

- [ ] **Step 3: Implement the loader and map**

In `structurezyme/runner.py`, after the imports (after line 13):

```python
REQUIRED_INPUT_COLUMNS: dict[str, list[str]] = {
    "_base": ["Sequence", "substrate_smiles", "Entry"],
    "vina": ["vina_residues"],
    "geometric_filter": ["substrate_moiety"],
}


def _load_input_frame(path: str) -> pd.DataFrame:
    """Load the seed input DataFrame from a CSV or pickle file."""
    if str(path).endswith((".pkl", ".pickle")):
        return pd.read_pickle(path)
    return pd.read_csv(path)


def missing_input_columns(df: pd.DataFrame, enabled: set[str]) -> dict[str, list[str]]:
    """Return {group: [missing cols]} for the base plus each enabled module."""
    cols = set(df.columns)
    groups = ["_base"] + [m for m in REQUIRED_INPUT_COLUMNS if m != "_base" and m in enabled]
    out: dict[str, list[str]] = {}
    for g in groups:
        miss = [c for c in REQUIRED_INPUT_COLUMNS[g] if c not in cols]
        if miss:
            out[g] = miss
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_seed.py -q -p no:cacheprovider`
Expected: PASS (7 passed total).

- [ ] **Step 5: Commit**

```bash
git add structurezyme/runner.py tests/test_seed.py
git commit -m "feat: add input-frame loader and module->required-columns map"
```

---

### Task 3: Seed `_input.pkl` in `Runner.run()` (resume-safe) + validate columns

**Files:**
- Modify: `structurezyme/runner.py` (`Runner.run`, ~line 82-93 start of method)
- Test: `tests/test_seed.py`

**Interfaces:**
- Consumes: `_load_input_frame`, `missing_input_columns` (Task 2); `self.layout.checkpoint_path`, `self.config` (existing).
- Produces: `Runner._seed_and_validate()` called at the top of `run()`. Writes `checkpoints/_input.pkl` from `config.paths.input_csv` only if absent; raises `ValueError` listing missing columns for enabled modules.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_seed.py
from structurezyme.config import RuntimeConfig, StepsConfig, StepConfig
from structurezyme.runner import Runner


def _runner_cfg(tmp_path, input_csv, **steps):
    return RunConfig(
        paths=PathsConfig(output_root=str(tmp_path / "out"),
                          boltz_cache_dir=str(tmp_path / "bcache"),
                          input_csv=str(input_csv)),
        runtime=RuntimeConfig(user="u", run_id="r1"),
        steps=StepsConfig(**steps),
    )


def _base_csv(tmp_path):
    p = tmp_path / "in.csv"
    pd.DataFrame({"Sequence": ["M"], "substrate_smiles": ["C"], "Entry": ["P1"]}).to_csv(p, index=False)
    return p


def test_seed_writes_input_pkl(tmp_path):
    r = Runner(_runner_cfg(tmp_path, _base_csv(tmp_path)))
    r._seed_and_validate()
    seed = r.layout.checkpoint_path("_input")
    assert seed.is_file()
    assert list(pd.read_pickle(seed)["Entry"]) == ["P1"]


def test_seed_does_not_clobber_existing(tmp_path):
    r = Runner(_runner_cfg(tmp_path, _base_csv(tmp_path)))
    seed = r.layout.checkpoint_path("_input")
    pd.DataFrame({"Sequence": ["X"], "substrate_smiles": ["N"], "Entry": ["KEEP"]}).to_pickle(seed)
    r._seed_and_validate()
    assert list(pd.read_pickle(seed)["Entry"]) == ["KEEP"]


def test_seed_raises_on_missing_enabled_column(tmp_path):
    cfg = _runner_cfg(tmp_path, _base_csv(tmp_path), vina=StepConfig(enabled=True))
    r = Runner(cfg)
    with pytest.raises(ValueError) as e:
        r._seed_and_validate()
    assert "vina" in str(e.value) and "vina_residues" in str(e.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_seed.py -q -p no:cacheprovider`
Expected: FAIL — `Runner` has no `_seed_and_validate`.

- [ ] **Step 3: Implement seeding + validation and call it from `run()`**

In `structurezyme/runner.py`, add this method to `Runner` (e.g. just before `run`):

```python
    def _seed_and_validate(self) -> None:
        """Seed checkpoints/_input.pkl from config.paths.input_csv (once) and
        validate that enabled modules' required input columns are present.

        Resume-safe: if the seed pickle already exists it is left untouched so
        downstream input hashes stay stable.
        """
        seed = self.layout.checkpoint_path("_input")
        if seed.is_file():
            df = pd.read_pickle(seed)
        else:
            df = _load_input_frame(self.config.paths.input_csv)
            df.to_pickle(seed)
        enabled = {n for n in STEPS if self.config.is_enabled(n)}
        miss = missing_input_columns(df, enabled)
        if miss:
            parts = [f"{g} needs {cols}" for g, cols in miss.items()]
            raise ValueError(
                "Input is missing required column(s): " + "; ".join(parts)
            )
```

Then, as the first line inside `run()` (before the `stop_after` check), call it:

```python
        self._seed_and_validate()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_seed.py -q -p no:cacheprovider`
Expected: PASS (10 passed total).

- [ ] **Step 5: Commit**

```bash
git add structurezyme/runner.py tests/test_seed.py
git commit -m "feat: seed _input.pkl and validate enabled-module columns in Runner.run"
```

---

### Task 4: Add `input_csv` to the CLI `init` template

**Files:**
- Modify: `structurezyme/cli.py` (`_template`, line 14-15)
- Test: `tests/test_seed.py`

**Interfaces:**
- Consumes: `cli._template` / `cmd_init` (existing).
- Produces: `init` template YAML now includes `paths.input_csv`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_seed.py
import yaml
from structurezyme.cli import _template


def test_init_template_has_input_csv():
    d = _template().model_dump()
    assert "input_csv" in d["paths"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_seed.py::test_init_template_has_input_csv -q -p no:cacheprovider`
Expected: PASS already IF the pydantic default serializes (it does, since `input_csv` is a model field). If it FAILS, the field was not added in Task 1 — fix Task 1 first.

Note: because `input_csv` is a `PathsConfig` field with default `""`, `model_dump()` includes it automatically, so no code change may be required here. This task exists to lock that behavior with a test. If the assertion passes with no code change, proceed to commit.

- [ ] **Step 3: (If needed) make the template explicit**

Only if the test fails, set it explicitly in `_template`:

```python
def _template() -> RunConfig:
    return RunConfig(paths=PathsConfig(output_root="", boltz_cache_dir="", input_csv=""))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_seed.py::test_init_template_has_input_csv -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add structurezyme/cli.py tests/test_seed.py
git commit -m "test: lock input_csv presence in init template"
```

---

### Task 5: Document input columns and BYO-PDB future note

**Files:**
- Modify: `docs/configuration.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Add the input-data section to `docs/configuration.md`**

Append a new section (adjust the heading level to match the file):

```markdown
## Input data (`paths.input_csv`)

The `run` command reads its input from the file at `paths.input_csv` and seeds
`checkpoints/_input.pkl` on the first run (a resume never overwrites an existing
seed). The file may be a `.csv` or a pandas `.pkl`/`.pickle`. Use a pickle when
you need `None`/list-typed columns (e.g. `cofactor_moiety`) preserved exactly.

The input is **tabular only** — it never contains structure files. Module
*behavior* settings (fastrelax mode, `placer_predict_ligand`, thresholds, …)
live in `run.yml`, not in the input file.

Always-required columns:

| Column             | Meaning                              |
|--------------------|--------------------------------------|
| `Sequence`         | Protein sequence                     |
| `substrate_smiles` | Substrate SMILES                     |
| `Entry`            | UniProt accession                    |

Required only when the module is enabled:

| Enabled module     | Extra required column |
|--------------------|-----------------------|
| `vina`             | `vina_residues`       |
| `geometric_filter` | `substrate_moiety`    |

Recommended (used opportunistically, may be empty/`None`): `substrate_name`,
`cofactor_smiles`, `cofactor_moiety`.

If an enabled module's required column is missing, `run` fails immediately with
a message naming the module and the missing column(s), before any GPU work.

### Bring-your-own-PDB single-module runs (future — not yet implemented)

Running a single downstream module (e.g. `fastrelax`, `placer`,
`geometric_filter`) directly on a PDB you already have — skipping Chai/Boltz
docking — is **not yet supported**. Those modules read structures from
upstream checkpoint columns (e.g. FastRelax needs the
`*_files_for_superimposition` columns), so a per-module "PDB → expected
columns" adapter is required. This is planned as a follow-up feature.
```

- [ ] **Step 2: Verify cross-links / build**

Run: `python -m pytest tests/ -q -p no:cacheprovider`
Expected: full suite still green (docs change is inert): summary shows `passed` with 0 failures.

- [ ] **Step 3: Commit**

```bash
git add docs/configuration.md
git commit -m "docs: document input columns and BYO-PDB future note"
```

---

### Task 6: Full-suite verification

**Files:** none.

- [ ] **Step 1: Run the whole suite**

Run: `python -m pytest tests/ -q -p no:cacheprovider`
Expected: previous count + new `tests/test_seed.py` tests, 0 failures.

- [ ] **Step 2: CLI smoke — init template includes input_csv**

Run: `python -m structurezyme.cli init --output /tmp/sz_seed.yml && grep input_csv /tmp/sz_seed.yml`
Expected: line `  input_csv: ''` present.

---

## Self-Review

**Spec coverage:**
- input_csv field + validate_paths → Task 1. ✓
- Runner seeding, resume-safe, CSV/pkl → Task 3 (loader in Task 2). ✓
- Enabled-module column validation → Tasks 2 + 3. ✓
- init template field → Task 4. ✓
- Docs tables + BYO-PDB future note → Task 5. ✓
- Tests (seed-csv, seed-pkl, resume-no-clobber, missing input_csv, missing enabled col ×2) → Tasks 1-4. ✓

**Placeholder scan:** none — all steps contain concrete code/commands.

**Type consistency:** `_load_input_frame(path:str)->DataFrame`, `missing_input_columns(df, enabled:set)->dict`, `REQUIRED_INPUT_COLUMNS`, `Runner._seed_and_validate()` used consistently across Tasks 2-3.

