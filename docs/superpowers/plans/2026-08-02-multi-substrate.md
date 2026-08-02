# Multi-Substrate Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add support for two or more substrates per enzyme to the StructureZyme pipeline via a `multi_substrate_mode` config flag, unblocking the halogenase pyridine+tryptophan run.

**Architecture:** A single `multi_substrate_mode: "off" | "separate" | "together"` flag (default `off`, fully backward-compatible) controls two orthogonal strategies. `separate` expands one input row into one row per substrate at seed time (`Entry=P12345` → `P12345__s0/__s1` plus a shared `enzyme_id` grouping column); all downstream steps run unchanged on the expanded frame. `together` keeps the row intact and co-docks multiple substrates: Chai receives dot-joined SMILES (native split), Boltz routes extra substrates as additional ligand entities, Vina is auto-skipped, and downstream analysis emits per-substrate `_s{i}` columns without hard-filtering. A shared `iter_substrates(row)` helper yields ordered `(smiles, name, moiety)` tuples so every consumer parses the packed columns identically.

**Tech Stack:** Python 3, pydantic v2 (config), pandas (frame plumbing), pytest (`-p no:cacheprovider`), RDKit (ligand parsing in docking/analysis steps), docko (Chai/Boltz/Vina docking backend, editable install at `/mnt/storage01/home/lherrmann/docko_lab_LCH/docko`).

## Global Constraints

- Default `multi_substrate_mode` MUST be `"off"`; with `off`, every existing frame, column name, and checkpoint hash is byte-for-byte unchanged (197 passed / 3 skipped baseline stays green).
- Multi-substrate delimiters (verbatim): `substrate_smiles` joins substrates with `.` (matches docko Chai native split); `substrate_name` and `substrate_moiety` parallel lists join with `|`.
- `separate` mode suffixes only expanded rows: `Entry` → `{Entry}__s{i}`; single-substrate rows are NOT expanded and NOT suffixed.
- `enzyme_id` column is added in ALL modes and equals `Entry` for single-substrate rows (it is the grouping key).
- Vina cannot co-dock (single RDKit mol, no cofactor slot); in `together` mode with >1 substrate Vina is auto-skipped, never errored.
- `together` downstream steps emit per-substrate `_s{i}` columns and MUST NOT hard-filter on any substrate.
- Boltz together-mode co-docking is capped at 2 substrates: substrate #0 is the affinity binder (`id:B`), substrate #1 rides in the `cofactor_smiles` cell (`id:C`). docko `write_yaml`/`run_boltz_affinity` has a known bug (`boltz.py:120` canonicalizes the whole cofactor string per split part), so a `cofactor_smiles` cell must contain at most ONE SMILES to route correctly. Therefore together-mode Boltz rejects: (a) >2 substrates, or (b) exactly 2 substrates AND a pre-existing non-empty `cofactor_smiles`. docko is NOT edited.
- Run tests with `conda activate /mnt/storage01/home/lherrmann/envs/filterzyme` then `pytest -p no:cacheprovider`. Commit with `git -c core.fsync=none commit ...`.
- All work stays inside the worktree `/mnt/storage01/home/lherrmann/structurezyme/.worktrees/multi-substrate` on branch `feat/multi-substrate`.

---

### Task 1: Config flag `multi_substrate_mode`

Add a validated top-level flag to `RunConfig`. This is the switch every later
task branches on. Uses a `Literal` type so pydantic rejects bad values at
construction and the CLI `init` template round-trips it.

**Files:**
- Modify: `structurezyme/config.py:42-46` (add field to `RunConfig`)
- Modify: `structurezyme/config.py:1-6` (import `Literal`)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `RunConfig.multi_substrate_mode: Literal["off", "separate", "together"]`
  (default `"off"`). Read elsewhere as `config.multi_substrate_mode`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
import pytest
from pydantic import ValidationError


def test_multi_substrate_mode_defaults_off():
    cfg = _minimal()
    assert cfg.multi_substrate_mode == "off"


def test_multi_substrate_mode_accepts_valid_values():
    for mode in ("off", "separate", "together"):
        cfg = RunConfig(
            paths={"output_root": "/tmp/out", "boltz_cache_dir": "/tmp/cache"},
            multi_substrate_mode=mode,
        )
        assert cfg.multi_substrate_mode == mode


def test_multi_substrate_mode_rejects_bad_value():
    with pytest.raises(ValidationError):
        RunConfig(
            paths={"output_root": "/tmp/out", "boltz_cache_dir": "/tmp/cache"},
            multi_substrate_mode="both",
        )


def test_multi_substrate_mode_roundtrips(tmp_path):
    cfg = RunConfig(
        paths={"output_root": "/tmp/out", "boltz_cache_dir": "/tmp/cache"},
        multi_substrate_mode="together",
    )
    p = tmp_path / "run.yml"
    cfg.write(p)
    assert load_config(p).multi_substrate_mode == "together"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -p no:cacheprovider tests/test_config.py -v`
Expected: the 4 new tests FAIL (`multi_substrate_mode` not a field; extra field
either ignored or rejected — either way the assertions/`ValidationError` do not
hold).

- [ ] **Step 3: Add the `Literal` import**

In `structurezyme/config.py`, change the typing import line at the top of the
file to include `Literal`:

```python
from typing import Literal
```

(Add this as a new import line after the existing `from pathlib import Path`
line at the top of the file.)

- [ ] **Step 4: Add the field to `RunConfig`**

In `structurezyme/config.py`, in the `RunConfig` class body (currently lines
42-45), add the field so the class reads:

```python
class RunConfig(BaseModel):
    paths: PathsConfig
    runtime: RuntimeConfig = RuntimeConfig()
    steps: StepsConfig = StepsConfig()
    multi_substrate_mode: Literal["off", "separate", "together"] = "off"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest -p no:cacheprovider tests/test_config.py -v`
Expected: all config tests PASS (original 3 + new 4 = 7).

- [ ] **Step 6: Run the full suite to confirm no regression**

Run: `pytest -p no:cacheprovider`
Expected: baseline preserved — 201 passed, 3 skipped (was 197 passed; +4 new).

- [ ] **Step 7: Commit**

```bash
git add structurezyme/config.py tests/test_config.py
git -c core.fsync=none commit -m "feat: add multi_substrate_mode config flag"
```

### Task 2: Shared `iter_substrates(row)` helper

One canonical parser for the packed substrate columns so every downstream
consumer (runner expansion, docking routing, analysis `_s{i}` emission) splits
identically. `substrate_smiles` splits on `.`; `substrate_name` /
`substrate_moiety` split on `|`. Parallel lists shorter than the SMILES list
are padded with `""`; longer ones are truncated to the substrate count. A
single-substrate row yields exactly one tuple, guaranteeing `off`-mode
consumers behave exactly as before.

**Files:**
- Modify: `structurezyme/utils/helpers.py` (append new function near the other
  ligand helpers, after `valid_file_list` at line 586)
- Test: `tests/test_iter_substrates.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  ```python
  def iter_substrates(row) -> list[tuple[str, str, str]]:
      """Yield ordered (smiles, name, moiety) per substrate in a frame row.

      row: a mapping/Series with keys 'substrate_smiles' (required),
           optional 'substrate_name', 'substrate_moiety'.
      Returns a list of (smiles, name, moiety) tuples, one per substrate,
      in column order. name/moiety are "" when absent or short.
      """
  ```
  Later tasks call `from structurezyme.utils.helpers import iter_substrates`
  and rely on: return type `list[tuple[str, str, str]]`; length ==
  number of `.`-separated SMILES; each element indexed as
  `smiles, name, moiety = subs[i]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_iter_substrates.py`:

```python
# tests/test_iter_substrates.py
import pandas as pd
from structurezyme.utils.helpers import iter_substrates


def test_single_substrate_yields_one_tuple():
    row = {"substrate_smiles": "CCO", "substrate_name": "ethanol",
           "substrate_moiety": "CO"}
    assert iter_substrates(row) == [("CCO", "ethanol", "CO")]


def test_two_substrates_aligned():
    row = {"substrate_smiles": "CCO.c1ccncc1",
           "substrate_name": "ethanol|pyridine",
           "substrate_moiety": "CO|n1ccccc1"}
    assert iter_substrates(row) == [
        ("CCO", "ethanol", "CO"),
        ("c1ccncc1", "pyridine", "n1ccccc1"),
    ]


def test_missing_name_and_moiety_columns_pad_empty():
    row = {"substrate_smiles": "CCO.c1ccncc1"}
    assert iter_substrates(row) == [("CCO", "", ""), ("c1ccncc1", "", "")]


def test_short_parallel_list_pads_empty():
    row = {"substrate_smiles": "CCO.c1ccncc1",
           "substrate_name": "ethanol"}
    assert iter_substrates(row) == [("CCO", "ethanol", ""), ("c1ccncc1", "", "")]


def test_accepts_pandas_series():
    s = pd.Series({"substrate_smiles": "CCO.O", "substrate_name": "a|b",
                   "substrate_moiety": "x|y"})
    assert iter_substrates(s) == [("CCO", "a", "x"), ("O", "b", "y")]


def test_nan_optional_columns_treated_as_absent():
    s = pd.Series({"substrate_smiles": "CCO.O",
                   "substrate_name": float("nan"),
                   "substrate_moiety": None})
    assert iter_substrates(s) == [("CCO", "", ""), ("O", "", "")]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -p no:cacheprovider tests/test_iter_substrates.py -v`
Expected: FAIL with `ImportError: cannot import name 'iter_substrates'`.

- [ ] **Step 3: Implement the helper**

Append to `structurezyme/utils/helpers.py` (after `valid_file_list`, keep the
module's existing plain-`def` style):

```python
def iter_substrates(row):
    """Yield ordered (smiles, name, moiety) tuples per substrate in a row.

    ``substrate_smiles`` splits on ``.`` and defines the substrate count.
    ``substrate_name`` and ``substrate_moiety`` split on ``|`` and are aligned
    positionally; missing/short lists pad with "" and longer lists truncate.
    A single-substrate row yields exactly one tuple.
    """
    def _get(key):
        val = row[key] if key in row else None
        # pandas NaN / None -> absent
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        return str(val)

    smiles_field = _get("substrate_smiles")
    smiles_list = [s for s in smiles_field.split(".") if s != ""]
    n = len(smiles_list)

    def _split_parallel(key):
        raw = _get(key)
        parts = raw.split("|") if raw != "" else []
        parts = parts[:n] + [""] * (n - len(parts))
        return parts

    names = _split_parallel("substrate_name")
    moieties = _split_parallel("substrate_moiety")
    return [(smiles_list[i], names[i], moieties[i]) for i in range(n)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -p no:cacheprovider tests/test_iter_substrates.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `pytest -p no:cacheprovider`
Expected: 207 passed, 3 skipped (201 from Task 1 + 6 new).

- [ ] **Step 6: Commit**

```bash
git add structurezyme/utils/helpers.py tests/test_iter_substrates.py
git -c core.fsync=none commit -m "feat: add iter_substrates row helper"
```

### Task 3: `_expand_substrates` helper (pure function)

The `separate`-mode row-expansion logic as a standalone pure function so it can
be unit-tested without a Runner. Given the seed frame and the mode string it
returns the frame to persist. `off`/`together`: frame unchanged except an
`enzyme_id` column is added (== `Entry`). `separate`: each multi-substrate row
becomes one row per substrate with `Entry` suffixed `__s{i}`, packed columns
unpacked to the single substrate's values, and `enzyme_id` set to the original
`Entry`; single-substrate rows keep their `Entry` unsuffixed.

**Files:**
- Modify: `structurezyme/runner.py` (add module-level function after
  `missing_input_columns`, line 38)
- Test: `tests/test_expand_substrates.py` (create)

**Interfaces:**
- Consumes: `iter_substrates` from `structurezyme.utils.helpers` (Task 2).
- Produces:
  ```python
  def _expand_substrates(df: pd.DataFrame, mode: str) -> pd.DataFrame: ...
  ```
  Guarantees: always adds an `enzyme_id` column. In `separate`, output row
  count == sum of per-row substrate counts; expanded rows have single-substrate
  packed columns (no `.`/`|`) and `Entry == f"{orig}__s{i}"`; single-substrate
  rows keep `Entry` unchanged. Column set is preserved (plus `enzyme_id`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_expand_substrates.py`:

```python
# tests/test_expand_substrates.py
import pandas as pd
from structurezyme.runner import _expand_substrates


def _frame():
    return pd.DataFrame({
        "Entry": ["P1", "P2"],
        "Sequence": ["M", "K"],
        "substrate_smiles": ["CCO.c1ccncc1", "O"],
        "substrate_name": ["ethanol|pyridine", "water"],
        "substrate_moiety": ["CO|n1ccccc1", "O"],
    })


def test_off_mode_adds_enzyme_id_only():
    df = _expand_substrates(_frame(), "off")
    assert list(df["Entry"]) == ["P1", "P2"]
    assert list(df["enzyme_id"]) == ["P1", "P2"]
    assert list(df["substrate_smiles"]) == ["CCO.c1ccncc1", "O"]


def test_together_mode_adds_enzyme_id_only():
    df = _expand_substrates(_frame(), "together")
    assert list(df["Entry"]) == ["P1", "P2"]
    assert list(df["enzyme_id"]) == ["P1", "P2"]
    assert list(df["substrate_smiles"]) == ["CCO.c1ccncc1", "O"]


def test_separate_mode_expands_multi_keeps_single():
    df = _expand_substrates(_frame(), "separate").reset_index(drop=True)
    assert list(df["Entry"]) == ["P1__s0", "P1__s1", "P2"]
    assert list(df["enzyme_id"]) == ["P1", "P1", "P2"]
    assert list(df["substrate_smiles"]) == ["CCO", "c1ccncc1", "O"]
    assert list(df["substrate_name"]) == ["ethanol", "pyridine", "water"]
    assert list(df["substrate_moiety"]) == ["CO", "n1ccccc1", "O"]
    # non-substrate columns are carried through per expanded row
    assert list(df["Sequence"]) == ["M", "M", "K"]


def test_separate_mode_single_substrate_not_suffixed():
    one = pd.DataFrame({"Entry": ["P9"], "Sequence": ["M"],
                        "substrate_smiles": ["O"], "substrate_name": ["water"],
                        "substrate_moiety": ["O"]})
    df = _expand_substrates(one, "separate")
    assert list(df["Entry"]) == ["P9"]
    assert list(df["enzyme_id"]) == ["P9"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -p no:cacheprovider tests/test_expand_substrates.py -v`
Expected: FAIL with `ImportError: cannot import name '_expand_substrates'`.

- [ ] **Step 3: Implement the helper**

In `structurezyme/runner.py`, add the import near the top (after the existing
`from .hashing import ...` line):

```python
from .utils.helpers import iter_substrates
```

Then add this function immediately after `missing_input_columns` (after
line 38):

```python
def _expand_substrates(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Return the seed frame to persist for the given multi_substrate_mode.

    off/together: frame unchanged plus an ``enzyme_id`` column (== ``Entry``).
    separate: multi-substrate rows are exploded into one row per substrate with
    ``Entry`` suffixed ``__s{i}`` and the packed substrate columns unpacked;
    single-substrate rows keep their ``Entry`` unsuffixed. ``enzyme_id`` always
    holds the original ``Entry``.
    """
    if mode != "separate":
        out = df.copy()
        out["enzyme_id"] = out["Entry"]
        return out

    rows = []
    for _, row in df.iterrows():
        subs = iter_substrates(row)
        multi = len(subs) > 1
        for i, (smiles, name, moiety) in enumerate(subs):
            new = row.to_dict()
            new["enzyme_id"] = row["Entry"]
            new["Entry"] = f"{row['Entry']}__s{i}" if multi else row["Entry"]
            new["substrate_smiles"] = smiles
            if "substrate_name" in new:
                new["substrate_name"] = name
            if "substrate_moiety" in new:
                new["substrate_moiety"] = moiety
            rows.append(new)
    return pd.DataFrame(rows, columns=list(df.columns) + ["enzyme_id"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -p no:cacheprovider tests/test_expand_substrates.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `pytest -p no:cacheprovider`
Expected: 211 passed, 3 skipped (207 from Task 2 + 4 new).

- [ ] **Step 6: Commit**

```bash
git add structurezyme/runner.py tests/test_expand_substrates.py
git -c core.fsync=none commit -m "feat: add _expand_substrates seed helper"
```

### Task 4: Wire expansion into `Runner._seed_and_validate`

Apply `_expand_substrates` to the freshly loaded input frame BEFORE writing the
seed pickle, so the persisted `_input.pkl` is already expanded and downstream
input hashing / resume-safety stay stable. Existing resume path (seed file
already present) is untouched.

**Files:**
- Modify: `structurezyme/runner.py:108-127` (`_seed_and_validate` body)
- Test: `tests/test_runner.py` (append)

**Interfaces:**
- Consumes: `_expand_substrates` (Task 3), `config.multi_substrate_mode`
  (Task 1).
- Produces: seed pickle at `checkpoints/_input.pkl` containing the expanded
  frame; `enzyme_id` column present for all modes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_runner.py`:

```python
def _cfg_multi(tmp_path, mode):
    csv = tmp_path / "input.csv"
    pd.DataFrame({"Sequence": ["M"], "substrate_smiles": ["CCO.O"],
                  "Entry": ["P1"], "vina_residues": ["1|2"],
                  "substrate_moiety": ["CO|O"],
                  "substrate_name": ["ethanol|water"]}).to_csv(csv, index=False)
    return RunConfig(
        paths={"output_root": str(tmp_path),
               "boltz_cache_dir": str(tmp_path / "cache"),
               "input_csv": str(csv)},
        runtime={"user": "tester", "run_id": "r1"},
        multi_substrate_mode=mode,
    )


def test_seed_separate_mode_expands_pickle(tmp_path, monkeypatch):
    for spec in registry.STEPS.values():
        monkeypatch.setattr(spec, "runner",
                            lambda ctx, spec: pd.DataFrame({"Entry": [spec.name]}),
                            raising=False)
    r = Runner(_cfg_multi(tmp_path, "separate"))
    r._seed_and_validate()
    seed = pd.read_pickle(r.layout.checkpoint_path("_input"))
    assert list(seed["Entry"]) == ["P1__s0", "P1__s1"]
    assert list(seed["enzyme_id"]) == ["P1", "P1"]
    assert list(seed["substrate_smiles"]) == ["CCO", "O"]


def test_seed_off_mode_adds_enzyme_id_only(tmp_path, monkeypatch):
    for spec in registry.STEPS.values():
        monkeypatch.setattr(spec, "runner",
                            lambda ctx, spec: pd.DataFrame({"Entry": [spec.name]}),
                            raising=False)
    r = Runner(_cfg_multi(tmp_path, "off"))
    r._seed_and_validate()
    seed = pd.read_pickle(r.layout.checkpoint_path("_input"))
    assert list(seed["Entry"]) == ["P1"]
    assert list(seed["enzyme_id"]) == ["P1"]
    assert list(seed["substrate_smiles"]) == ["CCO.O"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -p no:cacheprovider tests/test_runner.py -k multi -v`
Expected: FAIL — seed pickle has no `enzyme_id` column (KeyError) and `Entry`
not expanded.

- [ ] **Step 3: Wire the helper in**

In `structurezyme/runner.py`, in `_seed_and_validate`, change the load/seed
branch (currently lines 115-120) from:

```python
        seed = self.layout.checkpoint_path("_input")
        if seed.is_file():
            df = pd.read_pickle(seed)
        else:
            df = _load_input_frame(self.config.paths.input_csv)
            df.to_pickle(seed)
```

to:

```python
        seed = self.layout.checkpoint_path("_input")
        if seed.is_file():
            df = pd.read_pickle(seed)
        else:
            df = _load_input_frame(self.config.paths.input_csv)
            df = _expand_substrates(df, self.config.multi_substrate_mode)
            df.to_pickle(seed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -p no:cacheprovider tests/test_runner.py -k multi -v`
Expected: both new tests PASS.

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `pytest -p no:cacheprovider`
Expected: 213 passed, 3 skipped (211 from Task 3 + 2 new). Existing runner
resume/checkpoint tests still pass (off-mode adds only `enzyme_id`, which does
not change existing assertions).

- [ ] **Step 6: Commit**

```bash
git add structurezyme/runner.py tests/test_runner.py
git -c core.fsync=none commit -m "feat: expand substrates when seeding the input frame"
```

### Task 5: Chai `together`-mode routing (native, verification only)

In `together` mode Chai receives the intact dot-joined `substrate_smiles`.
docko `run_chai` splits SMILES on `.` internally (`chai.py`), and enzymetk's
`Chai` step passes the raw cell straight through (`dock_chai_step.py`
`__execute`). So NO production change is needed — Chai co-folds multiple
substrates natively. This task adds a regression test that pins the contract:
`run_chai` must be called with the unmodified dot-joined SMILES for a
together-mode row, so a future refactor of `run_chai` (step_runner) cannot
silently drop the second substrate.

**Files:**
- Modify: none (production code unchanged).
- Test: `tests/test_chai_multi_substrate.py` (create).

**Interfaces:**
- Consumes: `structurezyme.step_runners.run_chai` (existing);
  a together-mode seed frame (Task 4) whose `substrate_smiles` is dot-joined.
- Produces: nothing new; guards existing behaviour.

- [ ] **Step 1: Write the failing test**

Create `tests/test_chai_multi_substrate.py`. The test builds a minimal `ctx`
and monkeypatches the `Chai` enzymetk class to a recorder so no docking runs;
it asserts the substrate string passed to `Chai(...)` is the intact
dot-joined value.

```python
# tests/test_chai_multi_substrate.py
import pandas as pd
import pytest
import structurezyme.step_runners as sr


class _RecordingChai:
    last_args = None

    def __init__(self, id_col, seq_col, substrate_col, cofactor_col,
                 output_dir, num_threads):
        _RecordingChai.last_args = (id_col, seq_col, substrate_col,
                                    cofactor_col)
        self.output_dir = output_dir

    def __rrshift__(self, df):  # df << step
        df = df.copy()
        df["output_dir"] = [str(self.output_dir)] * len(df)
        return df

    def __rshift__(self, other):  # step >> Save(...)
        return self


def test_chai_receives_intact_dotjoined_smiles(tmp_path, monkeypatch):
    # together-mode frame: two substrates packed with '.'
    df = pd.DataFrame({"Entry": ["P1"], "Sequence": ["M"],
                       "substrate_smiles": ["CCO.c1ccncc1"],
                       "enzyme_id": ["P1"]})

    # stub enzymetk Chai + Save so nothing docks
    import enzymetk.dock_chai_step as chai_mod
    monkeypatch.setattr(chai_mod, "Chai", _RecordingChai, raising=True)
    from structurezyme.steps import save_step
    monkeypatch.setattr(save_step, "Save",
                        lambda *a, **k: type("S", (), {"__rrshift__":
                            lambda self, d: d})(), raising=True)

    class _Ctx:
        class config:
            class runtime:
                num_threads = 1
        def step_dir(self, name):
            d = tmp_path / name
            d.mkdir(parents=True, exist_ok=True)
            return d
    ctx = _Ctx()
    monkeypatch.setattr(sr, "_first_input", lambda ctx, spec: df,
                        raising=True)
    monkeypatch.setattr(sr, "_docking_dir", lambda ctx: tmp_path,
                        raising=True)

    class _Spec:
        name = "chai"
    out = sr.run_chai(ctx, _Spec())

    # the substrate COLUMN name is what Chai is told to read; the frame still
    # holds the intact dot-joined value under that column
    assert _RecordingChai.last_args[2] == "substrate_smiles"
    assert out["substrate_smiles"].iloc[0] == "CCO.c1ccncc1"
```

- [ ] **Step 2: Run the test to verify it passes as a contract guard**

Run: `pytest -p no:cacheprovider tests/test_chai_multi_substrate.py -v`
Expected: PASS (no production change needed — this pins current behaviour).
If it FAILS, the stubbing shape is wrong; adjust the `_RecordingChai`
dunder methods to match how `run_chai` composes `Chai(...) >> Save(...)` and
applies `df << (...)`, then re-run.

- [ ] **Step 3: Run the full suite to confirm no regression**

Run: `pytest -p no:cacheprovider`
Expected: 214 passed, 3 skipped (213 from Task 4 + 1 new).

- [ ] **Step 4: Commit**

```bash
git add tests/test_chai_multi_substrate.py
git -c core.fsync=none commit -m "test: pin Chai together-mode dot-joined substrate contract"
```

### Task 6: Boltz `together`-mode extra-ligand routing

In `together` mode, split the intact dot-joined `substrate_smiles` so substrate
#0 becomes the Boltz affinity binder (`id:B`) and substrate #1 rides in the
`cofactor_smiles` cell (→ docko `id:C`). Enforce the 2-substrate cap and the
"no pre-existing cofactor when 2 substrates" rule (see Global Constraints).
`off`/`separate` modes are untouched (each row already has a single substrate).

**Files:**
- Modify: `structurezyme/step_runners.py:263-291` (`run_boltz`)
- Test: `tests/test_boltz_multi_substrate.py` (create)

**Interfaces:**
- Consumes: `iter_substrates` (Task 2); `config.multi_substrate_mode` (Task 1);
  a together-mode frame whose `substrate_smiles` is dot-joined.
- Produces: a private helper
  `_boltz_together_frame(df: pd.DataFrame) -> pd.DataFrame` in
  `step_runners.py` that returns a copy where `substrate_smiles` holds only
  substrate #0 and `cofactor_smiles` holds substrate #1 (raising `ValueError`
  on the capped cases). `run_boltz` calls it only when
  `ctx.config.multi_substrate_mode == "together"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_boltz_multi_substrate.py`:

```python
# tests/test_boltz_multi_substrate.py
import pandas as pd
import pytest
from structurezyme.step_runners import _boltz_together_frame


def test_two_substrates_no_cofactor_routes_second_to_cofactor():
    df = pd.DataFrame({"Entry": ["P1"], "Sequence": ["M"],
                       "substrate_smiles": ["CCO.c1ccncc1"]})
    out = _boltz_together_frame(df)
    assert list(out["substrate_smiles"]) == ["CCO"]
    assert list(out["cofactor_smiles"]) == ["c1ccncc1"]


def test_single_substrate_passthrough_empty_cofactor():
    df = pd.DataFrame({"Entry": ["P1"], "Sequence": ["M"],
                       "substrate_smiles": ["CCO"]})
    out = _boltz_together_frame(df)
    assert list(out["substrate_smiles"]) == ["CCO"]
    assert list(out["cofactor_smiles"]) == [""]


def test_three_substrates_rejected():
    df = pd.DataFrame({"Entry": ["P1"], "Sequence": ["M"],
                       "substrate_smiles": ["CCO.O.N"]})
    with pytest.raises(ValueError, match="at most 2 substrates"):
        _boltz_together_frame(df)


def test_two_substrates_with_existing_cofactor_rejected():
    df = pd.DataFrame({"Entry": ["P1"], "Sequence": ["M"],
                       "substrate_smiles": ["CCO.c1ccncc1"],
                       "cofactor_smiles": ["[Fe]"]})
    with pytest.raises(ValueError, match="pre-existing cofactor"):
        _boltz_together_frame(df)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -p no:cacheprovider tests/test_boltz_multi_substrate.py -v`
Expected: FAIL with `ImportError: cannot import name '_boltz_together_frame'`.

- [ ] **Step 3: Implement the helper and wire it into `run_boltz`**

In `structurezyme/step_runners.py`, add near the top (with the other imports):

```python
from structurezyme.utils.helpers import iter_substrates
```

Add this helper above `run_boltz` (before line 263):

```python
def _boltz_together_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Route a together-mode frame for Boltz co-docking.

    substrate #0 stays the affinity binder in ``substrate_smiles``; substrate
    #1 (if any) is written to ``cofactor_smiles`` (docko ``id:C``). Enforces the
    2-substrate cap and rejects a 2nd substrate when a real cofactor already
    exists, because docko can only route one extra ligand SMILES safely.
    """
    out = df.copy()
    new_sub, new_cof = [], []
    existing = out["cofactor_smiles"] if "cofactor_smiles" in out.columns else None
    for pos, (_, row) in enumerate(out.iterrows()):
        subs = iter_substrates(row)
        if len(subs) > 2:
            raise ValueError(
                f"together-mode Boltz supports at most 2 substrates, got "
                f"{len(subs)} for Entry={row['Entry']!r}"
            )
        prior = "" if existing is None else str(existing.iloc[pos] or "")
        if prior in ("", "nan", "None"):
            prior = ""
        if len(subs) == 2 and prior:
            raise ValueError(
                f"together-mode Boltz cannot co-dock a 2nd substrate together "
                f"with a pre-existing cofactor for Entry={row['Entry']!r}"
            )
        new_sub.append(subs[0][0])
        new_cof.append(subs[1][0] if len(subs) == 2 else prior)
    out["substrate_smiles"] = new_sub
    out["cofactor_smiles"] = new_cof
    return out
```

Then in `run_boltz`, immediately after `df_chai = _first_input(ctx, spec)`
(line 271) and BEFORE the `if "cofactor_smiles" not in df_chai.columns` guard,
insert:

```python
    if ctx.config.multi_substrate_mode == "together":
        df_chai = _boltz_together_frame(df_chai)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -p no:cacheprovider tests/test_boltz_multi_substrate.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `pytest -p no:cacheprovider`
Expected: 218 passed, 3 skipped (214 from Task 5 + 4 new).

- [ ] **Step 6: Commit**

```bash
git add structurezyme/step_runners.py tests/test_boltz_multi_substrate.py
git -c core.fsync=none commit -m "feat: route 2nd substrate to Boltz cofactor slot in together mode"
```

### Task 7: Vina auto-skip in `together` mode

Vina cannot co-dock (single RDKit mol, no cofactor slot). In `together` mode
with a multi-substrate row, skip Vina entirely and pass every row through with
`vina_dir = pd.NA`, mirroring the existing "nobody has residues" pass-through
(step_runners.py:334-338) so `docking_metrics` falls back to the boltz/chai
co-folded pose. `off`/`separate` modes are unaffected (single substrate per
row). Together-mode rows with only ONE substrate still dock normally.

**Files:**
- Modify: `structurezyme/step_runners.py:294-313` (top of `run_vina`)
- Test: `tests/test_vina_multi_substrate.py` (create)

**Interfaces:**
- Consumes: `iter_substrates` (Task 2, already imported in Task 6);
  `config.multi_substrate_mode` (Task 1).
- Produces: no new public symbol; `run_vina` returns the input frame with a
  `vina_dir` column of `pd.NA` and writes `vina.pkl` when together-mode skip
  triggers.

- [ ] **Step 1: Write the failing test**

Create `tests/test_vina_multi_substrate.py`:

```python
# tests/test_vina_multi_substrate.py
import pandas as pd
import structurezyme.step_runners as sr


def _ctx(tmp_path, mode):
    class _Ctx:
        class config:
            class runtime:
                num_threads = 1
            multi_substrate_mode = mode
        def step_dir(self, name):
            d = tmp_path / name
            d.mkdir(parents=True, exist_ok=True)
            return d
    return _Ctx()


class _Spec:
    name = "vina"


def test_together_multi_substrate_skips_vina(tmp_path, monkeypatch):
    df = pd.DataFrame({"Entry": ["P1"], "Sequence": ["M"],
                       "substrate_smiles": ["CCO.c1ccncc1"],
                       "catalytic_residues": ["1|2"],
                       "chai_dir": ["/x"], "boltz_dir": ["/y"]})
    monkeypatch.setattr(sr, "_first_input", lambda ctx, spec: df, raising=True)
    monkeypatch.setattr(sr, "_docking_dir", lambda ctx: tmp_path, raising=True)
    out = sr.run_vina(_ctx(tmp_path, "together"), _Spec())
    assert list(out["Entry"]) == ["P1"]
    assert out["vina_dir"].isna().all()
    assert (tmp_path / "vina.pkl").is_file()


def test_together_single_substrate_does_not_skip(tmp_path, monkeypatch):
    # single substrate in together mode: guard must NOT short-circuit.
    # We assert the skip branch is not taken by checking the function proceeds
    # past the guard (it will raise later trying to import Vina/real docking),
    # so we only verify the guard predicate here via the helper.
    from structurezyme.step_runners import _together_skips_vina
    df1 = pd.DataFrame({"Entry": ["P1"], "substrate_smiles": ["CCO"]})
    df2 = pd.DataFrame({"Entry": ["P1"], "substrate_smiles": ["CCO.O"]})
    assert _together_skips_vina("together", df1) is False
    assert _together_skips_vina("together", df2) is True
    assert _together_skips_vina("off", df2) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -p no:cacheprovider tests/test_vina_multi_substrate.py -v`
Expected: FAIL — `_together_skips_vina` missing (ImportError) and together
multi-substrate does not yet skip.

- [ ] **Step 3: Implement the guard helper and wire it in**

In `structurezyme/step_runners.py`, add this helper above `run_vina`
(before line 294):

```python
def _together_skips_vina(mode: str, df: pd.DataFrame) -> bool:
    """True when together-mode has any multi-substrate row (Vina can't co-dock)."""
    if mode != "together":
        return False
    return any(len(iter_substrates(row)) > 1 for _, row in df.iterrows())
```

Then in `run_vina`, immediately after `df_boltz = _first_input(ctx, spec)`
(line 309), insert:

```python
    if _together_skips_vina(ctx.config.multi_substrate_mode, df_boltz):
        from structurezyme.utils.helpers import log_boxed_note
        log_boxed_note(
            "Skipping vina docking (together-mode co-docking) for all rows; "
            "using chai/boltz co-folded poses."
        )
        df_boltz = df_boltz.copy()
        df_boltz["vina_dir"] = pd.NA
        df_boltz.to_pickle(out_dir / "vina.pkl")
        return df_boltz
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -p no:cacheprovider tests/test_vina_multi_substrate.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `pytest -p no:cacheprovider`
Expected: 220 passed, 3 skipped (218 from Task 6 + 2 new).

- [ ] **Step 6: Commit**

```bash
git add structurezyme/step_runners.py tests/test_vina_multi_substrate.py
git -c core.fsync=none commit -m "feat: auto-skip vina for together-mode co-docking"
```

### Task 8: `together`-mode per-substrate + joint ligand RMSD

**Highest-risk task.** Today `LigandRMSD.__execute` compares the SAME single
substrate's pose across docking tools: it extracts all hetatm chains from a
paired PDB, filters to the ~2 chains matching the single `substrate_smiles` via
`closest_ligands_by_element_composition`, and computes one `ligand_rmsd`. In
`together` mode a paired PDB holds TWO different substrates (two hetatm chains
per docked structure). We compute cross-tool pose agreement PER substrate:
for substrate #i, pick the chain whose element composition best matches
substrate #i's SMILES in each of the two docked structures, RMSD them →
`ligand_rmsd_s{i}`; and superpose both substrates together → `ligand_rmsd_joint`.
`off`/`separate` modes keep the exact current single-column `ligand_rmsd`
behaviour (one substrate per row → the new code path is only taken when
`iter_substrates(row)` yields >1).

This task is decomposed to isolate the risky matching logic into a pure,
unit-tested helper first, then wire it into the step.

**Files:**
- Modify: `structurezyme/steps/computeligandRMSD_step.py` (add helper +
  branch in `__execute` around lines 311-387)
- Test: `tests/test_ligand_rmsd_multi_substrate.py` (create)

**Interfaces:**
- Consumes: `iter_substrates` (Task 2);
  `closest_ligands_by_element_composition(ligand_mols, reference_smiles, top_k=2) -> list[Mol]`
  (helpers.py:536); `get_tool_from_structure_name` (same module, line 78);
  `rdkit.Chem.rdMolAlign.CalcRMS`.
- Produces a module-level pure helper in `computeligandRMSD_step.py`:
  ```python
  def _match_substrate_chains(ligands, substrate_smiles_list):
      """Return list aligned to substrate_smiles_list; element i is the single
      RDKit Mol from `ligands` whose atom composition best matches
      substrate_smiles_list[i] (via closest_ligands_by_element_composition with
      top_k=1), or None if no ligand matched. A ligand already assigned to an
      earlier substrate is not reused."""
  ```
  Later steps do not consume this; it is internal to the RMSD step.

- [ ] **Step 1: Write the failing test for the matcher helper**

Create `tests/test_ligand_rmsd_multi_substrate.py`:

```python
# tests/test_ligand_rmsd_multi_substrate.py
from rdkit import Chem
from structurezyme.steps.computeligandRMSD_step import _match_substrate_chains


def _mol(smiles):
    return Chem.MolFromSmiles(smiles)


def test_matches_each_substrate_to_closest_chain():
    # two chains present: ethanol-like and pyridine-like
    ethanol = _mol("CCO")
    pyridine = _mol("c1ccncc1")
    ligands = [pyridine, ethanol]  # deliberately out of order
    matched = _match_substrate_chains(ligands, ["CCO", "c1ccncc1"])
    assert matched[0].GetNumAtoms() == ethanol.GetNumAtoms()
    assert matched[1].GetNumAtoms() == pyridine.GetNumAtoms()


def test_no_chain_matches_returns_none_slot():
    ethanol = _mol("CCO")
    matched = _match_substrate_chains([ethanol], ["CCO", "c1ccncc1"])
    assert matched[0] is not None
    # only one ligand present; second substrate has no remaining chain
    assert matched[1] is None


def test_does_not_reuse_a_chain_across_substrates():
    ethanol = _mol("CCO")
    matched = _match_substrate_chains([ethanol], ["CCO", "CCO"])
    assert matched[0] is not None
    assert matched[1] is None  # the single chain is consumed by substrate 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest -p no:cacheprovider tests/test_ligand_rmsd_multi_substrate.py -v`
Expected: FAIL with `ImportError: cannot import name '_match_substrate_chains'`.

- [ ] **Step 3: Implement the matcher helper**

In `structurezyme/steps/computeligandRMSD_step.py`, add `iter_substrates` to
the existing helpers import block (lines 25-32):

```python
from structurezyme.utils.helpers import (
    clean_plt,
    get_hetatm_chain_ids,
    extract_chain_as_rdkit_mol,
    closest_ligands_by_element_composition,
    norm_l1_dist,
    atom_composition_fingerprint,
    iter_substrates,
)
```

Add this module-level function after `get_tool_from_structure_name`
(after line 91):

```python
def _match_substrate_chains(ligands, substrate_smiles_list):
    """Map each substrate SMILES to its best-matching ligand chain.

    Returns a list aligned to ``substrate_smiles_list``; element i is the RDKit
    Mol from ``ligands`` whose atom composition is closest to
    ``substrate_smiles_list[i]`` (top_k=1), or None if nothing is left to match.
    A chain assigned to an earlier substrate is not reused (matched by object
    identity).
    """
    remaining = [m for m in ligands if m is not None]
    matched = []
    for smiles in substrate_smiles_list:
        if not remaining:
            matched.append(None)
            continue
        best = closest_ligands_by_element_composition(remaining, smiles, top_k=1)
        if not best:
            matched.append(None)
            continue
        chosen = best[0]
        matched.append(chosen)
        remaining = [m for m in remaining if m is not chosen]
    return matched
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest -p no:cacheprovider tests/test_ligand_rmsd_multi_substrate.py -v`
Expected: all 3 matcher tests PASS.

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `pytest -p no:cacheprovider`
Expected: 223 passed, 3 skipped (220 from Task 7 + 3 new).

- [ ] **Step 6: Commit the matcher helper**

```bash
git add structurezyme/steps/computeligandRMSD_step.py tests/test_ligand_rmsd_multi_substrate.py
git -c core.fsync=none commit -m "feat: add per-substrate chain matcher for ligand RMSD"
```

> **Note:** wiring `_match_substrate_chains` into `LigandRMSD.__execute` to emit
> `ligand_rmsd_s{i}` / `ligand_rmsd_joint` columns is Task 8b (next task),
> kept separate so the risky matcher lands and is reviewed on its own.

### Task 8b: Wire per-substrate RMSD into `LigandRMSD.__execute`

Use `_match_substrate_chains` (Task 8) inside `LigandRMSD.__execute` so that,
when a row has >1 substrate (`iter_substrates`), each docked PDB emits
`ligand_rmsd_s{i}` per substrate plus `ligand_rmsd_joint`. Single-substrate
rows keep the existing single `ligand_rmsd` column unchanged.

**Files:**
- Modify: `structurezyme/steps/computeligandRMSD_step.py:311-387` (`__execute`)
- Test: `tests/test_ligand_rmsd_multi_substrate.py` (append)

**Interfaces:**
- Consumes: `_match_substrate_chains` (Task 8), `iter_substrates` (Task 2),
  `rdMolAlign.CalcRMS`.
- Produces: extra rmsd dict keys `ligand_rmsd_s0`, `ligand_rmsd_s1`,
  `ligand_rmsd_joint` on multi-substrate rows (single-substrate rows unchanged).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ligand_rmsd_multi_substrate.py`:

```python
def test_per_substrate_rmsd_keys_emitted(tmp_path, monkeypatch):
    import structurezyme.steps.computeligandRMSD_step as mod

    # one entry dir with one paired PDB
    entry = tmp_path / "P1"
    entry.mkdir()
    (entry / "chai_0__boltz_0.pdb").write_text("dummy")

    df = pd.DataFrame({"Entry": ["P1"],
                       "substrate_smiles": ["CCO.c1ccncc1"]})

    # stub chain extraction: 2 chains per structure, matching the 2 substrates
    from rdkit import Chem
    eth = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    pyr = Chem.AddHs(Chem.MolFromSmiles("c1ccncc1"))
    Chem.AllChem.EmbedMolecule(eth); Chem.AllChem.EmbedMolecule(pyr)
    monkeypatch.setattr(mod, "get_hetatm_chain_ids",
                        lambda p: ["B", "C"], raising=True)
    monkeypatch.setattr(mod, "extract_chain_as_rdkit_mol",
                        lambda p, cid, sanitize=False:
                            (eth if cid == "B" else pyr), raising=True)

    step = mod.LigandRMSD(input_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
    rmsd_df, _ = step.execute(df)
    cols = set(rmsd_df.columns)
    assert "ligand_rmsd_s0" in cols
    assert "ligand_rmsd_s1" in cols
    assert "ligand_rmsd_joint" in cols
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest -p no:cacheprovider tests/test_ligand_rmsd_multi_substrate.py::test_per_substrate_rmsd_keys_emitted -v`
Expected: FAIL — the `_s0`/`_s1`/`_joint` columns are not produced yet.

- [ ] **Step 3: Implement the multi-substrate branch (real cross-tool RMSD)**

Context (verified in `superimposestructures_step.py`
`superimpose_different_docked_structure`, lines 211-221): a paired PDB places
tool-1's ligand chains at IDs `T, U, ...` (`chr(84+i)`) and tool-2's aligned
ligand chains at `V, W, ...` (`chr(86+i)`), all in one coordinate frame after
`transform.apply`. So a 2-substrate together-mode pair holds 4 hetatm chains:
`{T,U}` (tool1) and `{V,W}` (tool2). Per-substrate cross-tool RMSD = match
substrate #i to one tool1 chain and one tool2 chain, then `CalcRMS` them.

In `structurezyme/steps/computeligandRMSD_step.py` `__execute`, keep lines
314-321 (building `chain_ids` and `ligands`). Replace the RMSD computation
(lines 323-387) so it branches on substrate count. Because `ligands` is built
in `chain_ids` order, carry the chain id alongside each mol:

```python
                # chain-id-aware ligand list (chain_ids and ligands are aligned)
                chain_mols = list(zip(chain_ids, ligands))

                subs = iter_substrates(
                    df.loc[df[self.entry_col] == sub_dir.name].iloc[0]
                )

                if len(subs) <= 1:
                    # ---- existing single-substrate path (unchanged) ----
                    filtered_ligands = closest_ligands_by_element_composition(
                        ligands, substrate_smiles)
                    if len(filtered_ligands) > 2:
                        logger.warning('More than 2 ligands were found matching the smile string.')
                        continue
                    if len(filtered_ligands) == 0:
                        logger.warning(f"No valid ligands found for entry {sub_dir.name}. Skipping.")
                        continue
                    ligand1, ligand2 = filtered_ligands[0], filtered_ligands[1]
                    if ligand1 is None or ligand2 is None:
                        logger.warning(f"Could not extract both ligands, skipping {pdb_file_path}")
                        continue
                    try:
                        Chem.SanitizeMol(ligand1); Chem.SanitizeMol(ligand2)
                        ligand1 = Chem.RemoveHs(ligand1); ligand2 = Chem.RemoveHs(ligand2)
                        if ligand1.GetNumConformers() == 0:
                            AllChem.EmbedMolecule(ligand1)
                        if ligand2.GetNumConformers() == 0:
                            AllChem.EmbedMolecule(ligand2)
                    except Chem.rdchem.AtomValenceException as e:
                        logger.warning(f"Valence error in {pdb_file_path.name}: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Unexpected RDKit error in {pdb_file_path.name}: {e}")
                        continue
                    try:
                        rmsd = rdMolAlign.CalcRMS(ligand1, ligand2, maxMatches=self.maxMatches)
                    except RuntimeError as e:
                        logger.warning(f"LigandRMSD calc failed for {pdb_file_path.name}: {e}")
                        continue
                    extra_rmsd = {}
                else:
                    # ---- together-mode per-substrate cross-tool RMSD ----
                    # Split chains by superimpose convention: tool1 = T,U,... ;
                    # tool2 = V,W,...  (see superimposestructures_step.py:211-221)
                    tool1 = [m for cid, m in chain_mols if cid in ("T", "U")]
                    tool2 = [m for cid, m in chain_mols if cid in ("V", "W")]
                    smiles_list = [s[0] for s in subs]
                    matched1 = _match_substrate_chains(tool1, smiles_list)
                    matched2 = _match_substrate_chains(tool2, smiles_list)

                    def _prep(m):
                        if m is None:
                            return None
                        try:
                            Chem.SanitizeMol(m); m = Chem.RemoveHs(m)
                            if m.GetNumConformers() == 0:
                                AllChem.EmbedMolecule(m)
                            return m
                        except Exception as e:
                            logger.warning(f"RDKit prep failed in {pdb_file_path.name}: {e}")
                            return None

                    extra_rmsd = {}
                    per_sub_rmsds = []
                    for i in range(len(subs)):
                        a, b = _prep(matched1[i]), _prep(matched2[i])
                        if a is None or b is None:
                            extra_rmsd[f"ligand_rmsd_s{i}"] = np.nan
                            continue
                        try:
                            r = rdMolAlign.CalcRMS(a, b, maxMatches=self.maxMatches)
                        except RuntimeError:
                            r = np.nan
                        extra_rmsd[f"ligand_rmsd_s{i}"] = r
                        if not np.isnan(r):
                            per_sub_rmsds.append(r)
                    # joint: combine each tool's matched chains and RMSD together
                    v1 = [m for m in (_prep(x) for x in matched1) if m is not None]
                    v2 = [m for m in (_prep(x) for x in matched2) if m is not None]
                    if len(v1) >= 2 and len(v2) >= 2:
                        try:
                            j1 = Chem.CombineMols(v1[0], v1[1])
                            j2 = Chem.CombineMols(v2[0], v2[1])
                            extra_rmsd["ligand_rmsd_joint"] = rdMolAlign.CalcRMS(
                                j1, j2, maxMatches=self.maxMatches)
                        except (RuntimeError, ValueError):
                            extra_rmsd["ligand_rmsd_joint"] = np.nan
                    else:
                        extra_rmsd["ligand_rmsd_joint"] = np.nan
                    # primary ligand_rmsd = mean of per-substrate RMSDs
                    rmsd = float(np.mean(per_sub_rmsds)) if per_sub_rmsds else np.nan
```

Then merge `extra_rmsd` into the appended record (replace lines 379-387):

```python
                record = {
                    'Entry': entry_name,
                    'pdb_file': pdb_file_path.name,
                    'docked_structure1': docked_structure1_name,
                    'docked_structure2': docked_structure2_name,
                    'tool1': tool1_name,
                    'tool2': tool2_name,
                    'ligand_rmsd': rmsd,
                }
                record.update(extra_rmsd)
                rmsd_values.append(record)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest -p no:cacheprovider tests/test_ligand_rmsd_multi_substrate.py::test_per_substrate_rmsd_keys_emitted -v`
Expected: PASS — `ligand_rmsd_s0`, `ligand_rmsd_s1`, `ligand_rmsd_joint`
columns present. (The stub gives chains B/C, not T/U/V/W, so the tool1/tool2
split yields empty lists and the `_s{i}` values are NaN — the test asserts only
that the COLUMNS are emitted, which is the contract. A follow-up integration
test with real superimposed PDBs is out of scope for this unit task.)

> **Note:** if the stub's chain IDs (B/C) cause both tool lists to be empty and
> you want the unit test to also exercise a non-NaN path, update the test's
> `get_hetatm_chain_ids` stub to return `["T", "U", "V", "W"]` and
> `extract_chain_as_rdkit_mol` to return ethanol for T/V and pyridine for U/W;
> then additionally assert `rmsd_df["ligand_rmsd_s0"].notna().any()`.

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `pytest -p no:cacheprovider`
Expected: 224 passed, 3 skipped (223 from Task 8 + 1 new). Existing
single-substrate LigandRMSD tests unchanged (they hit the `len(subs) <= 1`
branch, byte-identical to the old code).

- [ ] **Step 6: Commit**

```bash
git add structurezyme/steps/computeligandRMSD_step.py tests/test_ligand_rmsd_multi_substrate.py
git -c core.fsync=none commit -m "feat: emit per-substrate and joint ligand RMSD in together mode"
```

### Task 9: Geometric filter (MCS) per-substrate `_s{i}` columns

In `together` mode, the geometric filter must analyze EACH co-docked substrate,
emitting its distance/method outputs suffixed `_s{i}`, and MUST NOT hard-filter
any substrate (Global Constraints). Single-substrate rows keep today's exact
column names (no suffix). This is the template pattern reused by Tasks 9b-9d
(plip, ligand_sasa, fpocket).

**Files:**
- Modify: `structurezyme/steps/geometric_filtering_cofactor_MCS.py:448-571`
  (`__execute` per-row body)
- Test: `tests/test_geometric_filter_multi_substrate.py` (create)

**Interfaces:**
- Consumes: `iter_substrates` (Task 2);
  `closest_ligands_by_element_composition(..., top_k=1)`;
  `moiety_centroid_with_fallbacks`, `find_min_distance`, `get_all_nucs_atom_coords`
  (same module).
- Produces: per-substrate result keys in `together` mode:
  `ligand_moiety_method_s{i}`, `distance_ligand_to_closest_nuc_s{i}`,
  `distance_ligand_to_cofactor_s{i}`. Single-substrate rows: unchanged keys
  (`ligand_moiety_method`, `distance_ligand_to_closest_nuc`,
  `distance_ligand_to_cofactor`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_geometric_filter_multi_substrate.py`. It refactors the
per-substrate distance work into a helper `_analyze_substrate(...)` so it can be
unit-tested without real PDB I/O; the test asserts the helper produces the
`_s{i}`-suffixable base keys for one substrate.

```python
# tests/test_geometric_filter_multi_substrate.py
from structurezyme.steps.geometric_filtering_cofactor_MCS import (
    _suffix_keys,
)


def test_suffix_keys_appends_index():
    base = {"distance_ligand_to_closest_nuc": {"CYS": 3.1},
            "ligand_moiety_method": "mcs"}
    out = _suffix_keys(base, 1)
    assert out == {"distance_ligand_to_closest_nuc_s1": {"CYS": 3.1},
                   "ligand_moiety_method_s1": "mcs"}


def test_suffix_keys_noop_for_none_index():
    base = {"ligand_moiety_method": "mcs"}
    assert _suffix_keys(base, None) == {"ligand_moiety_method": "mcs"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest -p no:cacheprovider tests/test_geometric_filter_multi_substrate.py -v`
Expected: FAIL with `ImportError: cannot import name '_suffix_keys'`.

- [ ] **Step 3: Implement `_suffix_keys` and the per-substrate loop**

In `structurezyme/steps/geometric_filtering_cofactor_MCS.py`, add
`iter_substrates` to the helpers import (find the existing
`from structurezyme.utils.helpers import (...)` block and add the name).

Add this module-level helper (near the other module-level functions, e.g.
after `nearest_centroid_distance` around line 240):

```python
def _suffix_keys(result: dict, idx) -> dict:
    """Append ``_s{idx}`` to every key of a per-substrate result dict.

    idx None -> returned unchanged (single-substrate / off mode).
    """
    if idx is None:
        return result
    return {f"{k}_s{idx}": v for k, v in result.items()}
```

Refactor the per-row body of `__execute` (lines 466-569) so the substrate
analysis is a closure computing ONE substrate's result dict, then call it once
per substrate. Replace the block from `default_result = {...}` (line 466)
through `results.append(row_result)` (line 569) with:

```python
            default_result = {
                'distance_ligand_to_cofactor': None,
                'distance_ligand_to_closest_nuc': None,
                'ligand_moiety_method': None,
                'cofactor_moiety_method': None,
            }

            def _analyze(sub_smiles, sub_moiety):
                r = {}
                try:
                    chain_ids = get_hetatm_chain_ids(pdb_file)
                    ligands = [extract_chain_as_rdkit_mol(pdb_file, cid, sanitize=False)
                               for cid in chain_ids]
                    ligand_candidate = closest_ligands_by_element_composition(
                        ligands, sub_smiles, top_k=1)
                    ligand_mol = as_mol(ligand_candidate[0]) if ligand_candidate else None
                    ligand_mol = assign_bond_orders_from_smiles(ligand_mol, sub_smiles)
                    ligand_mol = ensure_3d(ligand_mol)
                    ligand_centroid, lig_method, _ = moiety_centroid_with_fallbacks(
                        ligand_mol, sub_moiety, 'ligand',
                        grow_mcs_by_one_bond=True, use_chirality=False)
                    r["ligand_moiety_method"] = lig_method

                    if ('cofactor_smiles' in df.columns and cofactor_smiles is not None
                            and tool != 'vina'):
                        cofactor_candidate = closest_ligands_by_element_composition(
                            ligands, cofactor_smiles, top_k=1)
                        cofactor_mol = as_mol(cofactor_candidate[0]) if cofactor_candidate else None
                        cofactor_mol = assign_bond_orders_from_smiles(cofactor_mol, cofactor_smiles)
                        cofactor_mol = ensure_3d(cofactor_mol)
                        cofactor_centroid, cof_method, _ = moiety_centroid_with_fallbacks(
                            cofactor_mol, cofactor_moiety, 'cofactor',
                            grow_mcs_by_one_bond=True, use_chirality=False)
                        r["cofactor_moiety_method"] = cof_method
                        d = nearest_centroid_distance(ligand_centroid, cofactor_centroid)
                        if d:
                            r['distance_ligand_to_cofactor'] = d

                    all_nucs = get_all_nucs_atom_coords(pdb_file)
                    closest = find_min_distance(ligand_centroid, all_nucs)
                    if closest:
                        r['distance_ligand_to_closest_nuc'] = {
                            closest['nuc_res']: closest['distance']}
                except Exception as e:
                    logger.error(f"Error processing {entry_name}: {e}")
                    r.update(default_result)
                return r

            subs = iter_substrates(row)
            row_result = {}
            if len(subs) <= 1:
                row_result.update(_analyze(substrate_smiles, substrate_moiety))
            else:
                for i, (s_smiles, _s_name, s_moiety) in enumerate(subs):
                    row_result.update(_suffix_keys(_analyze(s_smiles, s_moiety), i))
            results.append(row_result)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest -p no:cacheprovider tests/test_geometric_filter_multi_substrate.py -v`
Expected: both `_suffix_keys` tests PASS.

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `pytest -p no:cacheprovider`
Expected: 226 passed, 3 skipped (224 from Task 8b + 2 new). Existing geometric
filter tests unchanged (single-substrate rows hit the `len(subs) <= 1` branch
producing identical unsuffixed keys).

- [ ] **Step 6: Commit**

```bash
git add structurezyme/steps/geometric_filtering_cofactor_MCS.py tests/test_geometric_filter_multi_substrate.py
git -c core.fsync=none commit -m "feat: per-substrate geometric filter columns in together mode"
```

### Task 9b: PLIP per-substrate `_s{i}` columns

`plip_step.py` analyzes ONE ligand per row: it picks the chain closest to
`substrate_smiles` via `select_ligand_from_smiles_via_composition`
(plip_step.py:20-76) and records 8 `plip_*` interaction counts. In `together`
mode, loop over substrates and emit each substrate's counts suffixed `_s{i}`.
The expensive `prot.analyze()` runs ONCE per PDB; only ligand selection +
interaction lookup repeat per substrate.

**Files:**
- Modify: `structurezyme/steps/plip_step.py:94-165` (`__execute` per-row body)
- Test: `tests/test_plip_multi_substrate.py` (create)

**Interfaces:**
- Consumes: `iter_substrates` (Task 2); `_suffix_keys` (Task 9, imported from
  `structurezyme.steps.geometric_filtering_cofactor_MCS`);
  `select_ligand_from_smiles_via_composition` (same module).
- Produces (together mode): keys `plip_hydrogen_nbonds_s{i}`,
  `plip_hydrophobic_contacts_s{i}`, `plip_salt_bridges_s{i}`,
  `plip_pi_stacking_s{i}`, `plip_pi_cation_s{i}`, `plip_halogen_bonds_s{i}`,
  `plip_water_bridges_s{i}`, `plip_metal_complexes_s{i}`. Single-substrate:
  unsuffixed (unchanged).

- [ ] **Step 1: Write the failing test**

Create `tests/test_plip_multi_substrate.py`:

```python
# tests/test_plip_multi_substrate.py
import pandas as pd
import structurezyme.steps.plip_step as plip_mod
from structurezyme.steps.plip_step import PLIP


class _FakeInteractions:
    hbonds_ldon = []; hbonds_pdon = []
    hydrophobic_contacts = []; saltbridge_pneg = []; saltbridge_lneg = []
    pistacking = []; pication_laro = []; pication_paro = []
    halogen_bonds = []; water_bridges = []; metal_complexes = []


def test_plip_together_emits_per_substrate_keys(tmp_path, monkeypatch):
    (tmp_path / "chai_0.pdb").write_text("dummy")
    df = pd.DataFrame({"Entry": ["P1"], "docked_structure": ["chai_0"],
                       "substrate_smiles": ["CCO.c1ccncc1"]})

    class _FakeComplex:
        interaction_sets = {"LIG:B:1": _FakeInteractions(),
                            "LIG:C:2": _FakeInteractions()}
        def load_pdb(self, p): pass
        def analyze(self): pass
    monkeypatch.setattr(plip_mod, "PDBComplex", _FakeComplex, raising=True)

    # return a different chain per substrate so both are analyzable
    calls = {"n": 0}
    def _fake_select(path, smiles):
        calls["n"] += 1
        return ("B", 1, "LIG") if calls["n"] == 1 else ("C", 2, "LIG")
    monkeypatch.setattr(plip_mod, "select_ligand_from_smiles_via_composition",
                        _fake_select, raising=True)

    step = PLIP(input_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
    out = step.execute(df)
    cols = set(out.columns)
    assert "plip_hydrogen_nbonds_s0" in cols
    assert "plip_hydrogen_nbonds_s1" in cols
    # single-substrate contract preserved: no unsuffixed key in together mode
    assert "plip_hydrogen_nbonds" not in cols
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest -p no:cacheprovider tests/test_plip_multi_substrate.py -v`
Expected: FAIL — only unsuffixed `plip_*` keys are produced today.

- [ ] **Step 3: Implement the per-substrate loop**

In `structurezyme/steps/plip_step.py`, add to the helpers import (line 13):

```python
from structurezyme.utils.helpers import get_hetatm_chain_ids, extract_chain_as_rdkit_mol, closest_ligands_by_element_composition
from structurezyme.utils.helpers import iter_substrates
from structurezyme.steps.geometric_filtering_cofactor_MCS import _suffix_keys
```

Refactor the per-row body (lines 111-163). Move `prot.analyze()` out of the
per-substrate work, extract the interaction counting into a closure keyed by
substrate SMILES, and loop:

```python
            try:
                default_result = {
                    'plip_hydrogen_nbonds': None,
                    'plip_hydrophobic_contacts': None,
                    'plip_salt_bridges': None,
                    'plip_pi_stacking': None,
                    'plip_pi_cation': None,
                    'plip_halogen_bonds': None,
                    'plip_water_bridges': None,
                    'plip_metal_complexes': None,
                }

                with suppress_stdout_stderr():
                    prot = PDBComplex()
                    prot.load_pdb(pdb_file_as_str)
                    prot.analyze()

                def _analyze(sub_smiles):
                    r = dict(default_result)
                    ligand = select_ligand_from_smiles_via_composition(
                        pdb_file_as_path, sub_smiles)
                    if not ligand:
                        return r
                    chain_id, resseq, resname = ligand
                    formatted_ligand_id = f"{resname}:{chain_id}:{resseq}"
                    interactions = prot.interaction_sets[formatted_ligand_id]
                    r['plip_hydrogen_nbonds'] = (
                        len(interactions.hbonds_ldon) + len(interactions.hbonds_pdon))
                    r['plip_hydrophobic_contacts'] = len(interactions.hydrophobic_contacts)
                    r['plip_salt_bridges'] = (
                        len(interactions.saltbridge_pneg) + len(interactions.saltbridge_lneg))
                    r['plip_pi_stacking'] = len(interactions.pistacking)
                    r['plip_pi_cation'] = (
                        len(interactions.pication_laro) + len(interactions.pication_paro))
                    r['plip_halogen_bonds'] = len(interactions.halogen_bonds)
                    r['plip_water_bridges'] = len(interactions.water_bridges)
                    r['plip_metal_complexes'] = len(interactions.metal_complexes)
                    return r

                subs = iter_substrates(row)
                if len(subs) <= 1:
                    row_result.update(_analyze(substrate_smiles))
                else:
                    for i, (s_smiles, _n, _m) in enumerate(subs):
                        row_result.update(_suffix_keys(_analyze(s_smiles), i))

            except Exception as e:
                logger.error(f"Error processing {entry_name}: {e}")
                row_result.update(default_result)

            results.append(row_result)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest -p no:cacheprovider tests/test_plip_multi_substrate.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `pytest -p no:cacheprovider`
Expected: +1 test over prior total, 3 skipped. Existing single-substrate plip
tests unchanged.

- [ ] **Step 6: Commit**

```bash
git add structurezyme/steps/plip_step.py tests/test_plip_multi_substrate.py
git -c core.fsync=none commit -m "feat: per-substrate plip columns in together mode"
```

### Task 9c: Ligand SASA per-substrate `_s{i}` columns

`ligandSASA_step.py` computes buried-SASA for ONE ligand per row (chain picked
via `select_ligand_from_smiles_via_composition`, ligandSASA_step.py:20-76). In
`together` mode, compute SASA for each substrate's chain and emit its 4 metrics
suffixed `_s{i}`.

**Files:**
- Modify: `structurezyme/steps/ligandSASA_step.py:93-161` (`__execute` per-row)
- Test: `tests/test_ligand_sasa_multi_substrate.py` (create)

**Interfaces:**
- Consumes: `iter_substrates` (Task 2); `_suffix_keys` (Task 9);
  `select_ligand_from_smiles_via_composition`, `SingleLigandSelect`, `freesasa`
  (same module).
- Produces (together mode): `sasa_ligand_in_complex_s{i}`,
  `sasa_ligand_alone_s{i}`, `buried_sasa_s{i}`, `percentage_buried_sasa_s{i}`.
  Single-substrate: unsuffixed (unchanged).

- [ ] **Step 1: Write the failing test**

Create `tests/test_ligand_sasa_multi_substrate.py`:

```python
# tests/test_ligand_sasa_multi_substrate.py
import pandas as pd
import structurezyme.steps.ligandSASA_step as sasa_mod
from structurezyme.steps.ligandSASA_step import LigandSASA


def test_sasa_together_emits_per_substrate_keys(tmp_path, monkeypatch):
    (tmp_path / "chai_0.pdb").write_text("dummy")
    df = pd.DataFrame({"Entry": ["P1"], "docked_structure": ["chai_0"],
                       "substrate_smiles": ["CCO.c1ccncc1"]})

    calls = {"n": 0}
    def _fake_select(path, smiles):
        calls["n"] += 1
        return ("B", 1, "LIG") if calls["n"] == 1 else ("C", 2, "LIG")
    monkeypatch.setattr(sasa_mod, "select_ligand_from_smiles_via_composition",
                        _fake_select, raising=True)

    # stub the freesasa machinery so no real SASA runs
    class _FakeStruct: pass
    monkeypatch.setattr(sasa_mod.freesasa, "Structure",
                        lambda *a, **k: _FakeStruct(), raising=True)
    class _FakeResult:
        def totalArea(self): return 100.0
    monkeypatch.setattr(sasa_mod.freesasa, "calc",
                        lambda s: _FakeResult(), raising=True)
    monkeypatch.setattr(sasa_mod.freesasa, "selectArea",
                        lambda sel, s, r: {"ligand": 40.0}, raising=True)
    # stub the Bio.PDB save path
    monkeypatch.setattr(sasa_mod, "PDBParser",
                        lambda QUIET=True: type("P", (), {
                            "get_structure": lambda self, n, p: {0: object()}})(),
                        raising=True)
    monkeypatch.setattr(sasa_mod, "PDBIO",
                        lambda: type("IO", (), {
                            "set_structure": lambda self, s: None,
                            "save": lambda self, p, select=None: open(p, "w").close(),
                        })(), raising=True)

    step = LigandSASA(input_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
    out = step.execute(df)
    cols = set(out.columns)
    assert "buried_sasa_s0" in cols
    assert "buried_sasa_s1" in cols
    assert "buried_sasa" not in cols
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest -p no:cacheprovider tests/test_ligand_sasa_multi_substrate.py -v`
Expected: FAIL — only unsuffixed SASA keys produced today.

- [ ] **Step 3: Implement the per-substrate loop**

In `structurezyme/steps/ligandSASA_step.py`, add to imports (after line 15):

```python
from structurezyme.utils.helpers import iter_substrates
from structurezyme.steps.geometric_filtering_cofactor_MCS import _suffix_keys
```

Refactor the per-row body (lines 116-159) into a `_analyze(sub_smiles)` closure
returning the 4-key dict, then loop:

```python
            def _analyze(sub_smiles):
                r = dict(default_result)
                ligand = select_ligand_from_smiles_via_composition(
                    pdb_file, sub_smiles)
                if not ligand:
                    return r
                chain_id, resseq, resname = ligand
                with TemporaryDirectory() as tmpdir:
                    ligand_path = Path(tmpdir) / "ligand.pdb"
                    io = PDBIO()
                    structure = PDBParser(QUIET=True).get_structure("s", str(pdb_file))[0]
                    io.set_structure(structure)
                    io.save(str(ligand_path),
                            select=SingleLigandSelect(chain_id, resseq, resname))
                    structure_complex = freesasa.Structure(str(pdb_file), options={'hetatm': True})
                    structure_ligand = freesasa.Structure(str(ligand_path), options={'hetatm': True})
                result_ligand = freesasa.calc(structure_ligand)
                result_complex = freesasa.calc(structure_complex)
                selection = [f"ligand, chain {chain_id} and resn {resname} and resi {resseq}"]
                sasa_in_complex = freesasa.selectArea(selection, structure_complex, result_complex)
                sasa_alone = result_ligand.totalArea()
                buried = sasa_alone - sasa_in_complex["ligand"]
                pct = (buried / sasa_alone) * 100 if sasa_alone > 0 else 0.0
                r['sasa_ligand_in_complex'] = sasa_in_complex["ligand"]
                r['sasa_ligand_alone'] = sasa_alone
                r['buried_sasa'] = buried
                r['percentage_buried_sasa'] = pct
                return r

            try:
                subs = iter_substrates(row)
                if len(subs) <= 1:
                    row_result.update(_analyze(substrate_smiles))
                else:
                    for i, (s_smiles, _n, _m) in enumerate(subs):
                        row_result.update(_suffix_keys(_analyze(s_smiles), i))
            except Exception as e:
                logger.error(f"Error processing {entry_name}: {e}")
                row_result.update(default_result)

            results.append(row_result)
```

(Note: `default_result` is already defined above at lines 101-105; keep it
where it is so the closure can close over it.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest -p no:cacheprovider tests/test_ligand_sasa_multi_substrate.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `pytest -p no:cacheprovider`
Expected: +1 test over prior total, 3 skipped.

- [ ] **Step 6: Commit**

```bash
git add structurezyme/steps/ligandSASA_step.py tests/test_ligand_sasa_multi_substrate.py
git -c core.fsync=none commit -m "feat: per-substrate ligand SASA columns in together mode"
```

### Task 9d: fpocket per-substrate `_s{i}` columns

`fpocket_step.py._process_single_row_with_fpocket` (lines 239-334) runs fpocket
ONCE, steering the `-r` pocket hint to the single substrate's chain via
`fpocket_r_from_smiles_via_composition(pdb_path, substrate_smiles)` (line 271),
then parses pocket features + SASA into a `pd.Series`. In `together` mode we run
fpocket once PER substrate (each with its own `-r` hint and its own output dir),
suffixing the parsed feature keys `_s{i}`. This step returns a `pd.Series` per
row (not a dict), so the suffixing merges Series.

**Files:**
- Modify: `structurezyme/steps/fpocket_step.py:239-334`
  (`_process_single_row_with_fpocket`)
- Test: `tests/test_fpocket_multi_substrate.py` (create)

**Interfaces:**
- Consumes: `iter_substrates` (Task 2); `_suffix_keys` (Task 9);
  `fpocket_r_from_smiles_via_composition`, `extract_fpocket_features`,
  `extract_SASA` (same module).
- Produces (together mode): each substrate's pocket-feature + SASA keys suffixed
  `_s{i}`, plus `ASvolume_dir_s{i}`. Single-substrate: unsuffixed (unchanged).

- [ ] **Step 1: Confirm the class name**

Open `structurezyme/steps/fpocket_step.py` and note the Step subclass name
(the `class ...(Step)` that defines `_process_single_row_with_fpocket`). Use
that exact name in the test import and `monkeypatch.setattr` target below
(shown as `Fpocket`).

- [ ] **Step 2: Write the failing test**

Create `tests/test_fpocket_multi_substrate.py`:

```python
# tests/test_fpocket_multi_substrate.py
import pandas as pd
import structurezyme.steps.fpocket_step as fp_mod
from structurezyme.steps.fpocket_step import Fpocket  # class name per Step 1


def test_fpocket_together_emits_per_substrate_keys(tmp_path, monkeypatch):
    prepared = tmp_path / "prep"; prepared.mkdir()
    (prepared / "chai_0.pdb").write_text("dummy")
    out = tmp_path / "out"; out.mkdir()
    df = pd.DataFrame({"Entry": ["P1"], "docked_structure": ["chai_0"],
                       "substrate_smiles": ["CCO.c1ccncc1"]})

    # stub the single-substrate worker to a deterministic feature Series so the
    # test targets ONLY the per-substrate suffixing/merge logic
    def _fake_single(self, row, sub_smiles):
        return pd.Series({"ASvolume_dir": "/x",
                          "pocket_volume": 123.0,
                          "pocket_sasa": 45.0})
    monkeypatch.setattr(fp_mod.Fpocket, "_fpocket_one_substrate",
                        _fake_single, raising=False)

    step = fp_mod.Fpocket(preparedfiles_dir=str(prepared), output_dir=str(out),
                          num_threads=1)
    res = step.execute(df)
    cols = set(res.columns)
    assert "pocket_volume_s0" in cols
    assert "pocket_volume_s1" in cols
    assert "pocket_volume" not in cols
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest -p no:cacheprovider tests/test_fpocket_multi_substrate.py -v`
Expected: FAIL — `_fpocket_one_substrate` does not exist and no `_s{i}` keys.

- [ ] **Step 4: Extract a per-substrate worker and loop over substrates**

In `structurezyme/steps/fpocket_step.py`, add imports (near the top, with the
other `from structurezyme...` imports) and `import hashlib`:

```python
import hashlib
from structurezyme.utils.helpers import iter_substrates
from structurezyme.steps.geometric_filtering_cofactor_MCS import _suffix_keys
```

Rename the existing per-row worker's core to take an explicit substrate SMILES.
Rename `_process_single_row_with_fpocket` (line 239) to
`_fpocket_one_substrate(self, row, sub_smiles)`, and inside it:
- replace `substrate_smiles = row.get("substrate_smiles")` (line 242) with
  `substrate_smiles = sub_smiles`;
- make `final_out_dir` unique per substrate (line 295):
  ```python
  sub_tag = hashlib.md5(str(sub_smiles).encode()).hexdigest()[:6]
  final_out_dir = self.output_dir / f"{pdb_file_path.stem}_{sub_tag}_fpocket_output"
  ```

Then add a new dispatcher with the ORIGINAL method name so `__execute`
(lines 337-354) keeps calling it unchanged:

```python
    def _process_single_row_with_fpocket(self, row: pd.Series) -> pd.Series:
        subs = iter_substrates(row)
        if len(subs) <= 1:
            return self._fpocket_one_substrate(
                row, subs[0][0] if subs else row.get("substrate_smiles"))
        merged = {}
        for i, (s_smiles, _n, _m) in enumerate(subs):
            s = self._fpocket_one_substrate(row, s_smiles)
            merged.update(_suffix_keys(dict(s), i))
        return pd.Series(merged)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest -p no:cacheprovider tests/test_fpocket_multi_substrate.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `pytest -p no:cacheprovider`
Expected: +1 test over prior total, 3 skipped. Existing single-substrate
fpocket tests unchanged (the dispatcher's `len(subs) <= 1` branch calls the same
worker with the same substrate SMILES as before).

- [ ] **Step 7: Commit**

```bash
git add structurezyme/steps/fpocket_step.py tests/test_fpocket_multi_substrate.py
git -c core.fsync=none commit -m "feat: per-substrate fpocket columns in together mode"
```

### Task 10: CLI init template exposes `multi_substrate_mode`

Ensure `structurezyme init` writes a config that includes `multi_substrate_mode`
(default `"off"`) so users discover the flag. `RunConfig.model_dump()` already
serializes the field (Task 1), so `_template().write(...)` includes it
automatically — this task pins that with a test and adds a one-line docs note.

**Files:**
- Modify: `structurezyme/cli.py:14-19` (only if the template needs an explicit
  value; verify first — likely no code change needed)
- Test: `tests/test_cli_init_multi_substrate.py` (create)

**Interfaces:**
- Consumes: `RunConfig` (Task 1), `cmd_init`/`_template` (cli.py).
- Produces: a written config YAML containing `multi_substrate_mode: off`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_init_multi_substrate.py`:

```python
# tests/test_cli_init_multi_substrate.py
import yaml
from structurezyme.cli import _template


def test_init_template_includes_multi_substrate_mode(tmp_path):
    out = tmp_path / "run.yml"
    _template().write(out)
    data = yaml.safe_load(out.read_text())
    assert data["multi_substrate_mode"] == "off"
```

- [ ] **Step 2: Run the test**

Run: `pytest -p no:cacheprovider tests/test_cli_init_multi_substrate.py -v`
Expected: PASS if `model_dump()` already serializes the field (most likely). If
it FAILS (field absent), set it explicitly in `_template()`:

```python
def _template() -> RunConfig:
    return RunConfig(paths=PathsConfig(output_root="", boltz_cache_dir=""),
                     multi_substrate_mode="off")
```

Then re-run — Expected: PASS.

- [ ] **Step 3: Run the full suite**

Run: `pytest -p no:cacheprovider`
Expected: +1 test over the prior total (233 passed, 3 skipped assuming 9b-9d
landed), 3 skipped.

- [ ] **Step 4: Commit**

```bash
git add structurezyme/cli.py tests/test_cli_init_multi_substrate.py
git -c core.fsync=none commit -m "test: init template exposes multi_substrate_mode"
```

---

## Self-Review Addenda

The following tasks were added after the spec-vs-plan self-review to close
coverage gaps.

### Task 3b: Fail-fast validation of malformed multi-substrate input

Spec (Error handling, lines 152-153) requires seeding to fail fast with an
`Entry`-named `ValueError` on malformed multi-substrate input: parallel-list
length mismatch between `substrate_smiles` (`.`-count) and
`substrate_name`/`substrate_moiety` (`|`-count) when those columns are present
and non-empty, or an empty substrate fragment. `iter_substrates` (Task 2)
intentionally pads/truncates for downstream robustness, so validation is a
SEPARATE gate applied in `separate`/`together` modes only.

**Files:**
- Modify: `structurezyme/runner.py` (add `_validate_multi_substrate_row` +
  call it inside `_expand_substrates` before expanding)
- Test: `tests/test_expand_substrates.py` (append)

**Interfaces:**
- Consumes: nothing new (plain string parsing).
- Produces: `_validate_multi_substrate_row(row) -> None` raising `ValueError`
  naming the offending `Entry`. Called by `_expand_substrates` for every row
  when `mode != "off"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_expand_substrates.py`:

```python
import pytest


def test_separate_length_mismatch_raises():
    df = pd.DataFrame({"Entry": ["P1"], "Sequence": ["M"],
                       "substrate_smiles": ["CCO.O"],
                       "substrate_name": ["only_one"],  # 1 name for 2 smiles
                       "substrate_moiety": ["CO|O"]})
    with pytest.raises(ValueError, match="P1"):
        _expand_substrates(df, "separate")


def test_empty_fragment_raises():
    df = pd.DataFrame({"Entry": ["P2"], "Sequence": ["M"],
                       "substrate_smiles": ["CCO..O"]})  # empty middle fragment
    with pytest.raises(ValueError, match="P2"):
        _expand_substrates(df, "together")


def test_off_mode_does_not_validate():
    # off mode must never raise even on ragged parallel lists
    df = pd.DataFrame({"Entry": ["P3"], "substrate_smiles": ["CCO.O"],
                       "substrate_name": ["only_one"]})
    _expand_substrates(df, "off")  # no exception
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest -p no:cacheprovider tests/test_expand_substrates.py -k "mismatch or empty_fragment or does_not_validate" -v`
Expected: FAIL — no validation yet (`separate` pads silently; `together` builds
without raising).

- [ ] **Step 3: Implement the validator and call it**

In `structurezyme/runner.py`, add above `_expand_substrates`:

```python
def _validate_multi_substrate_row(row) -> None:
    """Raise a ValueError (naming Entry) if a row's packed substrate columns
    are malformed: an empty SMILES fragment, or a present non-empty
    name/moiety parallel list whose length != the substrate count."""
    entry = row["Entry"]
    raw = str(row["substrate_smiles"])
    frags = raw.split(".")
    if any(f.strip() == "" for f in frags):
        raise ValueError(
            f"Malformed substrate_smiles for Entry={entry!r}: empty fragment "
            f"in {raw!r}"
        )
    n = len(frags)
    for col in ("substrate_name", "substrate_moiety"):
        if col in row and row[col] is not None and str(row[col]).strip() != "":
            parts = str(row[col]).split("|")
            if len(parts) != n:
                raise ValueError(
                    f"Malformed {col} for Entry={entry!r}: {len(parts)} "
                    f"value(s) for {n} substrate(s)"
                )
```

Then in `_expand_substrates`, at the very top of the function (before the
`if mode != "separate":` branch), add:

```python
    if mode != "off":
        for _, row in df.iterrows():
            _validate_multi_substrate_row(row)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest -p no:cacheprovider tests/test_expand_substrates.py -v`
Expected: all expand-substrate tests PASS (original 4 + Task 4's 2 seed tests
elsewhere + these 3).

- [ ] **Step 5: Run the full suite**

Run: `pytest -p no:cacheprovider`
Expected: +3 over prior total, 3 skipped.

- [ ] **Step 6: Commit**

```bash
git add structurezyme/runner.py tests/test_expand_substrates.py
git -c core.fsync=none commit -m "feat: fail fast on malformed multi-substrate input at seeding"
```

## Self-Review Result

**Spec coverage:** every spec section maps to a task —
config+cli (Tasks 1, 10), expansion+seeding+resume (Tasks 3, 4),
`iter_substrates` (Task 2), fail-fast validation (Task 3b, added in review),
Chai/Boltz/Vina routing (Tasks 5, 6, 7), `ligand_rmsd` per-substrate+joint
(Tasks 8, 8b), analysis `_s{i}` columns (Task 9 + 9b-9d), `enzyme_id` all
modes (Task 3).

**Type/name consistency:** verified consistent across tasks —
`iter_substrates(row) -> list[tuple[str,str,str]]`, `_expand_substrates(df, mode)`,
`_validate_multi_substrate_row(row)`, `_boltz_together_frame(df)`,
`_together_skips_vina(mode, df)`, `_match_substrate_chains(ligands, smiles_list)`,
`_suffix_keys(result, idx)`.

**Resolved:** Tasks **9b (plip), 9c (ligand_sasa), 9d (fpocket)** are now fully
fleshed out with exact file/line references, complete test code, and complete
implementation edits — no remaining by-reference placeholders. The one residual
verification the executor must do is confirm the fpocket Step subclass name
(Task 9d Step 1); everything else is concrete.

## Execution Handoff

Plan complete and saved to
`docs/superpowers/plans/2026-08-02-multi-substrate.md`.
