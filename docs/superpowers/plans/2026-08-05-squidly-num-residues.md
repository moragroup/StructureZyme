# squidly_num_residues (Top-N Catalytic Residue Selection) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `squidly_num_residues` config key that makes squidly return the N highest-mean-probability catalytic residues, ignoring the mean/variance thresholds.

**Architecture:** Applied entirely in the StructureZyme wrapper `Squidly` (`structurezyme/steps/squidly_step.py`), where the per-residue `mean` array is already surfaced. A new pure helper ranks residues by `mean` and returns the top N as a `|`-joined 0-indexed string. `run_squidly` forwards the config key into the `Squidly` constructor. No upstream squidly change is needed.

**Tech Stack:** Python 3.11, pandas, numpy, pytest.

## Global Constraints

- Selection when `num_residues` is set: pure top-N by `mean[i]` descending; **variance is ignored**.
- Ties broken deterministically by lower residue index (stable `np.argsort` ascending-index order).
- If N exceeds the number of scored positions (length of the `mean` array), return all available positions.
- `num_residues`, when set, must be a positive integer; reject zero, negative, and non-int values with `ValueError`.
- Opt-in: `num_residues is None` → behavior is byte-for-byte identical to today (threshold path unchanged).
- When `num_residues` is set, it wins over `mean_prob` / `mean_var` (thresholds ignored). If both a count and either threshold are set, emit a one-line warning via the module `logger`.
- Output column unchanged: `Squidly_CR_Position` is a `|`-joined string of 0-indexed positions.
- Config key naming follows the `squidly_*` snake_case convention.

---

### Task 1: Top-N selection helper

**Files:**
- Modify: `structurezyme/steps/squidly_step.py` (add helper after `_select_residues_from_ensemble`, which ends at line 52)
- Test: `tests/test_squidly_top_n_helper.py` (create)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `_select_top_n_from_ensemble(mean, num_residues: int) -> str` — takes a list-of-floats or numpy array `mean` and a positive int `num_residues`; returns a `|`-joined string of 0-indexed positions of the `num_residues` highest values, ordered ascending by index; returns `""` when `mean` is None/empty.

- [ ] **Step 1: Write the failing test**

Create `tests/test_squidly_top_n_helper.py`:

```python
"""Unit tests for _select_top_n_from_ensemble (pure top-N by mean)."""
from __future__ import annotations

import numpy as np

from structurezyme.steps.squidly_step import _select_top_n_from_ensemble


def test_picks_n_highest_mean_ordered_by_index():
    mean = [0.01, 0.90, 0.02, 0.80, 0.70]
    # top-3 by value are indices 1(0.90), 3(0.80), 4(0.70); output ascending index.
    assert _select_top_n_from_ensemble(mean, 3) == "1|3|4"


def test_n_larger_than_length_returns_all():
    mean = [0.2, 0.5, 0.1]
    assert _select_top_n_from_ensemble(mean, 10) == "0|1|2"


def test_ties_broken_by_lower_index():
    mean = [0.5, 0.5, 0.5, 0.1]
    # three-way tie at 0.5; pick the two lowest indices.
    assert _select_top_n_from_ensemble(mean, 2) == "0|1"


def test_accepts_numpy_array():
    mean = np.array([0.1, 0.9, 0.4])
    assert _select_top_n_from_ensemble(mean, 1) == "1"


def test_empty_or_none_returns_empty_string():
    assert _select_top_n_from_ensemble(None, 3) == ""
    assert _select_top_n_from_ensemble([], 3) == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_squidly_top_n_helper.py -v`
Expected: FAIL with `ImportError: cannot import name '_select_top_n_from_ensemble'`.

- [ ] **Step 3: Write minimal implementation**

In `structurezyme/steps/squidly_step.py`, immediately after line 52 (the end of `_select_residues_from_ensemble`, before `_normalize_residues`), add:

```python
def _select_top_n_from_ensemble(mean, num_residues: int) -> str:
    """Select the ``num_residues`` positions with the highest ensemble mean.

    Pure top-N by mean probability, ignoring variance. Ties are broken by
    lower residue index. If ``num_residues`` exceeds the number of scored
    positions, all positions are returned. Returns a '|'-joined string of
    0-indexed positions, ordered ascending by index (matching the threshold
    path's output format).
    """
    import numpy as _np
    if mean is None:
        return ""
    m = _np.asarray(mean, dtype=float)
    if m.size == 0:
        return ""
    n = int(min(num_residues, m.size))
    if n <= 0:
        return ""
    # argsort ascending is stable, so equal values keep ascending index order;
    # take the last n (highest values), then sort those indices ascending.
    order = _np.argsort(m, kind="stable")
    picks = sorted(int(i) for i in order[-n:])
    return "|".join(str(p) for p in picks)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_squidly_top_n_helper.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add structurezyme/steps/squidly_step.py tests/test_squidly_top_n_helper.py
git commit -m "feat(squidly): add _select_top_n_from_ensemble helper"
```

---

### Task 2: Constructor param + validation on `Squidly`

**Files:**
- Modify: `structurezyme/steps/squidly_step.py:97-128` (the `__init__`)
- Test: `tests/test_squidly_num_residues_init.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `Squidly(..., num_residues: int | None = None)` constructor param stored as `self.num_residues`; raises `ValueError` when `num_residues` is set to a non-positive or non-integer value.

- [ ] **Step 1: Write the failing test**

Create `tests/test_squidly_num_residues_init.py`:

```python
"""Squidly.__init__ accepts and validates num_residues."""
from __future__ import annotations

import pytest

from structurezyme.steps.squidly_step import Squidly


def test_default_num_residues_is_none():
    assert Squidly().num_residues is None


def test_positive_int_is_stored():
    assert Squidly(num_residues=3).num_residues == 3


@pytest.mark.parametrize("bad", [0, -1, 2.5, "3"])
def test_invalid_num_residues_raises(bad):
    with pytest.raises(ValueError):
        Squidly(num_residues=bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_squidly_num_residues_init.py -v`
Expected: FAIL — `test_positive_int_is_stored` errors with `TypeError: __init__() got an unexpected keyword argument 'num_residues'`.

- [ ] **Step 3: Write minimal implementation**

In `structurezyme/steps/squidly_step.py`, add the parameter to `__init__`. Change the signature line `mean_var: float | None = None,` (line 104) to insert the new param right after it:

```python
        mean_var: float | None = None,
        num_residues: int | None = None,
```

Then, after the `self.mean_var = mean_var` assignment (line 122), add validation and storage:

```python
        self.mean_var = mean_var
        if num_residues is not None:
            if isinstance(num_residues, bool) or not isinstance(num_residues, int):
                raise ValueError(
                    f"num_residues must be a positive integer, got {num_residues!r}"
                )
            if num_residues < 1:
                raise ValueError(
                    f"num_residues must be >= 1, got {num_residues!r}"
                )
        self.num_residues = num_residues
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_squidly_num_residues_init.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add structurezyme/steps/squidly_step.py tests/test_squidly_num_residues_init.py
git commit -m "feat(squidly): add validated num_residues constructor param"
```

---

### Task 3: Wire top-N into `Squidly.execute`

**Files:**
- Modify: `structurezyme/steps/squidly_step.py:266-273` (the `TODO(squidly-upstream)` selection block)
- Test: `tests/test_squidly_top_n_execute.py` (create)

**Interfaces:**
- Consumes: `_select_top_n_from_ensemble` (Task 1); `self.num_residues` (Task 2).
- Produces: `Squidly.execute` uses top-N selection when `self.num_residues` is set, overriding the threshold path; emits a `logger.warning` when both `num_residues` and a threshold are set.

- [ ] **Step 1: Write the failing test**

Create `tests/test_squidly_top_n_execute.py`. It reuses the fake-upstream pattern from `tests/test_squidly_threshold_local_apply.py`:

```python
"""Squidly.execute selects top-N by mean when num_residues is set."""
from __future__ import annotations

import types

import numpy as np
import pandas as pd


class _FakeActiveSitePred:
    def __init__(self, **kwargs):
        self._args = kwargs.get("args") or []

    def execute(self, df):
        out = df.copy()
        means, variances = [], []
        for _entry, seq in df[["Entry", "Sequence"]].values:
            n = len(seq)
            m = np.full(n, 0.001, dtype=float)
            v = np.full(n, 1e-6, dtype=float)
            # Four candidate residues with distinct means; high variance on 200.
            m[10] = 0.30
            m[78] = 0.50
            m[120] = 0.40
            m[200] = 0.99
            v[200] = 5.0  # would fail any variance gate; top-N must ignore this.
            means.append(m.tolist())
            variances.append(v.tolist())
        out["Squidly_Ensemble_Residues"] = [""] * len(out)
        out["mean"] = means
        out["variance"] = variances
        out["entropy"] = [[0.0] * len(s) for s in out["Sequence"]]
        out = out.rename(columns={"Entry": "label"})
        return out


def _make_df():
    seq = "M" + "A" * 400
    return pd.DataFrame({"Entry": ["E1"], "Sequence": [seq]})


def _patch(monkeypatch):
    import structurezyme.steps.squidly_step as sqmod
    monkeypatch.setattr(sqmod.shutil, "which", lambda name: "/fake/squidly")
    fake_module = types.ModuleType("enzymetk.predict_catalyticsite_step")
    fake_module.ActiveSitePred = _FakeActiveSitePred
    monkeypatch.setitem(
        __import__("sys").modules,
        "enzymetk.predict_catalyticsite_step",
        fake_module,
    )


def test_top_n_ignores_variance_and_returns_n(monkeypatch):
    from structurezyme.steps.squidly_step import Squidly

    _patch(monkeypatch)
    step = Squidly(num_residues=3)
    out = step.execute(_make_df())
    # highest means: 200(0.99),78(0.50),120(0.40); output ascending index.
    assert list(out["Squidly_CR_Position"]) == ["78|120|200"]


def test_num_residues_overrides_thresholds_with_warning(monkeypatch, caplog):
    import logging
    from structurezyme.steps.squidly_step import Squidly

    _patch(monkeypatch)
    step = Squidly(num_residues=1, mean_prob=0.03, mean_var=0.5)
    with caplog.at_level(logging.WARNING):
        out = step.execute(_make_df())
    # top-1 by mean is index 200 (0.99), even though its variance fails a gate.
    assert list(out["Squidly_CR_Position"]) == ["200"]
    assert any("num_residues" in r.message for r in caplog.records)


def test_num_residues_none_uses_threshold_path(monkeypatch):
    from structurezyme.steps.squidly_step import Squidly

    _patch(monkeypatch)
    # No num_residues; permissive thresholds -> threshold path selects by gate.
    step = Squidly(mean_prob=0.03, mean_var=0.1)
    out = step.execute(_make_df())
    # index 200 has variance 5.0 (> 0.1) so it is excluded; others pass.
    assert list(out["Squidly_CR_Position"]) == ["10|78|120"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_squidly_top_n_execute.py -v`
Expected: FAIL — `test_top_n_ignores_variance_and_returns_n` fails because `Squidly_CR_Position` is empty (top-N not yet wired).

- [ ] **Step 3: Write minimal implementation**

In `structurezyme/steps/squidly_step.py`, replace the current selection block at lines 266-273:

```python
        if (self.mean_prob is not None or self.mean_var is not None) \
                and "mean" in df_pred.columns and "variance" in df_pred.columns:
            mp = self.mean_prob if self.mean_prob is not None else 0.6
            mv = self.mean_var if self.mean_var is not None else 0.225
            df_pred["Squidly_CR_Position"] = [
                _select_residues_from_ensemble(m, v, mp, mv)
                for m, v in zip(df_pred["mean"], df_pred["variance"])
            ]
```

with:

```python
        if self.num_residues is not None and "mean" in df_pred.columns:
            if self.mean_prob is not None or self.mean_var is not None:
                logger.warning(
                    "Both num_residues=%s and a mean_prob/mean_var threshold "
                    "were set for Squidly; num_residues wins and the "
                    "thresholds are ignored.",
                    self.num_residues,
                )
            df_pred["Squidly_CR_Position"] = [
                _select_top_n_from_ensemble(m, self.num_residues)
                for m in df_pred["mean"]
            ]
        elif (self.mean_prob is not None or self.mean_var is not None) \
                and "mean" in df_pred.columns and "variance" in df_pred.columns:
            mp = self.mean_prob if self.mean_prob is not None else 0.6
            mv = self.mean_var if self.mean_var is not None else 0.225
            df_pred["Squidly_CR_Position"] = [
                _select_residues_from_ensemble(m, v, mp, mv)
                for m, v in zip(df_pred["mean"], df_pred["variance"])
            ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_squidly_top_n_execute.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the existing squidly regression tests (no regressions)**

Run: `python -m pytest tests/test_squidly_threshold_local_apply.py tests/test_squidly_step.py -v`
Expected: PASS (all previously-passing tests still pass).

- [ ] **Step 6: Commit**

```bash
git add structurezyme/steps/squidly_step.py tests/test_squidly_top_n_execute.py
git commit -m "feat(squidly): select top-N residues by mean when num_residues set"
```

---

### Task 4: Forward `squidly_num_residues` config key through `run_squidly`

**Files:**
- Modify: `structurezyme/step_runners.py:184-196` (the `Squidly(...)` construction)
- Test: `tests/test_squidly_num_residues_wiring.py` (create)

**Interfaces:**
- Consumes: `Squidly` constructor `num_residues` param (Task 2).
- Produces: `run_squidly` reads `squidly_num_residues` from step options and passes it to `Squidly(num_residues=...)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_squidly_num_residues_wiring.py`. It reuses the runner-fake pattern from `tests/test_squidly_threshold_wiring.py`:

```python
"""run_squidly forwards squidly_num_residues into Squidly()."""
from __future__ import annotations

import types

import pandas as pd


class _FakeSquidly:
    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        _FakeSquidly.last_kwargs = kwargs

    def execute(self, df):
        out = df.copy()
        out["Squidly_CR_Position"] = "10|20|30"
        return out


class _FakeLayout:
    def __init__(self, root):
        self.root = root


class _FakeRuntime:
    num_threads = 1


class _FakeConfig:
    def __init__(self, opts):
        self._opts = opts
        self.runtime = _FakeRuntime()

    def step_options(self, name):
        return self._opts


class _FakeCtx:
    def __init__(self, tmp_path, opts, seed_df):
        self.layout = _FakeLayout(tmp_path)
        self.config = _FakeConfig(opts)
        self._seed = seed_df

    def checkpoint_path(self, name):
        return self.layout.root / f"{name}.pkl"


def _run(monkeypatch, tmp_path, opts):
    import structurezyme.step_runners as sr
    from structurezyme.steps import squidly_step

    monkeypatch.setattr(squidly_step, "Squidly", _FakeSquidly)
    seed = pd.DataFrame({
        "Entry": ["E1"],
        "Sequence": ["MKAT"],
        "substrate_smiles": ["CCO"],
    })
    seed.to_pickle(tmp_path / "_input.pkl")
    ctx = _FakeCtx(tmp_path, opts, seed)
    spec = types.SimpleNamespace(inputs=[])
    sr.run_squidly(ctx, spec)
    return _FakeSquidly.last_kwargs


def test_forwards_num_residues(monkeypatch, tmp_path):
    kwargs = _run(monkeypatch, tmp_path,
                  {"enabled": True, "squidly_num_residues": 3})
    assert kwargs.get("num_residues") == 3


def test_num_residues_defaults_none(monkeypatch, tmp_path):
    kwargs = _run(monkeypatch, tmp_path, {"enabled": True})
    assert kwargs.get("num_residues") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_squidly_num_residues_wiring.py -v`
Expected: FAIL — `test_forwards_num_residues` asserts `None == 3` (kwarg not passed yet).

- [ ] **Step 3: Write minimal implementation**

In `structurezyme/step_runners.py`, in the `Squidly(...)` call, add the new kwarg after the `mean_var=` line (line 194) and before `num_threads=`:

```python
        mean_var=opts.get("squidly_mean_var", None),
        num_residues=opts.get("squidly_num_residues", None),
        num_threads=opts.get("squidly_num_threads") or ctx.config.runtime.num_threads,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_squidly_num_residues_wiring.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full squidly test set (no regressions)**

Run: `python -m pytest tests/ -k squidly -v`
Expected: PASS (all squidly tests pass).

- [ ] **Step 6: Commit**

```bash
git add structurezyme/step_runners.py tests/test_squidly_num_residues_wiring.py
git commit -m "feat(squidly): forward squidly_num_residues config key into Squidly"
```

---

### Task 5: Document the config key

**Files:**
- Modify: `docs/configuration.md` (squidly step options section) — add `squidly_num_residues`

**Interfaces:**
- Consumes: nothing.
- Produces: user-facing documentation of the new key.

- [ ] **Step 1: Locate the squidly options table**

Run: `grep -n "squidly_num_threads" docs/configuration.md`
Expected: the `squidly_num_threads` row of the `**squidly**` options table (around line 88). The table has three columns: `Option | Default | Meaning`.

- [ ] **Step 2: Add the new row(s) to the squidly options table**

The existing table (around lines 83-88) omits `squidly_mean_prob` /
`squidly_mean_var`; add them plus the new key so the table is complete. Insert
these rows after the `squidly_as_threshold` row and before the
`squidly_num_threads` row:

```markdown
| `squidly_mean_prob` | `null` | Ensemble mean-probability threshold (upstream default 0.6). Ignored if `squidly_num_residues` is set. |
| `squidly_mean_var` | `null` | Ensemble variance cutoff (upstream default 0.225). Ignored if `squidly_num_residues` is set. |
| `squidly_num_residues` | `null` | If set, return exactly this many catalytic residues (the N highest-mean positions), ignoring the mean/variance thresholds. Useful when a very low threshold is needed to surface the right residue so vina always gets a docking site. A warning is logged if a threshold is also set. |
```

- [ ] **Step 3: Commit**

```bash
git add docs/configuration.md
git commit -m "docs: document squidly_num_residues (and mean_prob/mean_var) config keys"
```

---

## Self-Review

**Spec coverage:**
- Selection logic (top-N by mean, ignore variance) → Task 1 + Task 3.
- Ties by lower index → Task 1 (`test_ties_broken_by_lower_index`).
- N > scored positions → Task 1 (`test_n_larger_than_length_returns_all`).
- Positive-int validation → Task 2.
- Opt-in / unset = unchanged → Task 3 (`test_num_residues_none_uses_threshold_path`) + Task 4 (`test_num_residues_defaults_none`).
- Precedence + warning when both set → Task 3 (`test_num_residues_overrides_thresholds_with_warning`).
- Config key wiring → Task 4.
- Documentation → Task 5.

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows full code.

**Type consistency:** `_select_top_n_from_ensemble(mean, num_residues) -> str` and `num_residues: int | None` are used identically across Tasks 1-4.

**Note on Task 5:** `docs/configuration.md` line numbers are resolved at execution time via grep (Step 1) rather than hard-coded, since that file is not otherwise touched by this plan.
