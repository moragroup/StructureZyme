# Energy-aware best-pose selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the best-docked-pose selector fuse inter-tool geometric consensus with Rosetta `fastrelax_score`, so an energy-superior pose can win, for any run and any engine combination.

**Architecture:** Add a `fused_rank` selection method to `select_best_docked_structures` (energy passed in from `LigandRMSD.__execute`), and make PLACER's `_select_one_per_entry` treat `fused_rank` as authoritative. Energy dominates when geometry is degenerate; pure-geometry fallback when energy is absent.

**Tech Stack:** Python 3.12, pandas, numpy, pytest.

## Global Constraints

- Tool-agnostic: chai/boltz/vina treated symmetrically; energy ranking is
  **ascending raw `fastrelax_score`** (lower = better) for all three engines.
- No regression when fastrelax is off: with no energy, `fused_rank` collapses
  to the `inter_tool_min_per_tool` geometric winner.
- Preserve the existing output schema of `select_best_docked_structures`:
  each emitted row is `{Entry, tool, best_structure, avg_ligandRMSD, method}`.
- `fused_rank` is emitted **exactly once per entry**, always (even with no
  energy).
- Determinism: all tie-breaks end in alphabetical `docked_structure`.
- `fastrelax_score` per-entry dict shape: `{"chai": {pose_key: score},
  "boltz": {...}, "vina": {...}}`; chai/boltz `pose_key` via
  `_pose_id_from_structure_name` (strips `_relaxed` then `_{tool}`); vina key
  is `int(stem_without_relaxed.split('_')[-2])`.
- Reuse `structurezyme/utils/helpers.py` helpers; do not duplicate pose-id
  logic.

---

### Task 1: Pose → energy lookup helper

Add a pure helper that, given a `docked_structure` name and an entry's
per-engine `fastrelax_score` dict, returns that pose's energy or `None`.

**Files:**
- Modify: `structurezyme/steps/computeligandRMSD_step.py` (add helper near the
  top-level functions, after `get_tool_from_structure_name` at line 78-91)
- Test: `tests/test_select_best_docked_structures.py` (new file)

**Interfaces:**
- Consumes: `get_tool_from_structure_name` (already in the module,
  line 78); `_pose_id_from_structure_name` from
  `structurezyme.utils.helpers` (line 224).
- Produces: `pose_energy(docked_structure: str, entry_scores: dict | None) ->
  float | None`. Returns `None` when `entry_scores` is falsy, the engine is
  unknown, or the pose key is absent.

- [ ] **Step 1: Write the failing test**

Create `tests/test_select_best_docked_structures.py`:

```python
"""Tests for the energy-aware best-pose selector in computeligandRMSD_step."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from structurezyme.steps.computeligandRMSD_step import (
    pose_energy,
    select_best_docked_structures,
)


def test_pose_energy_chai_lookup():
    scores = {"chai": {"P41365_0": -12.5}, "boltz": {}, "vina": {}}
    assert pose_energy("P41365_0_chai", scores) == -12.5


def test_pose_energy_strips_relaxed_suffix():
    scores = {"chai": {"P41365_0": -12.5}, "boltz": {}, "vina": {}}
    assert pose_energy("P41365_0_chai_relaxed", scores) == -12.5


def test_pose_energy_boltz_model_key():
    scores = {"chai": {}, "boltz": {"P41365_model_0": -9.0}, "vina": {}}
    assert pose_energy("P41365_model_0_boltz_relaxed", scores) == -9.0


def test_pose_energy_vina_integer_key():
    scores = {"chai": {}, "boltz": {}, "vina": {3: -7.2}}
    assert pose_energy("P41365_3_vina_relaxed", scores) == -7.2


def test_pose_energy_missing_returns_none():
    scores = {"chai": {"P41365_0": -12.5}, "boltz": {}, "vina": {}}
    assert pose_energy("P41365_9_chai", scores) is None


def test_pose_energy_none_scores_returns_none():
    assert pose_energy("P41365_0_chai", None) is None
    assert pose_energy("P41365_0_chai", {}) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_select_best_docked_structures.py -k pose_energy -v`
Expected: FAIL with `ImportError: cannot import name 'pose_energy'`.

- [ ] **Step 3: Write minimal implementation**

In `structurezyme/steps/computeligandRMSD_step.py`, add the import near the
other helper imports (the existing `from structurezyme.utils.helpers import (...)`
block at lines 25-32) — add `_pose_id_from_structure_name` to that import
list. Then add this function right after `get_tool_from_structure_name`
(after line 91):

```python
def pose_energy(docked_structure: str, entry_scores):
    """Return the fastrelax_score for a docked pose, or None if unavailable.

    `entry_scores` is one entry's per-engine dict:
    {"chai": {pose_key: score}, "boltz": {...}, "vina": {...}}.
    chai/boltz keys are `<Entry>_<pose_id>` (via _pose_id_from_structure_name);
    vina keys are the integer pose index int(stem.split('_')[-2]).
    """
    if not entry_scores:
        return None
    engine = get_tool_from_structure_name(docked_structure)
    scores = entry_scores.get(engine)
    if not scores:
        return None
    if engine == "vina":
        stem = docked_structure
        if stem.endswith("_relaxed"):
            stem = stem[: -len("_relaxed")]
        try:
            key = int(stem.split("_")[-2])
        except (ValueError, IndexError):
            return None
    else:
        key = _pose_id_from_structure_name(docked_structure)
    return scores.get(key)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_select_best_docked_structures.py -k pose_energy -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add structurezyme/steps/computeligandRMSD_step.py tests/test_select_best_docked_structures.py
git commit -m "feat(ligand_rmsd): add pose_energy lookup helper for fastrelax scores"
```

### Task 2: `fused_rank` selection in `select_best_docked_structures`

Add a new `fastrelax_score_by_entry` parameter and emit one `fused_rank` row
per entry. Keep the three existing method rows unchanged.

**Files:**
- Modify: `structurezyme/steps/computeligandRMSD_step.py` — function
  `select_best_docked_structures` (lines 159-281)
- Test: `tests/test_select_best_docked_structures.py`

**Interfaces:**
- Consumes: `pose_energy` (Task 1); the entry-local `closest_rmsd_scores`
  dict and `tool_to_structures` dict already built inside the function.
- Produces: `select_best_docked_structures(rmsd_df,
  fastrelax_score_by_entry=None) -> pd.DataFrame`. Adds rows with
  `method == "fused_rank"`, schema `{Entry, tool, best_structure,
  avg_ligandRMSD, method}`, one per entry.

- [ ] **Step 1: Add a fixture builder + failing test**

Append to `tests/test_select_best_docked_structures.py`. `_pairwise` builds
the pairwise `rmsd_df` shape the selector consumes (columns `Entry`,
`docked_structure1`, `docked_structure2`, `tool1`, `tool2`, `ligand_rmsd`):

```python
def _pairwise(entry, structures, rmsd_lookup):
    """Build a pairwise rmsd_df for one entry.

    structures: dict {name: tool}. rmsd_lookup: dict {frozenset({a,b}): rmsd}.
    """
    rows = []
    names = list(structures)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append({
                "Entry": entry,
                "docked_structure1": a,
                "docked_structure2": b,
                "tool1": structures[a],
                "tool2": structures[b],
                "ligand_rmsd": rmsd_lookup[frozenset({a, b})],
            })
    return pd.DataFrame(rows)


def test_fused_rank_energy_overrides_biased_geometry():
    # 3 chai poses clustered together + 1 boltz pose off to the side.
    # Geometry (min-per-tool) favors the chai pose nearest boltz, but the
    # boltz pose has clearly better (lower) energy and must win fused_rank.
    structs = {
        "E_0_chai": "chai", "E_1_chai": "chai", "E_2_chai": "chai",
        "E_0_boltz": "boltz",
    }
    rmsd = {
        frozenset({"E_0_chai", "E_1_chai"}): 0.5,
        frozenset({"E_0_chai", "E_2_chai"}): 0.5,
        frozenset({"E_1_chai", "E_2_chai"}): 0.5,
        frozenset({"E_0_chai", "E_0_boltz"}): 2.0,  # nearest chai to boltz
        frozenset({"E_1_chai", "E_0_boltz"}): 5.0,
        frozenset({"E_2_chai", "E_0_boltz"}): 5.0,
    }
    df = _pairwise("E", structs, rmsd)
    scores = {"E": {"chai": {"E_0": -5.0, "E_1": -5.0, "E_2": -5.0},
                    "boltz": {"E_0": -20.0}, "vina": {}}}
    out = select_best_docked_structures(df, scores)
    fused = out[out["method"] == "fused_rank"]
    assert len(fused) == 1
    assert fused.iloc[0]["best_structure"] == "E_0_boltz"
    assert fused.iloc[0]["tool"] == "boltz"
```

Note: this entry has only 2 tools and the minority tool (boltz) has a single
pose, so it hits the **degenerate-geometry** branch where energy dominates.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_select_best_docked_structures.py -k fused_rank_energy -v`
Expected: FAIL — `select_best_docked_structures()` takes 1 positional
argument but 2 were given (parameter not added yet).

- [ ] **Step 3a: Add the parameter and a module-level fusion helper**

In `structurezyme/steps/computeligandRMSD_step.py`, change the signature at
line 159 from:

```python
def select_best_docked_structures(rmsd_df: pd.DataFrame) -> pd.DataFrame:
```

to:

```python
def select_best_docked_structures(
    rmsd_df: pd.DataFrame,
    fastrelax_score_by_entry: dict | None = None,
) -> pd.DataFrame:
```

Update the docstring's method list to mention a 4th method:
`4. fused_rank: energy-aware fusion of inter_tool_min_per_tool geometry with
fastrelax_score (lower energy = better). Degrades to pure geometry when
energy is unavailable.`

Then add this module-level helper directly **above**
`select_best_docked_structures` (before line 159):

```python
def _dense_rank(score_by_key: dict) -> dict:
    """Ascending dense rank (1 = smallest). Ties share a rank.

    score_by_key: {key: numeric}. Returns {key: rank_int}.
    """
    order = sorted(set(score_by_key.values()))
    rank_of_value = {v: i + 1 for i, v in enumerate(order)}
    return {k: rank_of_value[v] for k, v in score_by_key.items()}
```

- [ ] **Step 3b: Insert the `fused_rank` block inside the per-entry loop**

Inside `select_best_docked_structures`, the per-entry loop already computes
`closest_rmsd_scores` (Method 2, the `inter_tool_min_per_tool` geometry score
per structure) and `tool_to_structures`. Insert the following block **after
the Method 3 (vina) block and before the loop iterates to the next entry**
(i.e. right before the final `return pd.DataFrame(best_structures)` moves out
of the loop — concretely, after line 279's Method-3 append, still inside the
`for entry, entry_df in rmsd_df.groupby("Entry")` loop):

```python
        # ----- Method 4: energy-aware rank fusion (fused_rank) -----
        entry_scores = (
            (fastrelax_score_by_entry or {}).get(entry)
            if fastrelax_score_by_entry else None
        )

        # energy per structure (None where unrelaxed / unavailable)
        energy_by_s = {
            s: pose_energy(s, entry_scores) for s in rmsd_matrix.index
        }
        have_energy = {s: e for s, e in energy_by_s.items() if e is not None}

        # geometry score = inter_tool_min_per_tool value (lower = better)
        geom = dict(closest_rmsd_scores)  # may be empty for single-tool entries

        # degeneracy: <2 tools, or exactly 2 tools with a single-pose minority
        pose_counts = sorted(len(v) for v in tool_to_structures.values())
        degenerate_geometry = (
            len(tool_to_structures) < 2
            or (len(tool_to_structures) == 2 and pose_counts[0] == 1)
        )

        fused_best = None
        if len(have_energy) == 0:
            # No energy anywhere -> pure geometry (min_per_tool winner).
            if geom:
                fused_best = min(geom, key=geom.get)
        elif degenerate_geometry:
            # Geometry can't discriminate -> energy dominates, geometry tiebreak.
            fused_best = min(
                have_energy,
                key=lambda s: (have_energy[s], geom.get(s, float("inf")), s),
            )
        elif len(have_energy) >= 2 and geom:
            # Normal fusion: summed dense ranks; missing energy = worst rank.
            geom_rank = _dense_rank(geom)
            energy_rank = _dense_rank(have_energy)
            worst = (max(energy_rank.values()) if energy_rank else 0) + 1
            candidates = [s for s in geom_rank]  # all structures with geometry
            fused_best = min(
                candidates,
                key=lambda s: (
                    geom_rank[s] + energy_rank.get(s, worst),
                    energy_by_s[s] if energy_by_s[s] is not None else float("inf"),
                    s,
                ),
            )
        else:
            # 0 or 1 relaxed poses but geometry usable -> pure geometry.
            if geom:
                fused_best = min(geom, key=geom.get)

        if fused_best is not None:
            best_structures.append({
                'Entry': entry,
                'tool': structure_to_tool[fused_best],
                'best_structure': fused_best,
                'avg_ligandRMSD': geom.get(fused_best, float('nan')),
                'method': 'fused_rank',
            })
```

- [ ] **Step 4: Run the fused-rank test to verify it passes**

Run: `python -m pytest tests/test_select_best_docked_structures.py -k fused_rank_energy -v`
Expected: PASS.

- [ ] **Step 5: Add the remaining behavior tests**

Append to `tests/test_select_best_docked_structures.py`:

```python
def test_fused_rank_no_energy_matches_geometry():
    # No energy passed -> fused_rank winner == inter_tool_min_per_tool winner.
    structs = {"E_0_chai": "chai", "E_1_chai": "chai", "E_0_boltz": "boltz"}
    rmsd = {
        frozenset({"E_0_chai", "E_1_chai"}): 0.5,
        frozenset({"E_0_chai", "E_0_boltz"}): 1.0,
        frozenset({"E_1_chai", "E_0_boltz"}): 3.0,
    }
    df = _pairwise("E", structs, rmsd)
    out = select_best_docked_structures(df, None)
    geo = out[out["method"] == "inter_tool_min_per_tool"].iloc[0]["best_structure"]
    fused = out[out["method"] == "fused_rank"].iloc[0]["best_structure"]
    assert fused == geo


def test_fused_rank_partial_energy_does_not_crash():
    # 2 chai + 2 boltz (normal case), only some poses relaxed.
    structs = {
        "E_0_chai": "chai", "E_1_chai": "chai",
        "E_0_boltz": "boltz", "E_1_boltz": "boltz",
    }
    rmsd = {
        frozenset({"E_0_chai", "E_1_chai"}): 0.5,
        frozenset({"E_0_boltz", "E_1_boltz"}): 0.5,
        frozenset({"E_0_chai", "E_0_boltz"}): 1.0,
        frozenset({"E_0_chai", "E_1_boltz"}): 2.0,
        frozenset({"E_1_chai", "E_0_boltz"}): 2.0,
        frozenset({"E_1_chai", "E_1_boltz"}): 2.0,
    }
    df = _pairwise("E", structs, rmsd)
    # only 2 of 4 poses have energy
    scores = {"E": {"chai": {"E_0": -30.0}, "boltz": {"E_0": -10.0}, "vina": {}}}
    out = select_best_docked_structures(df, scores)
    fused = out[out["method"] == "fused_rank"]
    assert len(fused) == 1  # never crashes, always emits


def test_fused_rank_vina_direction():
    # chai/boltz/vina in one entry; vina energy lower = better, same as others.
    structs = {"E_0_chai": "chai", "E_0_boltz": "boltz",
               "E_0_vina": "vina", "E_1_vina": "vina"}
    rmsd = {
        frozenset({"E_0_chai", "E_0_boltz"}): 1.0,
        frozenset({"E_0_chai", "E_0_vina"}): 1.0,
        frozenset({"E_0_chai", "E_1_vina"}): 1.0,
        frozenset({"E_0_boltz", "E_0_vina"}): 1.0,
        frozenset({"E_0_boltz", "E_1_vina"}): 1.0,
        frozenset({"E_0_vina", "E_1_vina"}): 0.5,
    }
    df = _pairwise("E", structs, rmsd)
    # vina keyed by int pose index; E_1_vina has the best (lowest) energy
    scores = {"E": {"chai": {"E_0": -5.0}, "boltz": {"E_0": -5.0},
                    "vina": {0: -6.0, 1: -50.0}}}
    out = select_best_docked_structures(df, scores)
    fused = out[out["method"] == "fused_rank"].iloc[0]
    assert fused["best_structure"] == "E_1_vina"


def test_existing_methods_schema_unchanged():
    structs = {"E_0_chai": "chai", "E_1_chai": "chai", "E_0_boltz": "boltz"}
    rmsd = {
        frozenset({"E_0_chai", "E_1_chai"}): 0.5,
        frozenset({"E_0_chai", "E_0_boltz"}): 1.0,
        frozenset({"E_1_chai", "E_0_boltz"}): 3.0,
    }
    df = _pairwise("E", structs, rmsd)
    out = select_best_docked_structures(df, None)
    assert set(out.columns) == {
        "Entry", "tool", "best_structure", "avg_ligandRMSD", "method"
    }
    assert {"inter_tool_weighted_avg", "inter_tool_min_per_tool"}.issubset(
        set(out["method"])
    )
```

- [ ] **Step 6: Run the full selector test file**

Run: `python -m pytest tests/test_select_best_docked_structures.py -v`
Expected: PASS (all tests, including Task 1's `pose_energy` tests).

- [ ] **Step 7: Commit**

```bash
git add structurezyme/steps/computeligandRMSD_step.py tests/test_select_best_docked_structures.py
git commit -m "feat(ligand_rmsd): add energy-aware fused_rank selection method"
```

### Task 3: Wire `fastrelax_score` into `LigandRMSD.__execute`

Extract the per-Entry `fastrelax_score` dict from the incoming `df` and pass
it to `select_best_docked_structures`.

**Files:**
- Modify: `structurezyme/steps/computeligandRMSD_step.py` — `LigandRMSD.__execute`
  (the `select_best_docked_structures(rmsd_df)` call at line 402)
- Test: `tests/test_select_best_docked_structures.py`

**Interfaces:**
- Consumes: `select_best_docked_structures(rmsd_df, fastrelax_score_by_entry)`
  (Task 2); the `df` passed to `__execute` (carries a per-row
  `fastrelax_score` dict duplicated per Entry, or no such column).
- Produces: a helper `_fastrelax_scores_by_entry(df, entry_col="Entry") ->
  dict[str, dict]` returning `{Entry: score_dict}` for the first non-empty
  score per entry (empty dict if the column is absent).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_select_best_docked_structures.py`:

```python
def test_fastrelax_scores_by_entry_extraction():
    from structurezyme.steps.computeligandRMSD_step import (
        _fastrelax_scores_by_entry,
    )
    df = pd.DataFrame({
        "Entry": ["A", "A", "B"],
        "fastrelax_score": [
            {"chai": {"A_0": -1.0}, "boltz": {}, "vina": {}},
            {"chai": {"A_0": -1.0}, "boltz": {}, "vina": {}},
            {"chai": {"B_0": -2.0}, "boltz": {}, "vina": {}},
        ],
    })
    out = _fastrelax_scores_by_entry(df)
    assert out["A"]["chai"]["A_0"] == -1.0
    assert out["B"]["chai"]["B_0"] == -2.0


def test_fastrelax_scores_by_entry_missing_column():
    from structurezyme.steps.computeligandRMSD_step import (
        _fastrelax_scores_by_entry,
    )
    df = pd.DataFrame({"Entry": ["A"]})
    assert _fastrelax_scores_by_entry(df) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_select_best_docked_structures.py -k fastrelax_scores_by_entry -v`
Expected: FAIL — `cannot import name '_fastrelax_scores_by_entry'`.

- [ ] **Step 3: Add the extraction helper**

In `structurezyme/steps/computeligandRMSD_step.py`, add this module-level
helper near `pose_energy`:

```python
def _fastrelax_scores_by_entry(df, entry_col: str = "Entry") -> dict:
    """Map each Entry to its fastrelax_score dict (first non-empty per entry).

    Returns {} if the column is absent. A per-entry score dict looks like
    {"chai": {pose_key: score}, "boltz": {...}, "vina": {...}}.
    """
    if "fastrelax_score" not in df.columns:
        return {}
    out: dict = {}
    for entry, group in df.groupby(entry_col, sort=False):
        for val in group["fastrelax_score"]:
            if isinstance(val, dict) and any(val.get(e) for e in ("chai", "boltz", "vina")):
                out[entry] = val
                break
    return out
```

- [ ] **Step 4: Pass the mapping into the selector**

In `LigandRMSD.__execute`, replace the call at line 402:

```python
            best_docked_structure_df = select_best_docked_structures(rmsd_df)
```

with:

```python
            fastrelax_scores = _fastrelax_scores_by_entry(df, self.entry_col)
            best_docked_structure_df = select_best_docked_structures(
                rmsd_df, fastrelax_scores
            )
```

Note: `df` here is the raw input to `__execute` (before the `rmsd_df.merge(df,
...)` at line 405), which still carries the per-row `fastrelax_score` column.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_select_best_docked_structures.py -v`
Expected: PASS (all tests).

- [ ] **Step 6: Commit**

```bash
git add structurezyme/steps/computeligandRMSD_step.py tests/test_select_best_docked_structures.py
git commit -m "feat(ligand_rmsd): pass per-entry fastrelax_score into best-pose selector"
```

### Task 4: Make `fused_rank` authoritative in PLACER

Update `_select_one_per_entry` so a pose chosen by `fused_rank` wins outright,
falling back to the existing `_method_count` logic when `fused_rank` is absent.

**Files:**
- Modify: `structurezyme/steps/PLACER_step.py` — `_select_one_per_entry`
  (lines 40-64)
- Test: `tests/test_placer_step.py`

**Interfaces:**
- Consumes: existing `is_best`, `best_method` (comma-joined token string),
  `docked_structure` columns; `_method_count` helper (line 33).
- Produces: `_select_one_per_entry(df, entry_col="Entry")` unchanged
  signature; new internal preference for the `fused_rank` token.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_placer_step.py`:

```python
def test_select_one_per_entry_fused_rank_wins_over_method_count():
    # Pose A chosen by 3 geometric methods; pose B chosen by fused_rank only.
    # fused_rank must win despite lower method count.
    df = pd.DataFrame({
        "Entry": ["Q1", "Q1"],
        "docked_structure": ["Q1_0_chai", "Q1_0_boltz"],
        "is_best": [True, True],
        "best_method": [
            "inter_tool_min_per_tool,inter_tool_weighted_avg,vina_avg_intra_tool",
            "fused_rank",
        ],
    })
    result = _select_one_per_entry(df, entry_col="Entry")
    assert len(result) == 1
    assert result.iloc[0]["docked_structure"] == "Q1_0_boltz"


def test_select_one_per_entry_fused_rank_absent_uses_method_count():
    # No fused_rank token -> existing _method_count behavior (highest wins).
    df = pd.DataFrame({
        "Entry": ["Q1", "Q1"],
        "docked_structure": ["Q1_a_chai", "Q1_b_chai"],
        "is_best": [True, True],
        "best_method": [
            "inter_tool_min_per_tool",
            "inter_tool_min_per_tool,inter_tool_weighted_avg",
        ],
    })
    result = _select_one_per_entry(df, entry_col="Entry")
    assert len(result) == 1
    assert result.iloc[0]["docked_structure"] == "Q1_b_chai"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_placer_step.py -k fused_rank -v`
Expected: the `wins_over_method_count` test FAILS (currently picks
`Q1_0_chai` by method count); the `absent` test PASSES already.

- [ ] **Step 3: Add a fused_rank helper and update the reduction**

In `structurezyme/steps/PLACER_step.py`, add this helper right after
`_method_count` (after line 37):

```python
def _has_fused_rank(best_method) -> bool:
    """True if 'fused_rank' is one of the comma-separated tokens."""
    if pd.isna(best_method) or best_method == "":
        return False
    return "fused_rank" in str(best_method).split(",")
```

Then in `_select_one_per_entry`, replace the loop body (lines 53-62) so the
`fused_rank` pool takes priority over `_method_count`. The current body is:

```python
    for _entry, group in df.groupby(entry_col, sort=False):
        pool = group[group["is_best"] == True]  # noqa: E712 (explicit bool compare intentional; NaN-safe)
        if pool.empty:
            pool = group
        pool = pool.copy()
        pool["_method_count"] = pool["best_method"].apply(_method_count)
        max_count = pool["_method_count"].max()
        pool = pool[pool["_method_count"] == max_count]
        pool = pool.sort_values("docked_structure")
        rows.append(pool.iloc[0])
```

Replace with:

```python
    for _entry, group in df.groupby(entry_col, sort=False):
        pool = group[group["is_best"] == True]  # noqa: E712 (explicit bool compare intentional; NaN-safe)
        if pool.empty:
            pool = group
        pool = pool.copy()
        # fused_rank is authoritative: if any pose in the pool was chosen by
        # the energy-aware fusion, restrict to those before other tie-breaks.
        fused = pool[pool["best_method"].apply(_has_fused_rank)]
        if not fused.empty:
            pool = fused
        pool["_method_count"] = pool["best_method"].apply(_method_count)
        max_count = pool["_method_count"].max()
        pool = pool[pool["_method_count"] == max_count]
        pool = pool.sort_values("docked_structure")
        rows.append(pool.iloc[0])
```

Also update `_has_fused_rank` import in the test file: it is not imported by
tests directly (tests only use `_select_one_per_entry`), so no test import
change is needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_placer_step.py -k "fused_rank or select_one_per_entry" -v`
Expected: PASS (new fused_rank tests + all existing `_select_one_per_entry`
tests stay green).

- [ ] **Step 5: Commit**

```bash
git add structurezyme/steps/PLACER_step.py tests/test_placer_step.py
git commit -m "feat(placer): prefer fused_rank pose in _select_one_per_entry"
```

### Task 5: Update pipeline docs + full-suite verification

Document the new `fused_rank` method and confirm nothing regressed.

**Files:**
- Modify: `docs/pipeline_overview.md` — "Known design gap" section (lines
  122-147) and the method list (lines 113-120)

- [ ] **Step 1: Update the method list**

In `docs/pipeline_overview.md`, after the `vina_avg_intra_tool` bullet
(line 119-120), add:

```markdown
4. **`fused_rank`** — energy-aware fusion: rank poses by
   `inter_tool_min_per_tool` geometry and by `fastrelax_score` (lower energy =
   better), combine as summed dense ranks (lowest wins). When geometry is
   degenerate (single-pose-per-tool / <2 tools) energy dominates; when energy
   is unavailable it falls back to the pure-geometry `inter_tool_min_per_tool`
   pick. Emitted once per entry, always.
```

- [ ] **Step 2: Rewrite the "Known design gap" section**

Replace the `### Known design gap: selector ignores fastrelax_score` section
(lines 122-147) with a resolved-status note:

```markdown
### Energy-aware selection (`fused_rank`)

The selector now fuses inter-tool geometric consensus with the Rosetta
`fastrelax_score` interaction energy via the `fused_rank` method. Energy is
passed per-Entry from `LigandRMSD.__execute` into
`select_best_docked_structures`, and PLACER's `_select_one_per_entry` treats a
`fused_rank` pick as authoritative (falling back to the geometric
`_method_count` vote only when `fused_rank` is absent, e.g. fastrelax
disabled).

Behavior:

- **Normal case** (>=2 tools, minority tool has >=2 poses, >=2 poses relaxed):
  `fused_score = geom_rank + energy_rank`; poses without energy get the worst
  energy rank so geometry still orders them.
- **Degenerate geometry** (<2 tools, or 2 tools with a single-pose minority):
  energy dominates, geometry breaks ties.
- **No energy** (fastrelax off / all relax failed): pure geometry
  (`inter_tool_min_per_tool`), identical to the previous behavior.

Score direction: `fastrelax_score` is ranked ascending (lower/more-negative
REU = better) for all engines.

**Future work:** 3-tool outlier trimming (best-2-of-3-tools consensus) is not
yet implemented; add only if a real 3-tool run shows geometry distortion.
```

- [ ] **Step 3: Run the full affected test suite**

Run: `python -m pytest tests/test_select_best_docked_structures.py tests/test_placer_step.py tests/test_tool_name_detection.py tests/test_extract_docking_metrics_helpers.py -v`
Expected: PASS (all). This covers the new selector, PLACER reduction, and the
pose-id helpers the wiring depends on.

- [ ] **Step 4: Commit**

```bash
git add docs/pipeline_overview.md
git commit -m "docs: document fused_rank energy-aware best-pose selection"
```

### Task 6: GPU integration validation (manual, user-run)

Not a code task — run by the user on a GPU host after Tasks 1-5 land.

- [ ] **Step 1:** Run the `fmo18` pipeline with `fastrelax` enabled
  (`tools/fmo18/run_fmo18.yml`).
- [ ] **Step 2:** Load `placer.pkl` and confirm best-pose selection now
  reflects energy: for entries where a pose has a clearly better
  `fastrelax_score` than the geometric-consensus pick, that pose is selected
  (previously every entry selected the geometric-consensus tool regardless of
  energy). Acceptance = general energy-aware behavior, not a dataset-specific
  count.

## Self-Review

- **Spec coverage:** Section 1 wiring → Task 3; Section 2 fused algorithm →
  Task 2 (+ helper Task 1); Section 3 PLACER → Task 4; Testing → tests in
  Tasks 1-4 + Task 5 Step 3; docs update → Task 5; GPU validation → Task 6.
  All spec sections mapped.
- **Placeholder scan:** no TBD/TODO; every code step has complete code.
- **Type consistency:** `select_best_docked_structures(rmsd_df,
  fastrelax_score_by_entry=None)`, `pose_energy(docked_structure,
  entry_scores)`, `_fastrelax_scores_by_entry(df, entry_col)`,
  `_has_fused_rank(best_method)`, `_dense_rank(score_by_key)` — names used
  consistently across tasks.

