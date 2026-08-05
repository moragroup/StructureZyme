# Energy-aware best-pose selection

**Date:** 2026-08-05
**Status:** Approved design (pre-implementation)
**Author:** Luca (with OpenCode)

## Problem

The best-pose selector `select_best_docked_structures`
(`structurezyme/steps/computeligandRMSD_step.py`) chooses the "best" docked
pose per entry using **only inter-tool geometric consensus**. It ignores the
Rosetta `fastrelax_score` (interaction energy) computed by the `fastrelax`
step and any docking-engine confidence.

This is the documented design gap in `docs/pipeline_overview.md:122-147`.

### Observed failure (fmo-fad-01)

The `fmo18` run (`tools/fmo18/run_fmo18.yml`, run id `fmo-fad-01`) enables
`chai` + `boltz` + `vina`, but for residue-less enzymes vina is skipped
per-row, so most entries reduce to **chai + boltz**. When one tool contributes
many more poses than the other (e.g. 4 chai vs 1 boltz), both consensus
methods (`inter_tool_weighted_avg`, `inter_tool_min_per_tool`) deterministically
favor a chai pose sitting near the chai cluster's own mean — the pose closest
to the single boltz point wins. This produced `tool="chai"` on **all 18
entries** in `placer.pkl`, even where the single boltz pose had a
substantially better (lower) `fastrelax_score` than any chai pose.

**Root cause:** two independent issues.

1. `select_best_docked_structures` runs on the pairwise `rmsd_df` **before**
   `fastrelax_score` is merged onto the DataFrame
   (`computeligandRMSD_step.py:402` runs before the `merge(df, ...)` at `:405`),
   so energy is never in scope for the selector.
2. Even if it were, the selector has no mechanism to combine energy with
   geometry, and PLACER's `_select_one_per_entry` (`PLACER_step.py:40-64`)
   prefers the pose chosen by the **most** geometric methods, which is exactly
   the geometrically-central chai pose.

## Goals

- Fuse inter-tool geometric consensus with Rosetta `fastrelax_score` to pick
  the best pose per entry.
- Fix the fmo-fad-01 behavior: when the single boltz pose has clearly better
  energy, it should win.
- Handle the single-pose-per-tool / degenerate-geometry case explicitly.
- Degrade gracefully when energy is missing (fastrelax disabled, or only the
  top_k poses per engine relaxed, or per-pose relax failures).
- No regression when fastrelax is off (behavior collapses to current geometry).

## Non-goals / Future work

- **3-tool outlier trimming** (best-2-of-3-tools consensus). Deferred. The
  documented bug is a 2-tool case, so trimming would not fire on fmo-fad-01,
  and `inter_tool_min_per_tool` is already partially outlier-resistant on the
  pose axis. Add only if a real 3-tool run shows geometry distortion.
- Tunable fusion weights / configurable alpha. Fusion uses equal-weight summed
  ranks; no new config knob.
- Changing engine confidence usage (chai_ptm, boltz confidence, vina
  affinities) beyond what `fastrelax` already uses for top_k selection.

## Key facts (verified during design)

- `fastrelax_score` **does** reach `LigandRMSD`'s input `df` as a **per-Entry**
  dict (a single `{"chai": {...}, "boltz": {...}, "vina": {...}}` duplicated
  across every row of the entry), because each intermediate RMSD step
  re-attaches it via `merge(df, on='Entry')`
  (`computeproteinRMSD_step.py:256`, `computeligandRMSD_step.py:405`). It is
  **not** keyed per `docked_structure`.
- `select_best_docked_structures` currently runs on the pairwise `rmsd_df`
  **before** that merge, so `fastrelax_score` is out of scope inside it.
- `fastrelax_score` dict keys (`_confidence_key_from_path`,
  `fastrelax_step.py:57`):
  - chai/boltz: file stem with trailing `_{engine}` removed, e.g.
    `"P41365_0_chai"` -> `"P41365_0"`. Keys are computed from the **original**
    (pre-relax) stem, so they carry **no** `_relaxed` suffix.
  - vina: `int(stem.split("_")[-2])` — a bare integer pose index.
- Score directions: chai/boltz higher-confidence sort is descending in
  `fastrelax`'s top_k selection, but the stored `fastrelax_score` is a Rosetta
  interaction energy where **lower (more negative) REU = better**; vina is
  also lower = better. So for energy ranking, **ascending raw score = better
  for all three engines**.
- `docked_structure` strings (`computeligandRMSD_step.py:369-387`) are pose
  stems from a `structureA__structureB.pdb` pairing, and after fastrelax carry
  a trailing `_relaxed` (e.g. `P41365_0_chai_relaxed`).
- Mapping a `docked_structure` to its energy: engine =
  `get_tool_from_structure_name(ds)` (strips `_relaxed`, takes last token);
  key = `_pose_id_from_structure_name(ds)` (`utils/helpers.py:224-249`, strips
  `_relaxed` then trailing `_{tool}`) for chai/boltz, and
  `int(stem_without_relaxed.split('_')[-2])` for vina.
- Only PLACER consumes `is_best`/`best_method`/`best_structure`. PLIP,
  ligand SASA, fpocket, and geometric-filter steps iterate **all** rows and use
  `docked_structure` directly — they do **not** filter on `is_best`, so they
  are unaffected by this change.
- No existing unit tests cover `select_best_docked_structures`. PLACER's
  `_select_one_per_entry` is covered in `tests/test_placer_step.py`.

## Design

### 1. Wiring: pass energy into the selector

In `LigandRMSD.__execute` (`computeligandRMSD_step.py`):

- Before calling the selector, extract a per-Entry `fastrelax_score` mapping
  from the incoming `df` (the column is present but currently unused there):
  `fastrelax_score_by_entry: dict[str, dict]` = `{Entry -> {"chai": {...},
  "boltz": {...}, "vina": {...}}}`. Build it by taking the first non-empty
  `fastrelax_score` value per Entry. If the column is absent or all-empty,
  the mapping is empty/`None`.
- Change the signature to
  `select_best_docked_structures(rmsd_df, fastrelax_score_by_entry=None)`.
  When `None`/empty, behavior is pure geometry (see degradation rules), so
  callers that never ran fastrelax are unaffected.

**Pose -> energy lookup helper** (new, in `computeligandRMSD_step.py` or reused
from `utils/helpers.py`): given a `docked_structure` string `ds` and the
entry's per-engine score dict, return the pose's energy or `None`:

```
engine = get_tool_from_structure_name(ds)          # 'chai'|'boltz'|'vina'
scores = entry_scores.get(engine, {})
if engine == 'vina':
    stem = ds[:-len('_relaxed')] if ds.endswith('_relaxed') else ds
    key = int(stem.split('_')[-2])
else:
    key = _pose_id_from_structure_name(ds)         # strips _relaxed then _tool
return scores.get(key)                             # None if not relaxed
```

### 2. Fused selector algorithm

`select_best_docked_structures` keeps emitting the existing three method rows
(`inter_tool_weighted_avg`, `inter_tool_min_per_tool`, `vina_avg_intra_tool`)
**unchanged**, and adds exactly one `fused_rank` row per entry.

Per entry:

1. **Geometry score per pose** = the `inter_tool_min_per_tool` value
   (`closest_rmsd_scores[s]`, already computed in the function). Lower = better.
   Poses without a defined geometry score (e.g. single-tool entries) are
   handled by the degenerate branch below.
2. **Energy score per pose** = raw `fastrelax_score` via the Section 1 lookup.
   Lower = better for all engines. `None` if the pose was not relaxed.
3. **Ranks** (dense rank within the entry, 1 = best):
   - `geom_rank`: ascending by geometry score.
   - `energy_rank`: ascending by energy score, computed over poses that have
     energy only.
4. **Fusion and degeneracy handling:**
   - **Degenerate geometry** — triggered when (a) fewer than 2 tools are
     present, OR (b) exactly 2 tools are present and the minority tool has a
      single pose (the fmo-fad-01 pattern). "Tools present" is derived from the
      poses actually present for the entry (`structure_to_tool` values), so a
      vina engine skipped per-row simply does not appear. Geometry cannot
      discriminate:
     - If >=1 pose has energy: rank by `energy_rank` (ascending); geometry
       score is the tiebreak.
     - If no pose has energy: use the current deterministic geometric pick
       (the `inter_tool_min_per_tool` winner). No regression.
   - **Normal case** (>=2 tools, minority tool has >=2 poses):
     - If >=2 poses have energy: `fused_score = geom_rank + energy_rank`.
       Poses lacking energy get `energy_rank = max_energy_rank + 1` (worst),
       so geometry still orders them relative to each other.
     - If <2 poses have energy (0 or 1 relaxed): fall back to the
       `inter_tool_min_per_tool` winner (pure geometry).
5. **Winner selection:** lowest `fused_score` wins; ties broken by better
   (lower) energy, then by alphabetical `docked_structure` (matching PLACER's
   final tiebreak style, keeps determinism).
6. **Emit** `{Entry, tool, best_structure, avg_ligandRMSD, method:
   'fused_rank'}`, schema-identical to the other method rows. `avg_ligandRMSD`
   = the winning pose's geometry score (or `NaN` if geometry undefined).

`fused_rank` is **always emitted** (it collapses to geometry when energy is
absent), so downstream consumers can rely on it being present whenever the
selector ran.

### 3. PLACER: make `fused_rank` authoritative

Update `_select_one_per_entry` (`PLACER_step.py:40-64`). Without this, PLACER's
`_method_count` still picks the 3-vote geometric pose over the 1-vote
`fused_rank` pose and the bug survives.

New reduction order per entry:

1. `pool = group[group["is_best"] == True]`; if empty, `pool = group`
   (unchanged fallback).
2. **NEW:** if any row in `pool` has `'fused_rank'` among its comma-joined
   `best_method` tokens, keep only those rows. `fused_rank` wins outright.
3. Existing tiebreaks apply to whatever `pool` remains: max `_method_count`,
   then alphabetical `docked_structure`, then `iloc[0]`.

When `fused_rank` is absent (older data / selector not run), step 2 is a no-op
and the existing `_method_count` logic is used unchanged. `_method_count`,
`best_method` token parsing, and the required-columns check
(`{entry_col, structure_col, "is_best", "best_method"}`) are unchanged.

**Blast radius:** only PLACER's single-pose-per-entry selection changes.
PLIP / ligand SASA / fpocket / geometric-filter iterate all rows and ignore
`is_best`, so they are unaffected.

## Testing

All unit tests run without GPU/PyRosetta using synthetic DataFrames, matching
the existing style in `tests/`.

New `tests/test_select_best_docked_structures.py`:

1. **fmo-fad-01 regression:** 4 chai poses + 1 boltz pose; boltz has
   substantially better (lower) energy. Assert the `fused_rank` winner is the
   boltz pose, not a chai pose.
2. **fastrelax-off fallback:** no energy passed. Assert `fused_rank` winner ==
   `inter_tool_min_per_tool` winner (pure geometry, no regression), and the
   three geometric method rows are unchanged.
3. **Partial energy:** some poses relaxed, some not. Assert no crash; energy
   poses ordered by energy, non-energy poses fall back to geometry.
4. **Single-pose-per-tool / degenerate geometry:** 2 tools with the minority
   tool contributing a single pose. With energy present, energy dominates;
   with no energy, deterministic geometric pick.
5. **vina direction:** vina energies (lower affinity = better) rank correctly
   alongside chai/boltz.
6. **Schema stability:** the three existing method rows keep the schema
   `{Entry, tool, best_structure, avg_ligandRMSD, method}`.

Extend `tests/test_placer_step.py`:

7. `fused_rank` pose wins even when a different pose has a higher
   `_method_count`.
8. When `fused_rank` is absent, existing `_method_count` behavior is unchanged
   (existing tests stay green).

**Final integration validation (on GPU):** run the `fmo18` pipeline with
`fastrelax` enabled and confirm `placer.pkl` no longer shows `tool="chai"` on
all 18 entries — specifically that entries where the boltz pose has a better
`fastrelax_score` now select boltz.

## Files touched

- `structurezyme/steps/computeligandRMSD_step.py` — energy wiring +
  `fused_rank` in `select_best_docked_structures`; pose->energy lookup helper.
- `structurezyme/steps/PLACER_step.py` — `_select_one_per_entry` prefers
  `fused_rank`.
- `docs/pipeline_overview.md` — update the "Known design gap" section to
  document the new `fused_rank` method and its behavior.
- `tests/test_select_best_docked_structures.py` — new.
- `tests/test_placer_step.py` — extended.

