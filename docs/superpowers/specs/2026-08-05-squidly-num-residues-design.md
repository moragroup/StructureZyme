# Design: `squidly_num_residues` — top-N catalytic residue selection

Date: 2026-08-05
Status: Approved (pending spec review)

## Problem

Squidly predicts catalytic residues by a **dual-threshold rule** applied to a
5-model ensemble. For each residue position `i` it produces:

- `mean[i]` — mean predicted catalytic probability across the 5 models.
- `variance[i]` — model disagreement at that position.

A residue is selected iff `mean[i] > mean_prob AND variance[i] < mean_var`
(defaults `mean_prob=0.6`, `mean_var=0.225`; see upstream
`squidly/squidly.py :: compute_uncertainties`).

For divergent enzyme families where the canonical `0.6` threshold is too
strict (e.g. flavin monooxygenases lacking a canonical catalytic triad), this
yields an **empty** `Squidly_CR_Position`. Downstream this leaves
`catalytic_residues` empty, so vina has no docking box and is skipped for that
entry.

Tuning the two thresholds by hand to surface exactly the best 2-3 residues is
fragile: `mean_prob` and `mean_var` interact (lowering `mean_prob` will never
surface a residue rejected by the `variance` gate), so getting a fixed count
requires adjusting two knobs together, per enzyme.

## Goal

Let the user request a fixed **number** of catalytic residues (1, 2, 3, …)
ranked by per-residue mean probability, instead of tuning thresholds. This
gives a predictable "give me the N best residues" knob and guarantees vina
gets a docking site for hard enzymes.

## Selection logic

- Rank all residue positions by `mean[i]` (descending).
- Take the top N positions. **Variance is ignored** — pure top-N by mean.
- Ties broken deterministically by lower residue index (natural stable
  `np.argsort` order).
- If N exceeds the number of scored positions (the length of the `mean`
  array, i.e. the sequence length), return all available positions.
- Output format unchanged: `Squidly_CR_Position` is a `|`-joined string of
  0-indexed positions, matching the existing threshold path.

## Behavior and precedence

- **Opt-in / backward compatible:** if `squidly_num_residues` is unset, squidly
  behaves exactly as today (threshold rule via `mean_prob` / `mean_var`).
  Nothing changes for existing configs.
- **When `squidly_num_residues` is set:** top-N by mean is used and
  `mean_prob` / `mean_var` are **ignored**, even if also set. If both a count
  and a threshold key are configured, log a note that the thresholds are
  ignored so the user is not surprised.
- Validation: `squidly_num_residues`, when set, must be a positive integer.

## Configuration

New YAML config key under the `squidly` step:

```yaml
squidly:
  squidly_num_residues: 3   # optional; unset => threshold behavior
```

No schema change is required: `StepConfig` uses `model_config =
ConfigDict(extra="allow")` (`structurezyme/config.py`), so `squidly_*` keys are
accepted. The naming follows the existing `squidly_*` snake_case convention
(`squidly_mean_prob`, `squidly_mean_var`, `squidly_num_threads`).

## Where the changes go

Mirrors the existing `mean_prob` / `mean_var` plumbing, applied in the
StructureZyme wrapper (where the per-residue `mean` array is already surfaced),
so no upstream squidly change is needed.

1. **Config key:** `squidly_num_residues` (int). No schema change (`extra="allow"`).
2. **`run_squidly`** (`structurezyme/step_runners.py:184-196`): read
   `opts.get("squidly_num_residues", None)` and pass to the `Squidly`
   constructor.
3. **`Squidly.__init__`** (`structurezyme/steps/squidly_step.py:97-128`): add
   `num_residues: int | None = None`; store `self.num_residues`; validate it is
   a positive int when set.
4. **Selection in `Squidly.execute`** (`structurezyme/steps/squidly_step.py`
   around the existing `TODO(squidly-upstream)` block, lines ~266-273): add a
   helper `_select_top_n_from_ensemble(mean, num_residues)` alongside
   `_select_residues_from_ensemble`. When `self.num_residues` is set, use it on
   the `mean` array and skip the threshold path. The "thresholds ignored"
   warning is emitted here when both a count and a threshold are set.

## Squidly upstream bug (context — already resolved)

The upstream threshold-to-worker bug (outer `squidly run` CLI dropped
`--mean_prob` / `--mean_var` when spawning the inner ensemble worker) has been
fixed in the fork `HerrLuca99/Squidly` (branch
`fix/forward-mean-prob-mean-var-cli`) and reinstalled into the `filterzyme`
env as an editable install. The local workaround in `squidly_step.py`
(`TODO(squidly-upstream)`) is retained for now and will be retired as part of
this top-N work later. Because the package is editable-installed, top-N is
applied in the wrapper and needs no further upstream change.

## Testing

- Top-N returns exactly N residues, highest-mean first.
- Ties broken by lower residue index (deterministic).
- N larger than the number of scored positions → returns all available.
- `squidly_num_residues` unset → identical to current threshold behavior
  (regression).
- `squidly_num_residues` set alongside `mean_prob` / `mean_var` → thresholds
  ignored and a warning is logged.
- Positive-integer validation rejects zero / negative / non-int values.
