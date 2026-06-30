# 1. Goals and Scope

## What "include Squidly" means here

Squidly (ESM2 embeddings + LSTM head) predicts per-residue catalytic-site probabilities for
an input protein sequence. In Filterzyme it produces the `catalytic_residues` column that
drives:

- Vina pocket selection (`dock_vina_step.py:44`)
- Geometric filtering distance/angle anchors (`geometric_filtering_*` modules)

Today there are **two** Squidly integrations and **neither is a clean default**:
- v1: local subprocess to a separate `AS_inference` conda env (`predict_catalyticsite_run.py:19`)
- v2: `enzymetk.ActiveSitePred` direct import (`pipeline_v2.py:31`)

They emit different column names, use different thresholds, and one silently swallows errors.
See `02_current_state.md` for details.

## Goals (must-have)

1. **Single, supported Squidly step** importable as `filterzyme.steps.squidly_step.Squidly`,
   used by `pipeline_v2.py` by default.
2. **Thin wrapper over `enzymetk.ActiveSitePred`** (same pattern as Chai/Boltz). The
   wrapper hides one subprocess hop to the upstream `squidly` CLI, but removes the
   v1-era `subprocess + os.system + conda run` chain entirely.
3. **Stable column contract**: produce a canonical column `catalytic_residues`
   (pipe-delimited 1-indexed residue numbers), plus `squidly_scores` (list[float]) and
   `squidly_threshold` (float) for traceability.
4. **Configurable threshold** (default 0.90 to match upstream CLI; not 0.97 as silently
   hard-coded in `predict_catalyticsite_run.py:14`).
5. **Model-size selector** (`3B` or `15B` ESM2 backbone), with weights resolved from
   `filterzyme/squidly_final_models/` and overridable by env var or config.
6. **GPU device selection** via config; fall back to CPU with a clear log warning.
7. **Deduplication by sequence** (preserve v2 behaviour at `pipeline_v2.py:78`) so identical
   sequences are only embedded once.
8. **Sanity test** wired into `tests/` that runs Squidly on one short sequence end-to-end
   in <2 min on a single GPU.

## Non-goals (out of scope for this plan)

- Re-training Squidly or changing the LSTM architecture.
- Replacing Chai/Boltz/Vina or the geometric filters.
- Multi-GPU sharding of Squidly itself (the 15B ESM2 forward pass is the bottleneck;
  multi-GPU is part of `09_features_to_add.md` F-1, tracked separately).
- Building a CLI; that is feature F-7 in the global roadmap.

## Definition of done

- `pipeline_v2.Pipeline(...).run()` on the README example produces a non-empty
  `catalytic_residues` column with the new `Squidly` step.
- `pipeline.py` and `filterzyme/steps/predict_catalyticsite_run.py` are removed
  (or kept only behind a `--legacy-squidly` flag).
- `enzymetk.ActiveSitePred` is no longer the default code path for catalytic-residue
  prediction; it remains available as a fallback for users without local weights.
