# Outdated / deprecated code

These files were moved here (not deleted) during the Squidly integration
(2026-06-30) because they are superseded by the new enzymetk-backed
`filterzyme/steps/squidly_step.py` wrapper and the canonical
`filterzyme/pipeline_v2.py`.

Nothing in this directory is imported by the active pipeline. It is kept for
reference and for PI sign-off before final removal.

## What's here

| File | Why it's here |
|------|---------------|
| `pipeline_v1.py` | The old `filterzyme/pipeline.py` — deprecated in favor of `pipeline_v2.py` (canonical-pipeline decision). Used a nested `os.system` + `conda run -n AS_inference` subprocess chain for Squidly, with a silently overridden threshold (0.97 vs documented 0.90) and discarded return codes. |
| `filterzyme_steps/predict_catalyticsite_step.py` | Old local-bundled Squidly wrapper. Called `predict_catalyticsite_run.py` via `subprocess.run`. Replaced by `filterzyme/steps/squidly_step.py` which calls `enzymetk.ActiveSitePred` (which in turn calls the upstream `squidly` CLI). |
| `filterzyme_steps/predict_catalyticsite_run.py` | Old subprocess shim that invoked `SQUIDLY_run_model_LSTM.py` via `os.system("conda run -n AS_inference ...")`. The `AS_inference` env was never present on this machine; the return code was discarded. Dead code. |
| `squidly_final_models/` | Locally-vendored ESM2-LSTM weights and inference scripts. With the enzymetk-backed approach, the upstream `squidly` CLI owns its own weights (downloaded from HuggingFace into `site-packages/squidly/models/`). These `.pth` files are dead weight. Kept here pending PI confirmation that they aren't needed elsewhere (see `1_analysis-planning/2_squidly-integration/07_risks_and_open_questions.md` Q-6). |

## When to fully remove

Once the PI signs off on Q-6 (the `.pth` files) and the Phase C lab smoke run
has been green for a reasonable period, this entire `outdated/` directory can
be deleted and the deletion committed.
