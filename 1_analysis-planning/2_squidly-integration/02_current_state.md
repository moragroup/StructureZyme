# 2. Current State of Squidly in the Repo

## Two divergent integrations

### v1 path (`pipeline.py`) - subprocess chain

`pipeline.py:9` imports `ActiveSitePred` from
`filterzyme.steps.predict_catalyticsite_step`. That class wraps
`predict_catalyticsite_run.py`, which at line 19 does:

```
os.system("conda run -n AS_inference python <squidly_dir>/SQUIDLY_run_model_LSTM.py ...")
```

Three nested process boundaries (Python -> `subprocess.run` -> `os.system` ->
`conda run`). Return code at `predict_catalyticsite_run.py:23` is discarded, so a missing
`AS_inference` env fails silently. Output columns: `label`, `Squidly_CR_Position`.
Threshold hard-coded to **0.97** at line 14, overriding the CLI default 0.90 at line 32.

### v2 path (`pipeline_v2.py`) - direct import

`pipeline_v2.py:31` imports `ActiveSitePred` from `enzymetk.predict_catalyticsite_step`.
Called at `pipeline_v2.py:86` via `ActiveSitePred('Entry', 'Sequence')`. No subprocess.
Output columns: `id`, `residues`. The v2 driver code at `pipeline_v2.py:78`
(`_catalytic_residue_prediction`) then normalises `Squidly_CR_Position` and
`vina_residues` into `catalytic_residues` - but the column it actually receives from
enzymetk is `residues`, not `Squidly_CR_Position`. **The two halves don't agree on
column names**; this is one of the silent bugs called out in
`../1_repo_analysis/05_bugs_by_severity.md`.

## Local model weights

`filterzyme/squidly_final_models/` contains:

```
3B/                       - ESM2-3B variant assets
15B/                      - ESM2-15B variant assets
infer_AS.py               - inference entry, takes sequence list
SQUIDLY_run_model_LSTM.py - older CLI entry used by v1 subprocess
Squidly_LSTM_3B.pth       - 3B LSTM head weights
Squidly_LSTM_15B.pth      - 15B LSTM head weights
```

Both LSTM `.pth` files are checked into git (sizing TBD; confirm git-lfs status before
any cleanup). A `.DS_Store` is also checked in - cleanup item.

## What works today

- The README example uses v2 with `skip_catalytic_residue_prediction=True`, i.e. it
  bypasses Squidly entirely. So end-to-end runs **never exercise** the integration.
- `benchmarking/*/run_*.py` scripts mostly call v1 directly, which is why they require
  the `AS_inference` conda env.

## What is broken or fragile

| Issue | Location | Effect |
|-------|----------|--------|
| Column name mismatch between enzymetk output and v2 driver | `pipeline_v2.py:78`-`pipeline_v2.py:120` vs `enzymetk.ActiveSitePred` | `catalytic_residues` may be empty when v2 is used |
| Threshold silently overridden | `predict_catalyticsite_run.py:14` | v1 uses 0.97 instead of documented 0.90 |
| Silent failure on missing env | `predict_catalyticsite_run.py:23` | `os.system` return value discarded |
| Three nested processes | `predict_catalyticsite_step.py` + `_run.py` | Opaque tracebacks; impossible to debug GPU OOM |
| No GPU device selection | both paths | Always claims default CUDA device |
| Weights path hard-coded | `SQUIDLY_run_model_LSTM.py` (in `squidly_final_models/`) | Cannot relocate weights |
| `.DS_Store` and large `.pth` in git | `squidly_final_models/` | Repo bloat; should be git-lfs or external |
| Dead code | `pipeline.py` Squidly branch | Will be removed once v2 has a real Squidly step |

## What `enzymetk.ActiveSitePred` actually does (verified 2026-06-29)

Source read at
`envs/filterzyme/lib/python3.11/site-packages/enzymetk/predict_catalyticsite_step.py`.

Constructor (`line 20-33`):

```
ActiveSitePred(id_col, seq_col, num_threads=1,
               esm2_model='esm2_t36_3B_UR50D',
               tmp_dir=None, args=None, env_name='enzymetk')
```

Exposed knobs:
- `esm2_model` -> selects ESM2 backbone (3B default; `esm2_t48_15B_UR50D` for 15B).
- `args` -> appended verbatim to the CLI invocation (line 50-51); the only way to pass
  a threshold.
- `num_threads`, `tmp_dir`, `env_name`.

Hidden behaviour:
- It is **not a Python model call**. It shells out (`line 49`):
  `squidly run <input.fasta> <esm2_model> <tmp_dir> [args...]`.
- It reads `<tmp_dir>/squidly_ensemble.pkl` (line 58) and returns whatever DataFrame
  the CLI wrote. The schema of that pickle is owned entirely by the upstream `squidly`
  CLI, not by enzymetk.

Critical install gap:
- The `squidly` CLI is **not installed** in any conda env on this host. Verified by
  searching `envs/filterzyme/bin/`, `envs/enzymetk/bin/`, and the site-packages tree
  for any `squidly` entry. None found.
- This means `_catalytic_residue_prediction` has never run successfully on this
  machine. The README example bypasses it via `skip_catalytic_residue_prediction=True`,
  hiding the gap.

See `09_phase0_squidly_install_and_schema.md` for the first concrete work item.
