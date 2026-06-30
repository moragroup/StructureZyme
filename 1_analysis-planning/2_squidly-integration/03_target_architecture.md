# 3. Target Architecture (enzymetk-backed)

## Decision

Squidly is wired through `enzymetk.ActiveSitePred`, mirroring how Chai-1 and Boltz-2 are
already wired in `pipeline_v2.py:28-30`. No local model loading inside Filterzyme.

## New thin wrapper module

`filterzyme/steps/squidly_step.py` defines a Filterzyme-owned `Squidly(Step)` class.
Its job is **only**:

1. Call `enzymetk.ActiveSitePred(...).execute(df)` underneath.
2. Normalize column names so the rest of `pipeline_v2.py` does not need a rename block.
3. Merge user-supplied `vina_residues` into the canonical `catalytic_residues` column.
4. Deduplicate by sequence so identical sequences are embedded once.
5. Log loudly when the upstream `squidly` CLI is missing, instead of failing deep in a
   `subprocess` traceback.

```
class Squidly(Step):
    def __init__(
        self,
        sequence_col: str = "Sequence",
        id_col: str = "Entry",
        model_size: Literal["3B", "15B"] = "3B",
        threshold: float | None = None,    # forwarded via enzymetk's `args=` if set
        num_threads: int = 1,
        dedup_by_sequence: bool = True,
        tmp_dir: Path | None = None,
    ): ...

    def execute(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

Internally `model_size` maps to:
- `"3B"`  -> `esm2_model="esm2_t36_3B_UR50D"`
- `"15B"` -> `esm2_model="esm2_t48_15B_UR50D"`

`threshold`, if set, is passed as `args=["--threshold", str(threshold)]`. The exact flag
name must be confirmed in Phase 0 (see `09_phase0_squidly_install_and_schema.md`); if
the CLI uses a different flag (e.g. `--AS-threshold`), update this line.

## Column contract (tentative - finalized in Phase 0)

Adds at least:

| Column | Type | Source |
|--------|------|--------|
| `catalytic_residues` | `str` | Pipe-delimited 1-indexed positions from the CLI's residue list, plus user-supplied `vina_residues` merged in |
| `squidly_threshold` | `float \| None` | Whatever we passed; recorded for traceability |

Conditional on Phase 0 schema discovery:

| Column | Type | Add if |
|--------|------|--------|
| `squidly_scores` | `list[float]` | The CLI's pkl includes per-residue probabilities |

`vina_residues` takes precedence over Squidly's prediction, exactly as today
(`pipeline_v2.py:136-137`).

## Driver integration

`pipeline_v2.py`:

1. Replace `from enzymetk.predict_catalyticsite_step import ActiveSitePred` (line 30)
   with `from filterzyme.steps.squidly_step import Squidly`.
2. In `Docking.__init__`, accept `squidly_model_size`, `squidly_threshold`,
   `squidly_num_threads` kwargs.
3. Rewrite `_catalytic_residue_prediction` (line 95) to call `Squidly(...).execute(df)`
   directly. The existing rename block at line 104 (`df_cat_res.label`,
   `df_cat_res.Squidly_Ensemble_Residues`) becomes redundant once `Squidly` already
   emits `Entry` and `catalytic_residues`.

## What is deleted

The enzymetk decision means the locally-vendored Squidly inference is **dead code** and
goes away in Phase D:

- `filterzyme/squidly_final_models/SQUIDLY_run_model_LSTM.py`
- `filterzyme/squidly_final_models/infer_AS.py`
- `filterzyme/squidly_final_models/Squidly_LSTM_3B.pth`
- `filterzyme/squidly_final_models/Squidly_LSTM_15B.pth`
- `filterzyme/squidly_final_models/3B/`, `15B/`
- `filterzyme/squidly_final_models/.DS_Store`
- `filterzyme/steps/predict_catalyticsite_step.py`
- `filterzyme/steps/predict_catalyticsite_run.py`
- `filterzyme/pipeline.py` (already approved for deletion by the canonical-pipeline
  decision)

Before deleting the `.pth` files, confirm with the PI they are not needed for an
unrelated workflow. They might be the upstream Squidly's own weights and could be
re-uploaded to wherever Squidly distributes them - see Q-6 in
`07_risks_and_open_questions.md`.

## Error model

- `squidly` CLI missing on `$PATH` -> `Squidly.execute` catches the `FileNotFoundError`
  from the subprocess and re-raises with a clear message pointing at
  `3_setup-sanity-check/10_squidly_install.md`.
- CLI returns non-zero -> log stderr and raise.
- Empty sequence or non-amino-acid characters -> raise `ValueError(entry_id)` before
  the subprocess.
- CUDA OOM in the CLI -> not catchable from Python; document as a tuning issue (smaller
  batch, switch to 3B).

## Module map (after the change)

```
filterzyme/
  steps/
    squidly_step.py            NEW - thin enzymetk wrapper, ~80-120 LOC
    predict_catalyticsite_step.py     DELETED
    predict_catalyticsite_run.py      DELETED
  squidly_final_models/        DELETED entire directory
  pipeline.py                  DELETED
  pipeline_v2.py               edited: import + _catalytic_residue_prediction body
```
