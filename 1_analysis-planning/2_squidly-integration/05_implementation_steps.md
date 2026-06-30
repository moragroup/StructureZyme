# 5. Implementation Steps (enzymetk-backed)

Phases in order. Phase 0 is a hard gate: do not start Phase A until Phase 0's exit
criteria are met.

## Phase 0 - install `squidly`, discover pickle schema

See `09_phase0_squidly_install_and_schema.md`. Exit criteria:

- `which squidly` resolves in `filterzyme` conda env.
- `squidly run --help` output captured.
- `squidly_ensemble.pkl` produced for a 1-sequence smoke test.
- Pickle schema documented.
- Column contract in `03_target_architecture.md` reconciled.

## Phase A - new thin wrapper

1. **Create `filterzyme/steps/squidly_step.py`** per `03_target_architecture.md`.
   - Wraps `enzymetk.ActiveSitePred`.
   - Maps `model_size` -> `esm2_model` string.
   - Maps `threshold` -> `args=["--threshold", str(t)]` (flag name confirmed in Phase 0).
   - Dedup by `Sequence` (mirror `pipeline_v2.py:98-101`).
   - After the underlying call, rename CLI output columns into the canonical
     `catalytic_residues` (+ `squidly_scores` if Phase 0 shows scores exist).
   - Merge `vina_residues` if present; user residues win.
   - Raise a clear error if `squidly` is not on `$PATH`.

2. **Unit test `tests/test_squidly_step.py`** - one short sequence, asserts
   `catalytic_residues` is a non-empty string and column types are correct. Skipped
   automatically (`pytest.skip`) if `which squidly` is missing.

## Phase B - wire into pipeline_v2

3. **Edit `filterzyme/pipeline_v2.py`**:
   - Line 30: replace `from enzymetk.predict_catalyticsite_step import ActiveSitePred`
     with `from filterzyme.steps.squidly_step import Squidly`.
   - `Docking.__init__`: add `squidly_model_size: str = "3B"`,
     `squidly_threshold: float | None = None`,
     `squidly_num_threads: int = 1`.
   - `_catalytic_residue_prediction` (line 95): call `Squidly(...).execute(df)`.
     Delete the rename block at lines 104-106 (no longer needed; the wrapper already
     emits `Entry` and `catalytic_residues`).
   - Leave the `skip_catalytic_residue_prediction` flag and the
     `vina_residues`-fallback logic untouched.

4. **Forward kwargs from `Pipeline` to `Docking`** so users can configure Squidly
   without subclassing.

## Phase C - lab smoke test

5. **Run the README quick-start without `skip_catalytic_residue_prediction`** on the
   lab machine. Document the run in `4_lab-sanity-run/05_squidly_first_run.md`.
6. Confirm `catalytic_residues` is populated in the final
   `structural_features_final.pkl`, and that Vina uses it (search the Vina logs for
   the residue list).

## Phase D - cleanup

Only after Phase C is green:

7. **Delete `filterzyme/pipeline.py`** (already approved deprecated).
8. **Delete `filterzyme/steps/predict_catalyticsite_step.py`** and
   `filterzyme/steps/predict_catalyticsite_run.py`.
9. **Delete `filterzyme/squidly_final_models/`** entirely (after PI confirms the
   `.pth` files are not needed elsewhere - see Q-6).
10. **Remove `.DS_Store`** appearances from the repo, add to `.gitignore`.
11. **Update `environment.yml`** with the `squidly` pin from Phase 0.
12. **Update `README.md`**:
    - Remove `skip_catalytic_residue_prediction=True` from the quick-start.
    - Add a one-liner for `squidly_model_size='3B'`.
    - Add a sentence: "Catalytic-residue prediction requires the `squidly` CLI; see
      docs/getting_started.md."

## Phase E - benchmarking compatibility

13. **Update `benchmarking/*/run_*.py`** to import from `pipeline_v2` and drop any
    references to the deleted `AS_inference` conda env or `predict_catalyticsite_*`
    modules.
14. **Replace `test_pipeline.py`** (currently imports the long-gone
    `filtering_pipeline`) with a minimal smoke test of `pipeline_v2.Pipeline` that is
    guarded by `pytest.skip` if `squidly` is missing.

## Phase F - docs

15. **Add `3_setup-sanity-check/10_squidly_install.md`** with the install command,
    `squidly --version`, and the smoke-test transcript.
16. **Add `4_lab-sanity-run/05_squidly_first_run.md`** with the first green run on
    lab hardware.

## File-by-file change summary

| File | Change |
|------|--------|
| `filterzyme/steps/squidly_step.py` | **NEW**, ~100 LOC |
| `filterzyme/pipeline_v2.py` | edit lines 30 (import), 49-69 (`__init__` kwargs), 95-142 (`_catalytic_residue_prediction` body) |
| `filterzyme/pipeline.py` | DELETE |
| `filterzyme/steps/predict_catalyticsite_step.py` | DELETE |
| `filterzyme/steps/predict_catalyticsite_run.py` | DELETE |
| `filterzyme/squidly_final_models/` | DELETE (after PI sign-off) |
| `environment.yml` | add `squidly` pin |
| `README.md` | drop `skip_catalytic_residue_prediction=True` from quick-start |
| `tests/test_squidly_step.py` | NEW |
| `test_pipeline.py` | replace stale import with `pipeline_v2` smoke test |
| `3_setup-sanity-check/10_squidly_install.md` | NEW |
| `4_lab-sanity-run/05_squidly_first_run.md` | NEW |
