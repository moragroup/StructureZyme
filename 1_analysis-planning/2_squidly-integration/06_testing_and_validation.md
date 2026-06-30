# 6. Testing and Validation

Three layers: unit, integration, lab sanity. All three must pass before Phase D cleanup
deletes the old v1 Squidly code.

## Unit tests (`tests/test_squidly_step.py`)

Fast (<60 s) tests that exercise the new step in isolation, CPU-only.

1. **Smoke**: short sequence, 3B model; assert non-empty `catalytic_residues`. The
   `len(squidly_scores) == len(Sequence)` assertion only applies if Phase 0 schema
   discovery confirmed the CLI pickle contains per-residue scores. Otherwise drop this
   assertion. All Squidly unit tests are auto-skipped (`pytest.skip("squidly CLI not on
   PATH")`) when the binary is missing, so CI does not fail on dev boxes.
2. **Known-residue**: a serine hydrolase from `benchmarking/serine_hydrolases/` whose
   catalytic Ser/His/Asp positions are known; assert the Ser index is in the predicted
   set at threshold 0.90.
3. **Dedup**: DataFrame with the same `Sequence` repeated 5x and 5 different `Entry`s;
   assert only one forward pass is made (mock the inference function and count calls).
4. **User-supplied `vina_residues`**: assert they are merged into `catalytic_residues`
   even when Squidly's own predictions are empty.
5. **Threshold sweep**: assert that lowering the threshold increases the residue count
   monotonically.
6. **Bad input**: non-amino-acid characters -> `ValueError` mentioning the `Entry`.
7. **enzymetk call**: monkey-patch `enzymetk.ActiveSitePred` and assert the wrapper
   passes through `esm2_model` and `args` correctly for both 3B and 15B `model_size`,
   and for `threshold=None` vs `threshold=0.9`.

## Integration test (`tests/test_pipeline_v2_smoke.py`)

Slow (~5 min) test, marked `@pytest.mark.slow`. Runs `pipeline_v2.Pipeline` on a 1-entry
DataFrame with a small sequence, Squidly enabled, Vina/Chai/Boltz disabled via flags
(or mocked). Asserts that the final pickle contains a non-empty `catalytic_residues`.

## Lab sanity run

Documented under `4_lab-sanity-run/05_squidly_first_run.md`:

1. Run the README quick-start example **without** `skip_catalytic_residue_prediction`.
2. Confirm the produced `<output_dir>/.../structural_features_final.pkl` has
   `catalytic_residues` populated for the test enzyme.
3. Compare the residue list against the v1 result (from before the cleanup, captured in
   `4_lab-sanity-run/04_first_green_run.md`). They should agree on at least the top-1
   predicted catalytic residue at threshold 0.90.

## Regression gates

- CI (or local pre-commit): run unit tests on every push.
- Lab sanity: re-run after every dependency bump to `fair-esm`, `torch`, or `enzymetk`.

## What we do **not** test here

- Numerical equivalence with upstream `SQUIDLY_run_model_LSTM.py` to 1e-6 - the LSTM
  forward pass is deterministic given the same weights and ESM2 version, but pinning
  every upstream byte-for-byte is out of scope. We assert ordinal agreement on the
  top-k predicted residues instead.
- Full pipeline end-to-end with Chai+Boltz+Vina - tracked in `4_lab-sanity-run/`.
