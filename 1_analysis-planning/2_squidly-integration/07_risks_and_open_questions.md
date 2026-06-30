# 7. Risks and Open Questions

## Risks

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| R-1 | Upstream `squidly` CLI not installed on this machine | **high, current blocker** | Phase 0 installs it; see `09_phase0_squidly_install_and_schema.md` |
| R-2 | `squidly_ensemble.pkl` schema differs from what `pipeline_v2.py:104` assumes | high | Phase 0 step 0.4 verifies; wrapper renames as needed |
| R-3 | `squidly` CLI threshold flag is named something other than `--threshold` | medium | Phase 0 step 0.2 captures `squidly run --help`; wrapper uses the actual flag |
| R-4 | `squidly` pip/git source not findable | medium | Ask PI; Q-1 below |
| R-5 | Per-residue scores not exposed by the CLI | low | Drop `squidly_scores` from column contract; Phase 0 decides |
| R-6 | Removing `filterzyme/squidly_final_models/*.pth` breaks an unrelated workflow | medium | PI sign-off before Phase D step 9 |
| R-7 | ESM2 15B does not fit on lab GPU | medium | Default `model_size='3B'`; 15B is opt-in |
| R-8 | Removing `pipeline.py` breaks an unknown downstream user | low | Already approved by repo analysis; announce in README |
| R-9 | enzymetk version drift changes `ActiveSitePred` signature | medium | Pin `enzymetk==<version>` in `environment.yml` once Phase 0 is complete |

## Open questions for the PI / upstream

- **Q-1 (upstream provenance).** Where do `Squidly_LSTM_3B.pth` and `Squidly_LSTM_15B.pth`
  in `filterzyme/squidly_final_models/` come from? Is there a public Squidly repo with a
  tagged release we should track? If yes, we should `git submodule` or pin a version
  rather than vendor `SQUIDLY_run_model_LSTM.py` in-tree.
- **Q-2 (canonical threshold).** Is 0.90 (CLI default) or 0.97 (silently hard-coded) the
  intended production threshold? Affects every existing benchmark number.
- **Q-3 (column contract).** Anyone consuming `Squidly_CR_Position` downstream of
  `pipeline.py`? Renaming to `catalytic_residues` everywhere will break callers that
  read the per-stage pickles directly.
- **Q-4 (model size policy).** Should 15B be the documented default for production runs,
  with 3B as a "fast mode", or vice versa? Drives the README example.
- **Q-5 (enzymetk relationship).** Is `enzymetk.ActiveSitePred` maintained by the same
  group? If yes, can we upstream the column-name and threshold fixes there and drop
  the local fallback entirely?
- **Q-6 (weights hosting).** Is there an approved place to host the `.pth` files for
  the `scripts/fetch_squidly_weights.sh` fallback? Group S3 bucket, Zenodo, HuggingFace?
- **Q-7 (license).** What license do the Squidly LSTM weights ship under? Must be
  compatible with Filterzyme's LICENSE before we redistribute.

## Decisions to record once made

Each answered question above should be turned into a one-line entry here, with a date and
the deciding person, so the plan stays auditable. Example template:

```
- [2026-07-03 / PI] Q-2: production threshold is 0.90.
```
