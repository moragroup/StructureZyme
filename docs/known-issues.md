# Known Issues

Bugs in upstream dependencies that StructureZyme currently works around. Each
entry documents the bug, our workaround, and how to remove the workaround once
upstream is fixed.

_No active workarounds at present._

## Resolved

### Squidly CLI dropped `--mean-prob` / `--mean-var` before subprocess (resolved 2026-08)

- **Upstream repo**: <https://github.com/WRiegs/Squidly>
- **Affected version**: `squidly==0.1.0` — `squidly/__main__.py :: run` built the
  inner `python squidly.py ...` command without appending `--mean_prob` /
  `--mean_var`, so the ensemble worker always filtered at its argparse defaults
  (`0.6 / 0.225`) regardless of the values passed on the outer CLI. For proteins
  whose ensemble probabilities never exceed 0.6 (e.g. flavin monooxygenases
  without a canonical Cys-His-His triad) this yielded an empty residue string.

- **Fix**: patched in the fork
  <https://github.com/HerrLuca99/Squidly> on branch
  `fix/forward-mean-prob-mean-var-cli` (commit `022fa40`), which appends
  `['--mean_prob', str(mean_prob), '--mean_var', str(mean_var)]` to every
  `cmd = [...]` build in the ensemble path of `run`. `environment.yml` pins
  `squidly` to that branch, so the fix is guaranteed on any env rebuild.

- **Workaround removed**: the local threshold-recompute in
  `Squidly.execute` (`_select_residues_from_ensemble` + the
  `TODO(squidly-upstream)` block) has been deleted. Thresholds are now applied
  inside squidly itself; `Squidly._build_cli_args` forwards
  `--mean-prob` / `--mean-var`, and the step consumes squidly's output
  unchanged. The obsolete `tests/test_squidly_threshold_local_apply.py` (which
  simulated the buggy upstream and asserted the local recompute) was removed;
  `tests/test_squidly_threshold_wiring.py` still verifies the thresholds are
  forwarded from run config into `Squidly()` and onto the CLI.

- **Follow-up**: once the fix is merged into the real upstream
  `WRiegs/Squidly` and released, repoint the `squidly` pin in
  `environment.yml` from the fork branch to that upstream release.
