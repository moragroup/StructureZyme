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

- **Fix**: resolved in public upstream `WRiegs/Squidly@main`, which now appends
  `['--mean_prob', str(mean_prob), '--mean_var', str(mean_var)]` to every
  `cmd = [...]` build in the ensemble path of `run` and ships
  `tests/test_cli_forwards_thresholds.py` as a regression guard. `environment.yml`
  pins `squidly` to the immutable upstream commit
  `58a8f7d6cac128c4d0915835d20ecb575cb72931`, so the fix is guaranteed on any env
  rebuild. (An earlier personal fork `HerrLuca99/Squidly@022fa40` carried the same
  fix before it landed upstream; it is no longer used.)

- **Workaround removed**: the local threshold-recompute in
  `Squidly.execute` (`_select_residues_from_ensemble` + the
  `TODO(squidly-upstream)` block) has been deleted. Thresholds are now applied
  inside squidly itself; `Squidly._build_cli_args` forwards
  `--mean-prob` / `--mean-var`, and the step consumes squidly's output
  unchanged. The obsolete `tests/test_squidly_threshold_local_apply.py` (which
  simulated the buggy upstream and asserted the local recompute) was removed;
  `tests/test_squidly_threshold_wiring.py` still verifies the thresholds are
  forwarded from run config into `Squidly()` and onto the CLI.

- **Follow-up**: upstream has no tagged release, so `environment.yml` pins a bare
  commit. Bump the pin to a proper version once `WRiegs/Squidly` cuts a tagged
  release.
