# Known Issues

Bugs in upstream dependencies that StructureZyme currently works around. Each
entry documents the bug, our workaround, and how to remove the workaround once
upstream is fixed.

### docko assumes a separate `vina` conda env and has a stale `run_boltz_affinity` signature

- **Upstream repo**: <https://github.com/ArianeMora/docko>
- **Affected version**: upstream `docko` up to and including `v0.1.4`
  (`d3273dd5e5ea879e7ffbe80a30d73c63b020086e`) / PyPI `docko<=0.1.3`.
- **Bugs**:
  1. `docko/helpers.py :: format_ligand` hardcodes
     `conda run -n vina mk_prepare_ligand.py ...`, which assumes a *separate*
     conda env literally named `vina`. In StructureZyme's single-env install
     this fails; `mk_prepare_ligand.py` is already on `PATH` (provided by
     `meeko`).
  2. `docko/boltz.py :: run_boltz_affinity` upstream takes 5 positional args,
     but `enzymetk`'s `dock_boltz_step` calls it with 6 (a trailing `args`
     list), raising `TypeError: run_boltz_affinity() takes 5 positional
     arguments but 6 were given`.
- **Workaround**: `environment.yml` pins `docko` to a patched fork
  (`moragroup/docko`, branch `fix/mk-prepare-ligand-no-conda-run`, off
  `ArianeMora/docko@d3273dd`) with two minimal patches: (1) call
  `mk_prepare_ligand.py` directly from `PATH`; (2) add a configurable `args`
  parameter to `run_boltz_affinity` and run `boltz` via `subprocess.run`.
- **Workaround removed**: drop the fork pin and use upstream `docko` once both
  fixes land upstream (a `mk_prepare_ligand.py`-on-PATH option and the 6-arg
  `run_boltz_affinity` signature).

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

## Environment notes (not upstream bugs, but easy to trip over)

### PyTorch must match the GPU architecture (CUDA 12.8 for Blackwell)

Some GPUs on this cluster are **NVIDIA RTX PRO 6000 Blackwell** cards (compute
capability `sm_120`). The `torch` build that `chai_lab` / `boltz` pull in by
default is a **CUDA 12.4** wheel (`torch==2.6.0+cu124`) that only supports up to
`sm_90`. On a Blackwell GPU it fails hard with:

```
CUDA error: no kernel image is available for execution on the device
```

`environment.yml` therefore pins the **CUDA 12.8** build
(`torch==2.11.0+cu128`) from the PyTorch cu128 wheel index
(`--extra-index-url https://download.pytorch.org/whl/cu128`). This build runs on
both Blackwell (`sm_120`) and Hopper/H100 (`sm_90`).

- `pip` prints a warning that `chai-lab 0.6.1 requires torch<2.7`. This
  constraint is overly strict; the known-good lab env runs `chai_lab 0.6.1`
  with `torch 2.11.0+cu128` without issue. The warning is expected and safe.
- If you run on a different cluster with older GPUs and no Blackwell cards, a
  `cu124` build also works — but the pinned `cu128` build is the safe default.

### Squidly model weights are a separate download (not shipped by the package)

The Squidly ensemble weights (`squidly/models/{3B,15B}/CataloDB_*.pt`, ~421 MB)
are **git-ignored in the Squidly source repo** and are **not** downloaded
automatically by either `squidly` or `enzymetk`. If they are missing, the first
pipeline step fails with:

```
ERROR: The model folder does not exist: <site-packages>/squidly/models
```

After installing the environment, download them once with Squidly's own helper:

```bash
python -m squidly.download_models_hf
```

This pulls the HuggingFace repo `WillRieger/Squidly` into
`<site-packages>/squidly/models`. See `docs/getting_started.md` for the full
first-run asset checklist (ESM2 base model, Chai assets, Boltz cache).

### The `vina` command-line binary is separate from the `vina` Python package

The Vina docking step needs the **`vina` command-line binary** on `$PATH`, which
is distinct from the `vina==1.2.7` Python bindings pinned in `environment.yml`
(docko shells out to the CLI; it does not import the Python module). bioconda
only packages an old `vina` 1.1.2. The known-good lab env uses the official
**AutoDock Vina 1.2.5** statically-linked Linux binary from the
`ccsb-scripps/AutoDock-Vina` GitHub releases
(sha256 `fa0126a28a9ea9162d1b161dfa92bc76e632416db28ca246278ea4b2dc6860cb`).

Place this binary on `$PATH` (e.g. copy it into `<env>/bin/vina` and
`chmod +x`). Verify with:

```bash
vina --version   # -> AutoDock Vina v1.2.5
```

## FastRelax & PLACER: external, out-of-process steps (default-disabled)

`fastrelax` and `placer` are opt-in steps that shell out to separate shared
installs under `/mnt/labs/data/mora/software/`; neither pyrosetta nor PLACER is
in `environment.yml` (both are licensed/heavy/external).

- **FastRelax** runs PyRosetta in a subprocess against the RosettaFastRelax venv
  at `/mnt/labs/data/mora/software/RosettaFastRelax/env` (py3.12). Override with
  `FASTRELAX_ENV=<env-dir>`. Enable via `steps.fastrelax.enabled: true`. If the
  venv is unreachable, the step raises `FileNotFoundError` at the first pose.
- **PLACER** runs `run_PLACER.py` under `/mnt/labs/data/mora/software/PLACER/env`
  (py3.10; weights `PLACER_model_1.pt` included). Enable via
  `steps.placer.enabled: true` AND set `steps.placer.placer_predict_ligand`
  (chain-resname-resnum, e.g. `A-HEM-154`, or the docked ligand resname `LIG`);
  the step raises if it is unset. Override the env with
  `paths.placer_env_path` or the host profile.
