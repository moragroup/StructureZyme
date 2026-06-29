# Lab sanity-run journal — chronological log

Every step is dated/timed where possible. Each item lists what we did, why, where the artefact lives, and how to revert.

## 2026-06-26 — Task 00 (env setup)

### Step 1 — GPU session
- `srun -p gpu --qos=normal --gres=gpu:1 -A mora --mem=64G -t 1-00:00:00 --pty bash -l` → landed on `gpu03`, job 76707.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, driver 610.43.02, compute capability (12, 0) = sm_120.
- All three gpu nodes are identical Blackwells (gres `gpu:rtx6000:8`). The `h100` partition has different GPUs (sm_90).

### Step 2 — Ariane's env not available
- `conda env list` shows no `filterzyme`. Her env is in her home dir, not shared. **Plan assumption broken.**
- Decision: build env locally per her README. Document divergence in PI note.

### Step 3 — Build env (per Ariane's README)
- `module load miniforge/25.9.1`
- `conda create --name filterzyme python=3.11 pip -y` → env at `/mnt/storage01/home/lherrmann/envs/filterzyme/`.
- `conda activate filterzyme`
- `pip install /mnt/labs/data/mora/code/Filterzyme/dist/filterzyme-0.0.6.tar.gz --use-deprecated=legacy-resolver`
  - Installed: filterzyme 0.0.6, enzymetk 0.1.0, docko 0.1.3, boltz 2.2.1, torch 2.12.1+cu13, chai_lab not yet.
  - Skipped Ariane's `pip install enzymetk==0.0.8` step because filterzyme already pulled 0.1.0; downgrading would break it.
- `pip install chai_lab` → installed chai_lab 0.6.1; **downgraded torch 2.12.1+cu13 → 2.6.0+cu12** (chai_lab pin).
- Revert: `rm -rf ~/envs/filterzyme`

### Step 4 — R-BLACKWELL hit and fixed
- Test: `torch.randn(8,8, device='cuda')` → `RuntimeError: CUDA error: no kernel image is available for execution on the device`. Cause: torch 2.6 supports sm_50..sm_90, GPU is sm_120.
- Snapshot saved: `$HOME/filterzyme-sanity/torch26_state_LCH.txt` (full `pip freeze`).
- Fix: `pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch triton`
- Result: torch 2.11.0+cu128, triton 3.6.0. Kernel probe succeeded.
- Revert: `pip install -r $HOME/filterzyme-sanity/torch26_state_LCH.txt`

### Step 5 — Missing conda packages (round 1)
- Snapshot: `$HOME/filterzyme-sanity/conda_pre_pdbfixer_LCH.txt`, `pip_pre_pdbfixer_LCH.txt`.
- `conda install -n filterzyme -c conda-forge pdbfixer -y` (needed by docko; not on PyPI).
- `pip install -U "regex>=2025.10.22"` (enzymetk needs newer; broke docko's `regex==2024.9.11` pin — pin appears over-restrictive, not a real conflict).
- All 6 key imports OK after this.

### Step 6 — Baseline snapshot of lab repo
- `git -C /mnt/labs/data/mora/code/Filterzyme rev-parse HEAD` = `d3bd86783cbf9010008a766891beba759b99ad76` (msg: "Updated readme", 2026-06-25 15:20).
- `git status --short`: 1 modified file `benchmarking/PDE_2H/run_PDE_2H.py`, many untracked dirs (af3/, openbabel-3.1.1/, boltz_cache/, ...).
- `git diff` captured (20 lines) in `$HOME/filterzyme-sanity/baseline_LCH.diff`.

### Step 7 — Boltz cache pre-flight
- `/mnt/labs/data/mora/code/Filterzyme/boltz_cache` = 9.1 GB, contains `boltz2_aff.ckpt` (2.1 GB), `boltz2_conf.ckpt` (2.3 GB), `mols.tar` (1.9 GB), `mols/` (1.1 MB dir). Healthy.
- Captured in `$HOME/filterzyme-sanity/boltz_cache_state_LCH.txt`.

## 2026-06-26 — Task 01 (workspace)

- `$HOME/filterzyme-sanity/PDE_2H_LCH/` created.
- Copied `run_PDE_2H.py` and `PDE_data_formatted.csv` from lab tree. CSV: 240 data rows.
- `sed`-edited the copy: `base_output_dir = "filterzyme_output"` → `"filterzyme_output_LCH"`. Verified at line 30.
- sha256 recorded in `run_inputs_LCH.sha256`. Lab tree untouched.

## 2026-06-26 — Task 02 (smoke run, iteration 1: openbabel)

- Ran `python -u run_PDE_2H_LCH.py`. Failed at import time:
  - `ModuleNotFoundError: No module named 'openbabel'` in `computeproteinRMSD_step.py:17`.
- Fix: `conda install -n filterzyme -c conda-forge openbabel -y`. Got 3.1.0.

## 2026-06-26 — Task 02 (iteration 2: plip)

- Re-ran. New failure:
  - `ModuleNotFoundError: No module named 'plip'` in `plip_step.py:6`.
- Static scan of all filterzyme imports inside the env identified `plip` and `esm` as the only remaining missing modules. `esm` is for Squidly (skipped), so only plip is real.
- Fix: `conda install -n filterzyme -c conda-forge plip -y`.

## 2026-06-26 — Task 02 (iteration 3: NumPy doubling, NameError)

- Re-ran. Passed import; reached Stage A.1. Chai docking failed at runtime:
  - `NameError: name 'run_chai' is not defined` in `enzymetk/dock_chai_step.py:50`.
- Cause: top-of-file `from docko.chai import run_chai` was silenced by a try/except printing `Numba needs NumPy 2.1 or less. Got NumPy 2.4`.
- Diagnosis: env had TWO numpys side-by-side. `pip show numpy` reported 1.26.4, but conda's pdbfixer/openbabel/plip installs had also dropped numpy 2.4.6 into `site-packages`. Numba refused.
- Fix: `pip install --force-reinstall "numpy>=2.0,<2.2"` → numpy 2.1.x, single copy.

## 2026-06-26 — Task 02 (iteration 4: Boltz arg mismatch — CURRENT)

- Re-ran. Chai docking **completed** (9.7 GB GPU mem, clean exit). Pipeline entered Stage A.2 Boltz, immediately failed:
  - `TypeError: run_boltz_affinity() takes 5 positional arguments but 6 were given` in `enzymetk/dock_boltz_step.py:55`.
- Inspected:
  - enzymetk passes 6 args: `(run_id, seq, substrate, output_dir, intermediate, self.args)` where `self.args = ['--cache', boltz_cache_path, ...]`.
  - PyPI `docko==0.1.3` signature: `def run_boltz_affinity(label, seq, smiles, output_dir, cofactor_smiles)` — 5 args.
  - **Lab repo** `/mnt/labs/data/mora/code/docko/docko/boltz.py:107` has 6-arg signature with `args=[...]` default. That's the version Ariane actually runs.
- Pending fix (next step): `pip install --no-deps -e /mnt/labs/data/mora/code/docko` to replace PyPI docko with editable lab docko.
- Status: paused here to write these journal documents per user request.

## Pending tasks

- 02 (continue): apply lab-docko editable install, re-run pipeline.
- 03 (characterize): build banner-sequence, traceback, file inventory.
- 04 (report): fill `08_report_template.md`.

## 2026-06-26 — Task 02 (iteration 5: docko version swap → BIG PROGRESS)

- `pip install --no-deps -e /mnt/labs/data/mora/code/docko` failed: pip needs to write `docko.egg-info/` inside Ariane's tree (no permission).
- Workaround: copied lab docko to `$HOME/docko_lab_LCH`, removed stale `egg-info/build/dist`, then `pip install --no-deps -e $HOME/docko_lab_LCH`.
- Result: docko 0.1.5 installed (PyPI's latest is 0.1.3 — lab has unreleased version). 6-arg `run_boltz_affinity` confirmed.
- Trade-off vs Ariane's setup: hers may be a live editable on the lab tree; ours is a one-time copy. Re-copy if she updates docko.

## 2026-06-26 — Task 02 (iteration 6: full smoke run, end-to-end-ish)

Pipeline ran through ALL of Stage A and most of Stage B. Stages cleared:

- ✓ Stage A.1 Chai docking (9.7 GB GPU mem)
- ✓ Stage A.2 Boltz docking (with non-fatal `cuequivariance_ops_torch` warnings — fell back to standard triangle multiplicative update)
- ✓ Stage A.3 Vina docking (via Chai-predicted structures, as configured)
- ✓ Stage A.4 Extract docking quality metrics
- ✓ Stage B.1 Superimposing docked structures
- ✓ Stage B.2 **Calculating protein RMSDs** ← past Ariane's predicted break point
- ✗ Stage B.3 Calculating ligand RMSDs — broke here

**Break:** `KeyError: "Expected column 'docked_structure' not found."` raised at `filterzyme/utils/helpers.py:319` in `extract_docking_metrics`, called from `pipeline_v2.py:318` inside `_ligandRMSD`. Pre-error log: `Error selecting best docked structures: 'Entry'` (suggests an earlier silent column-key mismatch on `Entry` that resulted in `docked_structure` never being added).

This is a NEW finding, not in `1_analysis-planning/05_bugs_by_severity.md`. Likely a column-rename / schema-drift bug in `extract_docking_metrics` or its upstream `select_best_docked_structures`.

### Soft warnings observed (non-fatal)

- Boltz emitted import errors for `cuequivariance_ops_torch` (CUDA 12 wheels are missing for cu128) and fell back to a slower-but-working path. Not a blocker but worth a future optimisation.

### Defensive snapshot saved at this milestone

```
$HOME/filterzyme-sanity/PDE_2H_LCH/sanity-run_LCH_first-success.log   # copy of the working log
$HOME/filterzyme-sanity/env_working_LCH.txt                          # pip freeze
$HOME/filterzyme-sanity/conda_working_LCH.txt                        # conda list --explicit
$HOME/filterzyme-sanity/produced-files_LCH.txt                       # file inventory of filterzyme_output_LCH
```

(Run these now if not already done — see chat for exact commands.)

## Updated pending tasks

- 03 (characterize): full traceback + file inventory + verdict table for the new ligandRMSD break.
- 04 (report): fill `08_report_template.md`.
- Optional follow-up: small patch to `extract_docking_metrics` to complete ligandRMSD and reach Stage C (GeometricFilters). Out of scope for the original plan but tempting given how close we are.

## 2026-06-26 — Cluster meta-setup (git + GitHub CLI)

Not part of the pipeline itself, but required to push results from `login02`:

- **Git identity** (one-time, global):
  ```
  git config --global user.name "Luca Herrmann"
  git config --global user.email "luca.herrmann98@gmail.com"
  ```
- **GitHub CLI** is available as a module — not loaded by default:
  ```
  module load gh/2.92.0
  gh auth login        # web-browser device-code flow
  ```
  After `gh auth login` succeeds, `git push` to HTTPS remotes works without a PAT prompt.
- **Branching:** new work branches off `origin/pbp` (Ariane's latest, 2026-06-25), not `main` (Nov 2025). `main` is months behind the lab's actual state.
- **.gitignore:** added `.kilo/` and `.kilo-tmp/` so the local agent state isn't committed.

Current branch: `lab-sanity-run_LCH` (one commit ahead of `origin/pbp`), pushed to GitHub.
