# Getting Started

StructureZyme is a modular pipeline for enzyme structure/function prediction.
This guide covers installing the environment, the one-time model-weight setup,
and the command-line workflow.

## Environment

StructureZyme runs in the `structurezyme` conda environment defined by
`environment.yml`. Create it and install the package itself:

```bash
conda env create -f environment.yml
conda activate structurezyme
pip install -e .
```

The environment pins Python 3.11 and the pipeline's scientific dependencies
(e.g. `foldseek`, `mmseqs2`, `openmm`, `fpocket`, `plip`) plus the pip packages
`squidly`, `chai_lab`, `docko`, `pydantic`, `pyyaml`, and `filelock`. Vina
docking additionally uses `meeko` (provides `mk_prepare_ligand.py`), the `vina`
Python bindings, and the `vina` command-line binary.

**GPU / CUDA note.** `environment.yml` pins `torch==2.11.0+cu128` (CUDA 12.8)
from the PyTorch cu128 wheel index. This is required for NVIDIA Blackwell GPUs
(e.g. RTX PRO 6000, `sm_120`) present on this cluster; the default cu124 build
that `chai_lab`/`boltz` would otherwise pull in fails on Blackwell with
`CUDA error: no kernel image is available for execution on the device`. See
[known-issues.md](known-issues.md) for details (and the harmless
`chai-lab requires torch<2.7` pip warning).

## CLI workflow

The `structurezyme` CLI drives runs. A typical session:

```bash
# 1. Write a template config, then edit its paths/steps
structurezyme init --output run.yml

# 2. Run the full pipeline (host defaults fill in site-specific paths)
structurezyme run --config run.yml --host default

# 3. Check progress (per-step status + wall time)
structurezyme status --run-dir <output_root>/<user>/<run_id>

# 4. Resume after a crash or a config change
structurezyme resume --run-dir <output_root>/<user>/<run_id>
```

Runs are resumable: completed steps with matching config/input hashes are
skipped on re-run. To re-run a single module use
`structurezyme step <name> --run-dir <dir>` (add `--continue` to also recompute
downstream steps). See [`api_reference.md`](api_reference.md) for the full
subcommand and module reference.

## Catalytic-residue model weights (Squidly)

StructureZyme does **not** manage model weights itself. Catalytic-residue
prediction shells out to the `squidly` CLI, which must be available on your
`$PATH`. The `squidly` package (installed as part of `environment.yml`, pinned
to a patched fork) ships and downloads its own model ensemble from the
HuggingFace repo `WillRieger/Squidly`; StructureZyme never bundles or loads
`.pth` files directly.

`squidly` is already installed by `environment.yml` — no separate
`pip install squidly` step is needed (doing so would pull the unpatched
upstream package and reintroduce the empty-residue bug; see
[known-issues.md](known-issues.md)). Run the following once, after creating
the `structurezyme` environment, to download the model weights:

```bash
# The Squidly ensemble weights are git-ignored in the Squidly repo and are NOT
# downloaded automatically by squidly or enzymetk. Download them once after
# creating the env, using Squidly's own helper (pulls HuggingFace repo
# WillRieger/Squidly into <site-packages>/squidly/models):
python -m squidly.download_models_hf

# Verify the ensemble landed in site-packages/squidly/models/{3B,15B}/:
python -c "import squidly, os, glob; d=os.path.join(os.path.dirname(squidly.__file__),'models'); print(sorted(glob.glob(d+'/*/*.p*')))"
```

If the weights are missing, the first pipeline step fails with
`ERROR: The model folder does not exist: <site-packages>/squidly/models`.

Notes:

- StructureZyme requires the `squidly` CLI on `$PATH` and does not manage
  weights itself. If the CLI is missing, the Squidly step fails loudly rather
  than silently falling back to local weights.
- ESM2 inference (used by Squidly) needs a GPU.

## First-run model assets (downloads)

Besides the Python packages pinned in `environment.yml`, the pipeline needs
several large model-asset downloads on first use. Only the **Squidly ensemble
weights** require a manual step; the rest download automatically the first time
the relevant step runs (they just need network access, and can be slow):

| Asset | Size | How it's obtained | Manual step? |
| --- | --- | --- | --- |
| Squidly ensemble weights (`CataloDB_*`) | ~421 MB | `python -m squidly.download_models_hf` | **Yes** (see above) |
| ESM2 base model (`esm2_t36_3B_UR50D.pt`) | ~5.6 GB | `fair-esm` auto-downloads to `~/.cache/torch/hub/checkpoints` | No |
| Chai inference assets (conformers, model weights, traced ESM2) | ~6.5 GB | `chai_lab` auto-downloads to `<site-packages>/downloads` (override with `CHAI_DOWNLOADS_DIR`) | No |
| Boltz cache | varies | `boltz` downloads on first run; pass `--boltz-cache <dir>` (smoke harness) | No |

Notes for shared/lab installs:

- To avoid every user re-downloading ~12 GB, point the auto-downloaders at a
  shared copy: set `CHAI_DOWNLOADS_DIR` to a shared Chai assets dir, reuse a
  shared `~/.cache/torch/hub` for the ESM2 base model, and pass a shared Boltz
  cache dir.
- The Chai CDN (`chaiassets.com`) occasionally drops large-file downloads
  mid-transfer (`ChunkedEncodingError: Connection broken`). If this happens,
  re-run the step, or pre-seed `<site-packages>/downloads` (or
  `CHAI_DOWNLOADS_DIR`) from a known-good copy.

## Smoke test (GPU, end-to-end)

`tools/smoke/run_phase_c_smoke.sbatch` runs the full chain
(Squidly → Chai → Boltz → Vina → docking metrics) on a single test enzyme
(CalB / UniProt P41365). Submit it from the repo root:

```bash
# Override the env path / partition if needed via STRUCTUREZYME_ENV, BOLTZ_CACHE.
sbatch tools/smoke/run_phase_c_smoke.sbatch
```

On success it writes `.../geometricfiltering/structural_features_final.pkl`
alongside per-step outputs (`docking/squidly.pkl`, `docking/boltz.pkl`,
`docking/vina/<Entry>/*.pdb`).

### Optional steps: FastRelax and PLACER

Both are disabled by default and depend on shared installs (not in the conda
env). To run the full 15-step DAG:

1. Ensure the shared installs are reachable:
   - FastRelax: `/mnt/labs/data/mora/software/RosettaFastRelax/env`
     (or set `FASTRELAX_ENV`).
   - PLACER: `/mnt/labs/data/mora/software/PLACER/env` (host-profile
     `placer_env_path`).
2. In your run config, set `steps.fastrelax.enabled: true` and
   `steps.placer.enabled: true` (with `steps.placer.placer_predict_ligand`).
   See `tools/gpu_run/run.yml` for a full-DAG example.
3. Smoke it on GPU:
   ```bash
   mkdir -p smoke_logs
   sbatch --output="$PWD/smoke_logs/%x_%j.out" --error="$PWD/smoke_logs/%x_%j.err" tools/smoke/run_full_dag_smoke.sbatch
   ```
