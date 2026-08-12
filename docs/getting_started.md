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
`squidly`, `chai_lab`, `docko`, `pydantic`, `pyyaml`, and `filelock`.

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
# Weights are shipped and downloaded by the `squidly` package itself
# (HuggingFace repo WillRieger/Squidly). Run once after creating the env:
python -c "import squidly, os; os.system(f'python {os.path.dirname(squidly.__file__)}/download_models_hf.py')"

# Verify the ensemble landed in site-packages/squidly/models/{3B,15B}/:
python -c "import squidly, os, glob; d=os.path.join(os.path.dirname(squidly.__file__),'models'); print(sorted(glob.glob(d+'/*/*.p*')))"
```

Notes:

- StructureZyme requires the `squidly` CLI on `$PATH` and does not manage
  weights itself. If the CLI is missing, the Squidly step fails loudly rather
  than silently falling back to local weights.
- ESM2 inference (used by Squidly) needs a GPU.
