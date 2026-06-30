# 10. Squidly CLI Install (verified 2026-06-30)

This file records how the `squidly` CLI was installed into the `filterzyme`
conda env on this machine. It is the canonical reference for reproducing the
setup on a new host.

## What Squidly is

Squidly is an external CLI (not a Python library call) that predicts catalytic
residues using ESM2 embeddings + an LSTM ensemble. It is invoked by
`enzymetk.ActiveSitePred` via `subprocess`:

```
squidly run <input.fasta> <esm2_model> <tmp_dir> [args...]
```

Filterzyme's `filterzyme/steps/squidly_step.py` wraps `ActiveSitePred` and
normalizes its output columns. The CLI must be on `$PATH` in the same conda
env that runs Filterzyme.

## Install steps

### 1. The `squidly` Python package (provides the CLI)

Already installed in the `filterzyme` env:

```
pip install squidly
```

Verified version on this host:

```
$ pip show squidly
Name: squidly
Version: 0.1.0
Home-page: https://github.com/WRiegs/Squidly
Location: /mnt/storage01/home/lherrmann/envs/filterzyme/lib/python3.11/site-packages
Requires: biopython, enzymetk, fair-esm, huggingface_hub, numpy, pandas, psutil, sciutil, tqdm, typer
```

The binary lives at:
```
/mnt/storage01/home/lherrmann/envs/filterzyme/bin/squidly
```

### 2. Download the model weights

The CLI does **not** ship weights in the pip package. Download them from
HuggingFace (`WillRieger/Squidly`) into the installed package directory:

```bash
python /mnt/storage01/home/lherrmann/envs/filterzyme/lib/python3.11/site-packages/squidly/download_models_hf.py
```

This creates:

```
site-packages/squidly/models/3B/
site-packages/squidly/models/15B/
```

Each subfolder contains the LSTM/CL weight files for that ESM2 backbone size.

### 3. ESM2 backbone weights (auto-cached on first run)

The first Squidly run downloads the ESM2 backbone (~5.7 GB for 3B) into:

```
~/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt
```

Subsequent runs reuse the cache. No manual action needed.

## Verification

### `squidly --help`

```
Usage: squidly [OPTIONS] COMMAND [ARGS]...

Commands:
  install  Install the models for the package.
  run      Find catalytic residues using Squidly and BLAST.
```

### `squidly run --help` (key options)

```
Usage: squidly run [OPTIONS] FASTA_FILE ESM2_MODEL [OUTPUT_FOLDER] [RUN_NAME]

Arguments:
  fasta_file         Full path to query fasta (required)
  esm2_model         esm2_t36_3B_UR50D or esm2_t48_15B_UR50D (required)
  output_folder      Where to store results (default: cwd)
  run_name           Name of the run (default: squidly)

Options:
  --single-model / --no-single-model   Use single model instead of ensemble (default: no)
  --cpu / --no-cpu                     CPU mode (ensemble only, default: no)
  --iterative / --no-iterative         Run ensemble models sequentially (default: no)
  --model-folder TEXT                  Full path to the model folder
  --database TEXT                      Full path to database csv (Entry,Sequence,Residue)
  --as-threshold FLOAT                 Single-model prediction threshold (default: 0.95)
  --blast-threshold FLOAT              Sequence identity for BLAST vs Squidly (default: 0.3)
  --chunk INTEGER                      Max chunk size for large datasets (default: 0)
  --mean-prob FLOAT                    Mean probability threshold (ensemble, default: 0.6)
  --mean-var FLOAT                     Mean variance cutoff (ensemble, default: 0.225)
  --filter-blast / --no-filter-blast   Only run on BLAST-miss sequences (default: yes)
```

### Smoke test (run on a GPU node)

```bash
conda activate filterzyme
python -m pytest tests/test_squidly_step.py::test_squidly_execute_smoke \
                 tests/test_squidly_step.py::test_squidly_execute_dedup_broadcast -v -s
```

Verified result on 2026-06-30:

```
tests/test_squidly_step.py::test_squidly_execute_smoke PASSED
tests/test_squidly_step.py::test_squidly_execute_dedup_broadcast PASSED
2 passed, 7 warnings in 39.09s
```

The test ran the ensemble (5 models) on one ~120 aa sequence in ~0.28 min.
ESM2 3B inference requires a GPU; on a CPU-only node the tests are skipped
automatically.

## Output pickle schema

`enzymetk.ActiveSitePred` reads `<tmp_dir>/squidly_ensemble.pkl`, which the
CLI writes with these columns (verified in
`site-packages/squidly/__main__.py:327-406`):

| Column | Description |
|--------|-------------|
| `label` | Sequence id (copied from the fasta header) |
| `Squidly_Ensemble_Residues` | Pipe-delimited 1-indexed catalytic residue positions |
| `Squidly_CR_Position` | Mirror of `Squidly_Ensemble_Residues` (ensemble path only) |
| `Sequence` | The input sequence |
| `mean` | Ensemble mean probability |
| `entropy` | Ensemble entropy |
| `variance` | Epistemic uncertainty |
| `all_AS_probs_{1..5}` | Per-model per-residue probabilities |

`filterzyme/steps/squidly_step.py` renames `label` -> `Entry` and
`Squidly_Ensemble_Residues` -> `Squidly_CR_Position` so `pipeline_v2.py`
gets the canonical column it expects.

## Troubleshooting

- **`RuntimeError: The squidly CLI was not found on $PATH`** -- activate the
  `filterzyme` env (`conda activate filterzyme`) so `bin/squidly` is on PATH.
- **`ERROR: The model folder does not exist`** -- run step 2 above to download
  weights.
- **Hangs on a login node** -- ESM2 inference needs a GPU. Run on a compute
  node, or use `--cpu` (very slow, not recommended).
- **`conda run -n enzymetk squidly ...` fails** -- the upstream
  `ActiveSitePred` defaults to `env_name='enzymetk'`. The Filterzyme wrapper
  passes `env_name=None` so the CLI runs in the current env. Do not change
  this unless squidly is also installed in the `enzymetk` env.
