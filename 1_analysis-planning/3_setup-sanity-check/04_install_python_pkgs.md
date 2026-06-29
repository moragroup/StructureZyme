# Task 04 — Install torch (CUDA), filterzyme, and verify enzymetk/docko/chai/boltz

**Depends on:** Task 03 (env + binaries). **Owner unit:** standalone.

## Why / ordering rule

`enzymetk`, `docko`, `chai_lab`, and `boltz` all pull `torch`. **Install torch FIRST, matched to the cluster CUDA** (from Task 01's `nvidia-smi`), so pip doesn't grab a mismatched CPU/CUDA wheel.

## Step 1 — torch matched to CUDA

Pick the index URL matching the CUDA version recorded in Task 01:

```bash
conda activate filterpipeline
# Example for CUDA 12.1; swap cu121 -> cu118 / cu124 as appropriate:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

`torch.cuda.is_available()` MUST be `True` inside the GPU session. If `False`, the wheel/CUDA mismatch or the session lacks the GPU — stop and fix before continuing.

## Step 2 — install the repo (editable)

```bash
cd /mnt/storage01/home/lherrmann/StructureZyme
pip install -e .
```

This installs `filterzyme` 0.0.6 and pulls its `install_requires` (pandas, numpy, tqdm, biopython, biotite, matplotlib, seaborn, rdkit, freesasa, plip, enzymetk, docko). The `environment.yml` pip section already requested `enzymetk`, `docko`, `chai_lab` — confirm versions.

## Step 3 — watch for the openmm conflict (from Task 03)

When `docko` resolves, it may try to change `openmm` (it pins 8.1.1; env has 8.3.1). Capture pip's resolution output. If docko import later fails, see Task 03's openmm fallback.

## Step 4 — confirm the heavy backends import

```bash
python - <<'PY'
import importlib
for m in ["enzymetk", "docko", "docko.docko", "chai_lab", "boltz"]:
    try:
        importlib.import_module(m)
        print("OK  ", m)
    except Exception as e:
        print("FAIL", m, "->", type(e).__name__, e)
PY
```

`boltz` may be absent (docko lists it optional). Record which import.

## Success criterion

`torch.cuda.is_available()` is True; `pip install -e .` completes; `import filterzyme`, `enzymetk`, `docko.docko`, `chai_lab` all succeed (note any failures + the openmm version actually installed).
