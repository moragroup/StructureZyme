# Task 06 — Import-level sanity checks

**Depends on:** Task 04. **Owner unit:** standalone (can run in parallel with Task 05).

## Why

Before any GPU-heavy run, confirm the package and every step module import. Cheap, fast, and isolates code-level breakage (typos, bad imports) from runtime/model breakage.

## Step 1 — package import + version

```bash
conda activate filterpipeline
cd /mnt/storage01/home/lherrmann/StructureZyme
python -c "import filterzyme; print('filterzyme', filterzyme.__version__)"   # expect 0.0.6
```

## Step 2 — import every step module, report pass/fail individually

```bash
python - <<'PY'
import importlib, pkgutil, filterzyme.steps as steps
mods = [m.name for m in pkgutil.iter_modules(steps.__path__)]
for m in sorted(mods):
    full = f"filterzyme.steps.{m}"
    try:
        importlib.import_module(full)
        print("OK  ", full)
    except Exception as e:
        print("FAIL", full, "->", type(e).__name__, e)
# pipeline + helpers
for full in ["filterzyme.pipeline", "filterzyme.utils.helpers"]:
    try:
        importlib.import_module(full); print("OK  ", full)
    except Exception as e:
        print("FAIL", full, "->", type(e).__name__, e)
PY
```

## Expected known failures (do NOT fix; just record)

- `filterzyme.steps.PLACER_step` — line 8 `from step import Step` is a broken bare import (dead code). Expected `ModuleNotFoundError: No module named 'step'`. The PLACER steps are not used by `pipeline.py`, so this does not block the run.
- Importing `filterzyme.utils.helpers` itself should succeed; the bug `add_metrics` (NameError on `dict_columns`/`extract_vina_index`) only fires when the function is *called*, not on import.

## Step 3 — confirm the example input is well-formed

```bash
python - <<'PY'
import pandas as pd
df = pd.read_pickle("examples/DEHP-MEHP.pkl")
print("rows:", len(df))
print("cols:", list(df.columns))
req = ["Entry","Sequence","substrate_name","substrate_smiles","substrate_moiety"]
print("missing required:", [c for c in req if c not in df.columns])
print(df.head().to_string())
PY
```

Record the columns present vs the README contract (required: Entry, Sequence, substrate_name, substrate_smiles, substrate_moiety; optional cofactor_*/vina_residues). This tells Task 07 whether Squidly can be skipped (i.e. whether `vina_residues` is present).

## Success criterion

`import filterzyme` prints 0.0.6; the per-module table is captured (PLACER_step failure is the only expected one); the example df columns are documented.
