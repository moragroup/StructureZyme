# Task 07 — Staged end-to-end run on the example

**Depends on:** Task 05 + Task 06. **Owner unit:** the core sanity check.

## Why

Run the three pipeline stages **separately** (not via `Pipeline.run()`) so you can see exactly which stage and which model breaks, and capture intermediate pickles. The full `Pipeline` chains Docking → Superimposition → GeometricFilters and aborts on the first error.

Input: `examples/DEHP-MEHP.pkl`. Work in a scratch output dir.

## Setup

```bash
conda activate filterpipeline
cd /mnt/storage01/home/lherrmann/StructureZyme
export OUT=/tmp/kilo/sanity_out          # or a scratch path with space
mkdir -p "$OUT"
```

## Stage A — Docking (Squidly → Chai → Boltz → Vina)

This is the heaviest stage and the most likely to break (GPU, enzymetk env, model downloads).

```bash
python - <<'PY'
import os, pandas as pd
from importlib.resources import files
from filterzyme.pipeline import Docking
df = pd.read_pickle("examples/DEHP-MEHP.pkl")
out = os.environ["OUT"]
d = Docking(
    df=df,
    output_dir=f"{out}/docking",
    squidly_dir=str(files("filterzyme") / "squidly_final_models"),
    metagenomic_enzymes=0,
    skip_catalytic_residue_prediction=True,   # set False to also test Squidly
    alternative_structure_for_vina="Chai",
    num_threads=1,
)
d.run()
print("DOCKING DONE")
PY
```

Notes / expected failure points:
- If `skip_catalytic_residue_prediction=True`, the df must already carry `catalytic_residues`/`vina_residues`; if not, Vina has no residues. If the example lacks them, set `skip_...=False` to let Squidly populate them (needs Task 05 resolved + ESM2 download).
- **P1-3 bug**: `dock_vina_step.py:35-36` `int(r)+1` crashes on empty residue strings — fires if any entry has no residues.
- RTX6000 24 GB may OOM on Chai/Boltz for large sequences — note if it does; retry on `h100`.
- Chai/Boltz download weights on first use (multi-GB, needs node network egress).

Outputs to expect under `$OUT/docking/`: `squidly.pkl` (if not skipped), `chai.pkl`, `boltz.pkl`, `vina.pkl`, `dockingmetrics.pkl`.

## Stage B — Superimposition

```bash
python - <<'PY'
import os
from filterzyme.pipeline import Superimposition
out = os.environ["OUT"]
s = Superimposition(maxMatches=1000, input_dir=f"{out}/docking",
                    output_dir=f"{out}/superimposition", num_threads=1)
s.run()
print("SUPERIMPOSITION DONE")
PY
```

Reads `dockingmetrics.pkl`; produces `superimposedstructures.pkl`, `proteinRMSD.pkl`, `ligandRMSD.pkl`.
**Watch P0-3**: `_ligandRMSD` calls `extract_docking_metrics` (the working one) — but if any path routes through `helpers.add_metrics`, it raises NameError.

## Stage C — GeometricFilters (fpocket / PLIP / FreeSASA)

```bash
python - <<'PY'
import os, pandas as pd
from filterzyme.pipeline import GeometricFilters
out = os.environ["OUT"]
df = pd.read_pickle(f"{out}/superimposition/ligandRMSD.pkl")
g = GeometricFilters(df=df, esterase=0, input_dir=f"{out}/superimposition",
                     output_dir=f"{out}/geometricfiltering", num_threads=1)
g.run()
print("GEOMETRIC FILTERING DONE -> structural_features_final.pkl")
PY
```

Needs `fpocket`, `plip`, `freesasa`, `openbabel` on PATH (Task 03). This stage is CPU-only and the most likely to fully succeed.

## Capture everything

Run each stage with `2>&1 | tee $OUT/stageA.log` (etc.). For every break, record: stage, model, exception type + message, and the last successful pickle. Feed into Task 08.

## Success criterion

Each stage either completes or fails with a captured, attributed error. Best outcome: `$OUT/geometricfiltering/structural_features_final.pkl` exists. Either way, the break-map is documented.
