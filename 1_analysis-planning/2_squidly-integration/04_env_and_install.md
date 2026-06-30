# 4. Environment and Install (enzymetk-backed)

## Where Squidly lives

Squidly runs as an **external CLI** (`squidly run ...`) invoked by
`enzymetk.ActiveSitePred` via `subprocess`. So the install task is twofold:

1. Make sure `enzymetk` is installed (already done; it ships `ActiveSitePred`).
2. Make sure the `squidly` CLI itself is installed and on `$PATH` in the same conda env.

**Today only (1) is done on this machine. (2) is missing.** See
`09_phase0_squidly_install_and_schema.md` for the fix.

## Conda env

The existing `filterzyme` conda env (see `environment.yml`) is the target. Heavy deps
(`torch`, ESM2 backbone, transformers) are already pulled by `chai_lab` and `enzymetk`,
so adding Squidly should not require a new env.

Add to `environment.yml` pip section:

```
- squidly>=<TBD>      # exact pin set in Phase 0 once the source is located
```

If the upstream `squidly` is git-only:

```
- git+https://<URL>/Squidly.git@<commit-or-tag>
```

## Model weights

The `squidly` CLI ships and locates its own LSTM weights (this is the whole point of
using it - we stop owning the `.pth` files). ESM2 backbone weights are downloaded by
the CLI on first run, cached under `~/.cache/torch/hub/checkpoints/` or wherever
`transformers` puts them.

The two `.pth` files currently in `filterzyme/squidly_final_models/` become dead
weight and are removed in Phase D.

## GPU vs CPU

- Selection: set `CUDA_VISIBLE_DEVICES` in the shell before launching Python. There is
  **no Python-level device kwarg** in `enzymetk.ActiveSitePred`. This is the same way
  Chai and Boltz pick devices in pipeline_v2 today.
- Memory: 3B model fits on ~12 GB VRAM, 15B needs ~40 GB. Default to `model_size="3B"`
  in the `Squidly` wrapper.
- CPU fallback: depends on the CLI. Worth verifying in Phase 0 step 0.3 with
  `CUDA_VISIBLE_DEVICES=""` set.

## Install verification (run after Phase 0)

In `3_setup-sanity-check/10_squidly_install.md` (to be created):

```
conda activate filterzyme
which squidly
squidly --help
python -c "
import pandas as pd
from filterzyme.steps.squidly_step import Squidly
df = pd.DataFrame({
    'Entry':['t'],
    'Sequence':['MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAATGFHI'],
})
print(Squidly(model_size='3B', num_threads=1).execute(df))
"
```

Should print a DataFrame whose `catalytic_residues` cell is a non-empty pipe-delimited
string (or explicitly empty for a non-catalytic sequence) and finish in under 2 minutes.
