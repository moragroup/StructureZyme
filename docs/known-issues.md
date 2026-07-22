# Known Issues

Bugs in upstream dependencies that StructureZyme currently works around. Each
entry documents the bug, our workaround, and how to remove the workaround once
upstream is fixed.

## Squidly CLI drops `--mean-prob` / `--mean-var` before subprocess

- **Upstream repo**: <https://github.com/WRiegs/Squidly>
- **Affected version**: `squidly==0.1.0` (any version where
  `squidly/__main__.py :: run` builds the inner `python squidly.py ...` command
  without appending `--mean_prob` / `--mean_var`).
- **Our workaround**: `structurezyme/steps/squidly_step.py :: Squidly.execute`
  (search the file for `TODO(squidly-upstream)`).
- **Regression test**: `tests/test_squidly_threshold_local_apply.py`.

### Symptom

Running `squidly run <fasta> <esm2_model> <output> <name> --mean-prob 0.03 --mean-var 0.5`
produces a `*_ensemble.pkl` whose `Squidly_CR_Position` column was filtered
using the argparse defaults `0.6 / 0.225`, not the values the user passed on
the CLI. For proteins whose ensemble probabilities never exceed 0.6 (e.g.
flavin monooxygenases without a canonical Cys-His-His triad) this yields an
empty residue string for every enzyme.

### Root cause

In `squidly/__main__.py` the `run` command declares:

```python
mean_prob: Annotated[float, typer.Option(...)] = 0.6,
mean_var:  Annotated[float, typer.Option(...)] = 0.225,
```

and then in all four branches of the ensemble path builds a subprocess command:

```python
cmd = ['python', os.path.join(pckage_dir, 'squidly.py'),
       fasta_file, esm2_model, os.path.join(model_folder, esm2_model_dir),
       output_folder, ...maybe --cpu / --single_model...]
```

The `--mean_prob` / `--mean_var` flags accepted by the inner worker
(`squidly/squidly.py :: create_parser`) are never appended to `cmd`, so the
worker always receives its argparse defaults. The outer typer options are
effectively ignored for the ensemble code path.

### Verifying the bug

```bash
# In an env with squidly installed:
squidly run seqs.fasta esm2_t36_3B_UR50D out/ run1 --mean-prob 0.03 --mean-var 0.5
# Then:
python -c "
import pandas as pd
df = pd.read_pickle('out/run1_ensemble.pkl')
print(df['Squidly_CR_Position'].tolist())
# -> filtered at 0.6/0.225, not 0.03/0.5
"
```

The `mean` and `variance` array columns in the same pickle are populated
correctly, which is what our workaround exploits.

### Our workaround

In `Squidly.execute` we detect when the user supplied a non-None `mean_prob`
or `mean_var`, and recompute `Squidly_CR_Position` locally from the ensemble
`mean` and `variance` arrays using the exact rule upstream applies internally
(`squidly/squidly.py :: compute_uncertainties`):

```python
picks = np.where((mean > mean_prob) & (variance < mean_var))[0]
Squidly_CR_Position = "|".join(str(int(p)) for p in picks)
```

If the user leaves both thresholds at `None`, the workaround is a no-op and
upstream's own (buggy but default-consistent) selection is kept.

### Removing this workaround

1. Confirm upstream `squidly` version has been patched: the four `cmd = [...]`
   builds in `squidly/__main__.py :: run` must include
   `'--mean_prob', str(mean_prob), '--mean_var', str(mean_var)`.
2. Bump the required `squidly` version in `environment.yml` / `setup.py`.
3. Delete the block marked `TODO(squidly-upstream)` in
   `structurezyme/steps/squidly_step.py :: Squidly.execute`.
4. Delete `_select_residues_from_ensemble` in the same file if no other caller
   remains.
5. Keep `tests/test_squidly_threshold_local_apply.py` but adapt it to invoke
   the upstream CLI end-to-end as a smoke test that the fix has landed.
