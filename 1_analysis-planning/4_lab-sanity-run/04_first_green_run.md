# 04 — First green end-to-end run

**Date:** 2026-06-29  
**Status:** Pipeline completes (exit 0) with 9 rows × 67 cols final output.

## What was broken

After the protein-RMSD stage that Ariane predicted as the break point passed cleanly,
two further blockers surfaced:

1. **`LigandRMSD` silent empty-df bug** (`computeligandRMSD_step.py:495`)
2. **Missing `fpocket` binary** in the env (Fpocket step on the active-site-volume stage)

## Bug 1 — LigandRMSD empty `best_structures`

### Symptom
```
Error selecting best docked structures: 'Entry'
KeyError: 'Entry'
  at computeligandRMSD_step.py:495
  .groupby(['Entry', 'best_structure'])['method']
```

The original `select_best_docked_structures` returns `pd.DataFrame(best_structures)`.
When `best_structures` is empty (all entries had only one tool, all RMSDs NaN, or
empty SMILES), this yields a **DataFrame with zero columns**. The subsequent
`groupby(['Entry', 'best_structure'])` at line 495 then raises `KeyError: 'Entry'`
because the column doesn't exist.

A surrounding `except Exception` at line 553 silently swallowed this and returned
`(rmsd_df, pd.DataFrame())` — producing an empty `ligandRMSD.pkl` and breaking
downstream `extract_docking_metrics` with `KeyError: 'docked_structure'`.

### Fix
Guarantee the returned DataFrame always has the expected schema:

```python
# in select_best_docked_structures, before `return pd.DataFrame(best_structures)`
expected_cols = ['Entry', 'tool', 'best_structure', 'avg_ligandRMSD', 'method']
if not best_structures:
    return pd.DataFrame(columns=expected_cols)
return pd.DataFrame(best_structures)
```

Patch saved at `patches/01_ligandRMSD_empty_best_structures_LCH.patch` (unified
diff vs lab tree at sha d3bd867).

### Diagnostic approach
Replaced the silent `except` with `traceback.print_exc(); raise` to surface the
real line and stack. After the fix verified, restored the original except handler.

## Bug 2 — Missing fpocket

### Symptom
```
FileNotFoundError: [Errno 2] No such file or directory: 'fpocket'
  at fpocket_step.py:274 (subprocess.run)
```

`fpocket` is a system binary required by the GeometricFilters stage. Not on PATH,
not in the env, not pip-installable. README didn't mention it.

### Fix
```
conda install -n filterzyme -c conda-forge fpocket -y
```
Installed fpocket 4.2.2 → `~/envs/filterzyme/bin/fpocket`. Snapshot of pre-state
saved at `~/filterzyme-sanity/conda_pre_fpocket_LCH.txt`.

## End-to-end outcome

- Exit code 0
- 9 rows × 67 cols in final output (single entry `AK1_Nicotine`)
- 232 files in `filterzyme_output_LCH/`
- Defensive snapshots: `sanity-run_LCH_full-green.log`, `produced-files_LCH_green.txt`,
  `~/filterzyme-sanity/conda_green_LCH.txt`, `~/filterzyme-sanity/pip_green_LCH.txt`

## For the PI (Ariane)

Two patches need to land in the lab tree:

1. Apply `patches/01_ligandRMSD_empty_best_structures_LCH.patch` to
   `filterzyme/steps/computeligandRMSD_step.py` (defensive; only fires when
   `select_best_docked_structures` would return an empty result).
2. Add `fpocket` to the env build instructions:
   `conda install -c conda-forge fpocket`.
