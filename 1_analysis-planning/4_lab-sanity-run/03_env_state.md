# Env state — `filterzyme` (rebuilt, LCH)

**Location:** `/mnt/storage01/home/lherrmann/envs/filterzyme/`
**Python:** 3.11.15

## Versions (as of 2026-06-26, after iteration 3 fixes)

| Package | Version | Source | Notes |
|---------|---------|--------|-------|
| torch | 2.11.0+cu128 | pip (pytorch.org/whl/cu128) | upgraded for Blackwell sm_120 |
| triton | 3.6.0 | pip cu128 | bundled with torch upgrade |
| filterzyme | 0.0.6 | pip (lab tarball) | |
| enzymetk | 0.1.0 | pip (transitive) | README says 0.0.8; we kept 0.1.0 |
| docko | 0.1.5 | pip -e ~/docko_lab_LCH | replaced PyPI 0.1.3; copy of /mnt/labs/data/mora/code/docko |
| boltz | 2.2.1 | pip (transitive) | |
| chai_lab | 0.6.1 | pip | pins old torch; we ignored the pin |
| numpy | ~2.1.x | pip force-reinstall | numba requires <2.2; conda installs had dropped a 2.4 alongside |
| regex | 2026.5.9 | pip -U | enzymetk required ≥2025.10.22 |
| pdbfixer | latest | conda-forge | not on PyPI |
| openbabel | 3.1.0 | conda-forge | not on PyPI |
| plip | latest | conda-forge | not on PyPI |

## Divergences from Ariane's README

1. README's `pip install enzymetk==0.0.8` step skipped (filterzyme pulled 0.1.0; downgrade would break it).
2. torch upgraded to cu128 (Blackwell).
3. Conda packages pdbfixer / openbabel / plip — README does not mention them; Ariane installed them ad-hoc.
4. numpy force-pinned <2.2 to keep numba happy.
5. regex upgraded past docko's pin (docko pin appears spurious).
6. docko: switched from PyPI 0.1.3 to lab 0.1.5 (editable from `~/docko_lab_LCH`); PyPI lacks the 6-arg `run_boltz_affinity` enzymetk needs.

## Revert paths

- Full nuke: `rm -rf /mnt/storage01/home/lherrmann/envs/filterzyme`
- Torch only: `pip install -r $HOME/filterzyme-sanity/torch26_state_LCH.txt`
- Pre-pdbfixer state: `pip_pre_pdbfixer_LCH.txt`, `conda_pre_pdbfixer_LCH.txt` under `$HOME/filterzyme-sanity/`

## Snapshots on disk

```
$HOME/filterzyme-sanity/
├── torch26_state_LCH.txt           # before cu128 upgrade
├── pip_pre_pdbfixer_LCH.txt        # before conda pdbfixer
├── conda_pre_pdbfixer_LCH.txt
├── baseline_LCH.sha                # lab git sha
├── baseline_LCH.status
├── baseline_LCH.diff               # lab uncommitted edits
├── baseline_LCH.timestamp
├── boltz_cache_state_LCH.txt
└── PDE_2H_LCH/
    ├── run_PDE_2H_LCH.py           # smoke-test script (edited base_output_dir)
    ├── PDE_data_formatted.csv
    ├── run_inputs_LCH.sha256
    └── sanity-run_LCH.log          # latest run output (overwritten each retry)
```
