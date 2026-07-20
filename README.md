# StructureZyme

Structural filtering pipeline using docking and active-site heuristics to prioritize
ML-predicted enzyme variants for experimental validation. StructureZyme predicts enzyme
structures (Chai / Boltz), docks substrates, and filters the resulting poses using
geometric criteria such as distances, angles, and optionally esterase-specific filters or
nucleophilic proximity.

StructureZyme is driven by a **YAML config + CLI** front end backed by a modular,
resumable runner. A backward-compatible programmatic API is also provided.

---

## Features

- Structure prediction with multiple tools (Chai, Boltz), writing **mmCIF (`.cif`)** models.
- Enzyme–ligand docking analysis using both ML- and physics-based tools.
- Optional catalytic-nucleophile-focused analysis for esterases and other enzymes with
  nucleophilic catalytic residues.
- Optional Vina docking for enzymes with known active sites.
- Config-driven, checkpointed pipeline: resume after a crash, re-run individual steps, and
  submit to Slurm.

---

## Installation

```bash
conda env create -f environment.yml
conda activate structurezyme
python setup.py sdist bdist_wheel
pip install dist/structurezyme-0.1.0.tar.gz --use-deprecated=legacy-resolver
pip install enzymetk==0.0.8
```

> **Catalytic-residue prediction requires the `squidly` CLI** (installed via
> `environment.yml`). The model weights are downloaded from HuggingFace on first setup.
> ESM2 inference requires a GPU.

For full installation instructions and troubleshooting, see
[docs/getting_started.md](docs/getting_started.md).

---

## Quick Start (CLI)

```bash
# 1. Generate a config template
structurezyme init --output run.yml
# 2. Edit run.yml (set output_root, boltz_cache_dir, enable/disable steps)
# 3. Run the pipeline
structurezyme run --config run.yml
# 4. Inspect progress / resume after a crash
structurezyme status --run-dir /path/to/output_root/<user>/<run_id>
structurezyme resume --run-dir /path/to/output_root/<user>/<run_id>
```

### CLI commands

| Command | Description |
|---------|-------------|
| `structurezyme init --output run.yml` | Write a template config file. |
| `structurezyme run --config run.yml [--host HOST] [--force S1,S2]` | Run the pipeline. `--force` recomputes the named steps. |
| `structurezyme resume --run-dir DIR` | Resume an interrupted run from its checkpoints. |
| `structurezyme step NAME --run-dir DIR [--continue]` | Re-run a single step. With `--continue`, also run downstream steps whose inputs changed. |
| `structurezyme status --run-dir DIR` | Show per-step status and wall time. |
| `structurezyme submit --config run.yml [--host --job-name --partition --gpus --time --dry-run]` | Render/submit a Slurm job (`--dry-run` prints the script). |

Structure-prediction steps (Chai/Boltz) emit **mmCIF (`.cif`)** files; PDB files are only
produced by the downstream prepare/clean step consumed by the geometric filters.

---

## Programmatic API

Preferred (modular) API:

```python
from structurezyme.config import load_config
from structurezyme.runner import Runner

cfg = load_config("run.yml")
Runner(cfg).run()
```

A backward-compatible `Pipeline` adapter is still available for existing scripts:

```python
from structurezyme.pipeline import Pipeline
import pandas as pd

df = pd.DataFrame({
    'Entry': ['enzyme_1'],
    'Sequence': ['MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAATGFHI'],
    'substrate_smiles': ['CCOC(=O)C'],
    'substrate_name': ['ethyl_acetate'],
    'substrate_moiety': ['[C](=O)([O])([O])'],
})

pipeline = Pipeline(df=df, boltz_cache_dir="/path/to/boltz/cache")
pipeline.run()
```

> **Migration note:** legacy imports such as `from filterzyme.pipeline import Pipeline`
> still work via a deprecation shim that re-exports `structurezyme`, but emit a
> `DeprecationWarning` and will be removed in a future release. Update imports to
> `structurezyme`.

---

## Input DataFrame Schema

When using the `Pipeline` adapter, the input pandas **DataFrame** must include:
- `Entry` -- unique identifier for each enzyme and substrate pair
- `Sequence` -- amino acid sequence of the enzyme
- `substrate_name` -- name of the substrate
- `substrate_smiles` -- SMILES string of substrate
- `substrate_moiety` -- SMARTS pattern to define chemical moiety of interest within substrate

If cofactors are included, add:
- `cofactor_name` -- name of the cofactor
- `cofactor_smiles` -- SMILES string of cofactor (e.g., PLP: `CC1=NC=C(C(=C1O)C=O)COP(=O)(O)O`)
- `cofactor_moiety` -- SMARTS pattern for the cofactor moiety of interest

---

## Documentation

- [Getting Started](docs/getting_started.md) -- Installation and setup
- [Pipeline Overview](docs/pipeline_overview.md) -- Architecture and design
- [Configuration](docs/configuration.md) -- Config file reference
- [Resume & Checkpoints](docs/resume_and_checkpoints.md) -- Recovering and re-running steps
- [Multi-user Runs](docs/multi_user.md) -- Shared output roots and per-user run dirs
- [API Reference](docs/api_reference.md) -- Full parameter documentation

---

## Repository

Source and issues: <https://github.com/moragroup/StructureZyme>
