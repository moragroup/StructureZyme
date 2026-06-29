# Task 03 — Build the `filterpipeline` conda env + binaries

**Depends on:** Task 02 (Miniforge). **Owner unit:** standalone.

## Why

`environment.yml` declares Python + the bioconda binaries (`foldseek`, `mmseqs2`, `fpocket`, `openbabel`, `plip`) + `faiss-gpu`, `pdbfixer`, `openmm==8.3.1`, plus a `pip:` section (`enzymetk`, `rdkit`, `freesasa`, `docko`, `chai_lab`). Building this env provides the non-Python executables that uv/PyPI cannot.

## Known risk to watch: openmm pin conflict

- `environment.yml` pins `openmm==8.3.1`.
- `docko` (PyPI metadata) hard-pins `OpenMM==8.1.1` and its README recommends `openmm==8.0` to avoid a pdbfixer/Modeller error.
- **Expect a pip/conda resolution conflict when `docko` is pip-installed in Task 04.** Document it. If docko breaks at import/use with 8.3.1, fall back to `openmm==8.1.1` (docko's pin) or `8.0`. Note that lowering openmm may affect the `cleanPDB`/pdbfixer steps — verify Task 07.

## Commands

From inside the interactive session, with conda activated:

```bash
cd /mnt/storage01/home/lherrmann/StructureZyme
source "$HOME/miniforge3/etc/profile.d/conda.sh"

# Build from the repo file (uses mamba solver under the hood with miniforge)
mamba env create -f environment.yml
conda activate filterpipeline
python --version            # expect 3.11.8
```

If the solve fails on `faiss-gpu` or `pytorch::faiss-gpu` (common on newer CUDA), note it and try without faiss first, or `mamba install -c pytorch faiss-cpu` as a stopgap (faiss is only used by some enzymetk paths).

## Verify the binaries landed

```bash
for t in foldseek mmseqs fpocket obabel plip; do
  printf "%s: " "$t"; command -v "$t" || echo "MISSING"
done
```

`vina` is **not** in `environment.yml`. AutoDock Vina executable is required by `docko` for the Vina docking path. Install it now:

```bash
mamba install -c bioconda -c conda-forge autodock-vina
command -v vina || command -v autodock_vina || echo "VINA MISSING — see docko note"
```

(If bioconda's `autodock-vina` is unavailable for the platform, download a static release from https://github.com/ccsb-scripps/AutoDock-Vina/releases and put it on PATH.)

## Network note

Compute nodes may lack outbound internet. If `mamba env create` cannot reach channels, either (a) run the solve/download on the login node (no GPU needed for the env build itself, only for running), then `conda activate` it inside the GPU session, or (b) request a partition with egress.

## Success criterion

`conda activate filterpipeline` works, `python --version` = 3.11.8, and `foldseek`/`mmseqs`/`fpocket`/`obabel`/`plip`/`vina` are all on PATH (record any that are MISSING).
