#!/bin/bash
#SBATCH --job-name=sz-fmo18-energy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# Absolute paths so the job does not depend on the submit directory or PATH.
ENV_PY="$HOME/envs/filterzyme/bin/python"
WORKTREE="$HOME/structurezyme/.worktrees/energy-aware-selection"

# Put the env's bin on PATH so pipeline steps that shell out to external
# binaries (e.g. fpocket) can find them. The Python code is still invoked
# by absolute path below; this is only for subprocess lookups.
export PATH="$HOME/envs/filterzyme/bin:$PATH"

cd "$WORKTREE"

# Confirm the energy-aware code (not the main checkout) is what gets imported.
"$ENV_PY" -c "from structurezyme.steps.computeligandRMSD_step import pose_energy; print('energy-aware code loaded OK')"

# Run the pipeline via the module entry point (no reliance on PATH).
# --force is overridden to ONLY the steps from fpocket onward: the
# fresh Aug-7 checkpoints for fastrelax/superimpose/protein_rmsd/
# ligand_rmsd/geometric_filter (the energy-aware output) are reused,
# so the ~3 h FastRelax is NOT repeated. Drop --force to re-run the
# full forced pipeline from fastrelax.
"$ENV_PY" -m structurezyme.cli run \
    --config tools/fmo18/run_fmo18.yml \
    --host default \
    --force fpocket,ligand_sasa,plip,placer
