#!/bin/bash
#SBATCH --job-name=sz-fmo18-energy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

STRUCTUREZYME_ENV="${STRUCTUREZYME_ENV:-/mnt/labs/data/mora/envs/structurezyme}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$STRUCTUREZYME_ENV"
ENV_PY="$(command -v python)"
# Repo root: override with STRUCTUREZYME_REPO if not submitting from the checkout.
WORKTREE="${STRUCTUREZYME_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"

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
