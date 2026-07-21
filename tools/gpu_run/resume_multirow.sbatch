#!/bin/bash
#SBATCH --job-name=sz_multirow_resume
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=sz_resume_%j.out
#SBATCH --error=sz_resume_%j.err

# Resume the multi-row run from the `superimpose` step onward, exercising the
# per-row-vina-skip fix in run_superimpose (commit 4a2bac5). Reuses the
# already-completed squidly/chai/boltz/vina/docking_metrics/prepare_files/
# fastrelax checkpoints from gpu-multirow-02, so this run only touches the
# CPU-only tail (superimpose -> protein_rmsd -> ligand_rmsd -> geometric_filter
# -> fpocket -> ligand_sasa -> plip -> placer).
#
# Submit from the repo root:
#   sbatch tools/gpu_run/resume_multirow.sbatch
#
# No GPU is requested; the tail steps are CPU-only. We stay on the gpu
# partition because that's the only partition referenced in this repo and it
# will accept a no-GPU CPU allocation.

set -euo pipefail

export PATH=/mnt/storage01/home/lherrmann/envs/filterzyme/bin:$PATH

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi
cd "$REPO_ROOT"

RUN_DIR="/mnt/labs/data/mora/structurezyme_runs/lherrmann/gpu-multirow-02"

echo "host=$(hostname)  python=$(which python)"
echo "run_dir=$RUN_DIR"

# Force superimpose to re-run (its old checkpoint has only 1 entry from the
# pre-fix run) and let every downstream step follow, recomputing any whose
# inputs change.
structurezyme step superimpose --run-dir "$RUN_DIR" --continue

echo "---- MANIFEST ----"
structurezyme status --run-dir "$RUN_DIR" || true
echo "---- CHECKPOINTS ----"
ls -1 "$RUN_DIR/checkpoints" 2>/dev/null || echo "no checkpoints dir"
echo "---- LOG TAIL ----"
tail -60 "$RUN_DIR/logs/structurezyme.log" 2>/dev/null || echo "no log"
