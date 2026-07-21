#!/bin/bash
#SBATCH --job-name=sz_multirow_resume
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=sz_resume_%j.out
#SBATCH --error=sz_resume_%j.err

# Resume the multi-row run from the `ligand_rmsd` step onward, exercising the
# entry-reconciliation fix (commit 5aad838). superimpose + protein_rmsd already
# preserved all 3 entries after the per-row-vina superimpose fix (4a2bac5);
# ligand_rmsd dropped the FMOs because all pose comparisons raised in rdkit.
# This resume forces ligand_rmsd (and everything downstream) to recompute
# against the reconciled frame so geometric_filter/fpocket/ligand_sasa/plip/
# placer all see 3 entries.
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

# Force ligand_rmsd to re-run against the reconciled superimpose/protein_rmsd
# frames, and let every downstream step follow -- the reconciled ligand_rmsd
# pickle will change bytes, invalidating the geometric_filter/fpocket/
# ligand_sasa/plip/placer checkpoint hashes so they recompute automatically.
structurezyme step ligand_rmsd --run-dir "$RUN_DIR" --continue

echo "---- MANIFEST ----"
structurezyme status --run-dir "$RUN_DIR" || true
echo "---- CHECKPOINTS ----"
ls -1 "$RUN_DIR/checkpoints" 2>/dev/null || echo "no checkpoints dir"
echo "---- LOG TAIL ----"
tail -60 "$RUN_DIR/logs/structurezyme.log" 2>/dev/null || echo "no log"
