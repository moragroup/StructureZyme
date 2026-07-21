#!/bin/bash
#SBATCH --job-name=sz_multirow
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=sz_multirow_%j.out
#SBATCH --error=sz_multirow_%j.err

# Multi-row all-modules GPU validation run (3 enzymes).
#
# Submit from the repo root:
#   sbatch tools/gpu_run/run_multirow.sbatch
#
# Validates N>1 handling across all 15 modules with num_threads=1. Time budget
# is larger than the single-row run because 3 enzymes are docked (chai+boltz+
# vina) and relaxed sequentially.
#
# Prereqs are identical to run_allmodules.sbatch (see tools/gpu_run/CHECKLIST.md).
set -euo pipefail

export PATH=/mnt/storage01/home/lherrmann/envs/filterzyme/bin:$PATH

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi
cd "$REPO_ROOT"

echo "host=$(hostname)  python=$(which python)  squidly=$(which squidly || echo MISSING)"
nvidia-smi | head -20 || true

# Build the 3-row input pickle (idempotent; overwrites with the same content).
python tools/gpu_run/make_input_multirow.py --output tools/gpu_run/input_multirow.pkl

# Run the full pipeline. --host default fills output_root/boltz_cache_dir/etc.
# from structurezyme/hosts.yml. run_id in run_multirow.yml is gpu-multirow-01,
# so the run dir is {output_root}/lherrmann/gpu-multirow-01/.
structurezyme run --config tools/gpu_run/run_multirow.yml --host default

# Locate the run dir and print the manifest for verification.
RUN_DIR="/mnt/labs/data/mora/structurezyme_runs/lherrmann/gpu-multirow-01"
echo "---- MANIFEST ----"
structurezyme status --run-dir "$RUN_DIR" || true
echo "---- CHECKPOINTS ----"
ls -1 "$RUN_DIR/checkpoints" 2>/dev/null || echo "no checkpoints dir"
echo "---- LOG TAIL ----"
tail -40 "$RUN_DIR/logs/structurezyme.log" 2>/dev/null || echo "no log"
