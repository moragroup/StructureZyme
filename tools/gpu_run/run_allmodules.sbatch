#!/bin/bash
#SBATCH --job-name=sz_allmod
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=sz_allmod_%j.out
#SBATCH --error=sz_allmod_%j.err

# Phase F all-modules GPU verification run.
#
# Submit from the repo root:
#   sbatch tools/gpu_run/run_allmodules.sbatch
#
# Prereqs (see tools/gpu_run/CHECKLIST.md):
#   - conda env at the path below has: structurezyme (pip install -e .),
#     squidly on PATH, chai/boltz, vina toolchain (vina, obabel, meeko),
#     pyrosetta (fastrelax), and the PLACER env at the host path.
#   - default host profile paths exist and are writable
#     (/mnt/labs/data/mora/structurezyme_runs, .../boltz_cache).
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

# Build the input pickle (idempotent; overwrites with the same content).
python tools/gpu_run/make_input.py --output tools/gpu_run/input.pkl

# Run the full pipeline. --host default fills output_root/boltz_cache_dir/etc.
# from structurezyme/hosts.yml. run_id in run.yml is gpu-allmodules-01, so the
# run dir is {output_root}/lherrmann/gpu-allmodules-01/.
structurezyme run --config tools/gpu_run/run.yml --host default

# Locate the run dir and print the manifest for verification.
RUN_DIR="/mnt/labs/data/mora/structurezyme_runs/lherrmann/gpu-allmodules-01"
echo "---- MANIFEST ----"
structurezyme status --run-dir "$RUN_DIR" || true
echo "---- CHECKPOINTS ----"
ls -1 "$RUN_DIR/checkpoints" 2>/dev/null || echo "no checkpoints dir"
echo "---- LOG TAIL ----"
tail -40 "$RUN_DIR/logs/structurezyme.log" 2>/dev/null || echo "no log"
