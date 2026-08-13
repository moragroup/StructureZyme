#!/bin/bash
#SBATCH --job-name=sz_fmo18
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=sz_fmo18_%j.out
#SBATCH --error=sz_fmo18_%j.err

# Full-pipeline run over 18 FMO-family enzymes with indole substrate + FAD
# cofactor. Squidly at mean_prob=0.05, mean_var=0.5 (defaults produce no
# residues for FMOs). Rows without predicted residues are still co-folded by
# chai/boltz and analysed downstream; only vina is skipped per-row for them.
#
# Submit from the repo root:
#   sbatch tools/fmo18/run_fmo18.sbatch
#
# Prereqs (see tools/gpu_run/CHECKLIST.md for full details):
#   - env `structurezyme` (default `/mnt/labs/data/mora/envs/structurezyme`, override with `$STRUCTUREZYME_ENV`)
#   - squidly weights at /mnt/labs/data/mora/squidly_weights (host profile)
#   - PLACER env at /mnt/labs/data/mora/software/PLACER/env (host profile)
#   - boltz cache at /mnt/labs/data/mora/boltz_cache (host profile)

set -euo pipefail

STRUCTUREZYME_ENV="${STRUCTUREZYME_ENV:-/mnt/labs/data/mora/envs/structurezyme}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$STRUCTUREZYME_ENV"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi
cd "$REPO_ROOT"

echo "host=$(hostname)  python=$(which python)  squidly=$(which squidly || echo MISSING)"
nvidia-smi | head -20 || true

# Rebuild the 18-row input pickle (idempotent).
python tools/fmo18/make_input_fmo18.py --output tools/fmo18/input_fmo18.pkl

# Run the full pipeline. --host default fills unset paths from
# structurezyme/hosts.yml (boltz_cache_dir, squidly_weights_dir, placer_env_path);
# output_root is set explicitly in run_fmo18.yml to /mnt/labs/data/mora so the
# run lands under /mnt/labs/data/mora/Luca/fmo-fad-01/ (user=Luca, run_id=fmo-fad-01).
structurezyme run --config tools/fmo18/run_fmo18.yml --host default

RUN_DIR="/mnt/labs/data/mora/Luca/fmo-fad-01"
echo "---- MANIFEST ----"
structurezyme status --run-dir "$RUN_DIR" || true
echo "---- CHECKPOINTS ----"
ls -1 "$RUN_DIR/checkpoints" 2>/dev/null || echo "no checkpoints dir"
echo "---- LOG TAIL ----"
tail -40 "$RUN_DIR/logs/structurezyme.log" 2>/dev/null || echo "no log"
