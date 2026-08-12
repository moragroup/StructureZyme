#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRUCTUREZYME_ENV="${STRUCTUREZYME_ENV:-/mnt/labs/data/mora/envs/structurezyme}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$STRUCTUREZYME_ENV"
cd /mnt/storage01/home/lherrmann/StructureZyme
mkdir -p "$HOME/boltz_cache"
echo "=== host=$(hostname)  python=$(which python)  squidly=$(which squidly) ==="
nvidia-smi | head -12 || true
python "$SCRIPT_DIR/run_phase_c_smoke.py" --boltz-cache "$HOME/boltz_cache" --output-dir pipeline_output_phaseC_vina
