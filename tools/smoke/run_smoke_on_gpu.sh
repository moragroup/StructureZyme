#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/mnt/storage01/home/lherrmann/envs/filterzyme/bin:$PATH
cd /mnt/storage01/home/lherrmann/StructureZyme
mkdir -p "$HOME/boltz_cache"
echo "=== host=$(hostname)  python=$(which python)  squidly=$(which squidly) ==="
nvidia-smi | head -12 || true
python "$SCRIPT_DIR/run_phase_c_smoke.py" --boltz-cache "$HOME/boltz_cache" --output-dir pipeline_output_phaseC_vina
