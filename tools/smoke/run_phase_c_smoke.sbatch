#!/bin/bash
#SBATCH --job-name=fz_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=fz_smoke_%j.out
#SBATCH --error=fz_smoke_%j.err

set -euo pipefail

STRUCTUREZYME_ENV="${STRUCTUREZYME_ENV:-/mnt/labs/data/mora/envs/structurezyme}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$STRUCTUREZYME_ENV"

# Resolve repo root. Normal usage is `sbatch tools/smoke/run_phase_c_smoke.sbatch`
# from the repo root, so SLURM_SUBMIT_DIR == repo root. Prefer the Slurm
# submission dir; fall back to the dir two levels above this script
# (tools/smoke/ -> repo root) for direct `bash` runs.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi
cd "$REPO_ROOT"

# Boltz cache path - edit if you keep it elsewhere
BOLTZ_CACHE="${BOLTZ_CACHE:-$HOME/boltz_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-pipeline_output_phaseC_vina}"

echo "host=$(hostname)  python=$(which python)  squidly=$(which squidly)"
nvidia-smi | head -20 || true

python "$REPO_ROOT/tools/smoke/run_phase_c_smoke.py" \
    --boltz-cache "$BOLTZ_CACHE" \
    --output-dir "$OUTPUT_DIR"

echo "---- VALIDATION ----"
python - <<PY
import pandas as pd, pathlib
out = pathlib.Path("${OUTPUT_DIR}")
sq = out / "docking" / "squidly.pkl"
print("squidly.pkl exists:", sq.exists())
if sq.exists():
    df = pd.read_pickle(sq)
    print(df[[c for c in ('Entry','catalytic_residues','Squidly_CR_Position') if c in df.columns]])
for p in out.rglob("structural_features_final.pkl"):
    print("final pkl:", p)
PY
