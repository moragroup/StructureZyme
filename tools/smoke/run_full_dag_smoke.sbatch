#!/bin/bash
#SBATCH --job-name=sz_fulldag_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Full-DAG GPU smoke: all 15 steps incl. fastrelax (subprocess -> RosettaFastRelax
# venv) and placer, against the FRESH handoff env. Submit from repo root:
#   mkdir -p smoke_logs
#   sbatch --output="$PWD/smoke_logs/%x_%j.out" --error="$PWD/smoke_logs/%x_%j.err" \
#          tools/smoke/run_full_dag_smoke.sbatch
# Logs MUST be on shared storage; the override above puts them under the repo.
set -euo pipefail

STRUCTUREZYME_ENV="${STRUCTUREZYME_ENV:-/mnt/storage01/home/lherrmann/envs/structurezyme}"
FASTRELAX_ENV="${FASTRELAX_ENV:-/mnt/labs/data/mora/software/RosettaFastRelax/env}"
export FASTRELAX_ENV
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$STRUCTUREZYME_ENV"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi
cd "$REPO_ROOT"

echo "host=$(hostname) python=$(which python) fastrelax_env=$FASTRELAX_ENV"
nvidia-smi | head -20 || true

python tools/gpu_run/make_input.py --output tools/gpu_run/input.pkl
structurezyme run --config tools/gpu_run/run.yml --host default

RUN_DIR="/mnt/labs/data/mora/structurezyme_runs/lherrmann/gpu-allmodules-01"
echo "---- VALIDATION ----"
python - "$RUN_DIR" <<'PY'
import sys, glob
import pandas as pd
from pathlib import Path

run = Path(sys.argv[1])
rc = 0


def check(name, ok, detail=""):
    global rc
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        rc = 1


def _dict_has_scores(v):
    """fastrelax_score is {engine: {pose: score, ...}, ...}; True if any
    engine has at least one populated pose score."""
    if not isinstance(v, dict):
        return False
    for engine_scores in v.values():
        if isinstance(engine_scores, dict) and len(engine_scores) > 0:
            return True
        if engine_scores not in (None, {}, [], ""):
            return True
    return False


# fastrelax: checkpoint + relaxed pdbs + score column populated
fr_ckpt = run / "checkpoints" / "fastrelax.pkl"
check("fastrelax checkpoint exists", fr_ckpt.exists(), str(fr_ckpt))
relaxed = list((run / "superimposition" / "fastrelax").rglob("*_relaxed.pdb"))
check("fastrelax produced relaxed pdbs", len(relaxed) > 0, f"{len(relaxed)} files")
if fr_ckpt.exists():
    fr = pd.read_pickle(fr_ckpt)
    has = "fastrelax_score" in fr.columns and any(
        _dict_has_scores(v) for v in fr["fastrelax_score"]
    )
    check("fastrelax_score populated", has)

# placer: checkpoint + CSV + placer_* columns populated
pl_ckpt = run / "checkpoints" / "placer.pkl"
check("placer checkpoint exists", pl_ckpt.exists(), str(pl_ckpt))
csvs = glob.glob(str(run / "placer" / "*.csv"))
check("placer produced a CSV", len(csvs) > 0, f"{len(csvs)} files")
if pl_ckpt.exists():
    pl = pd.read_pickle(pl_ckpt)
    score_cols = [
        c for c in pl.columns if c.startswith("placer_") and c != "placer_dir"
    ]
    populated = (
        any(pl[c].notna().any() for c in score_cols) if score_cols else False
    )
    check("placer_* score columns populated", populated, f"cols={score_cols}")

# end-to-end: final structural features pickle (written under geometricfiltering/)
finals = list(run.rglob("structural_features_final.pkl"))
check(
    "structural_features_final.pkl exists",
    len(finals) > 0,
    str(finals[0]) if finals else "MISSING",
)

print("SMOKE RESULT:", "PASS" if rc == 0 else "FAIL")
sys.exit(rc)
PY
