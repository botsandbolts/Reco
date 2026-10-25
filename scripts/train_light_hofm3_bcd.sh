#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt duckdb sentence-transformers
python -u scripts/retrain_all_ml32m.py \
  --recipe light_hofm3 --dataset ml-32m --backend sentence \
  --epochs 10 --steps 200 --threads 2 \
  --run-dir artifacts/retrain_runs/light_hofm3_bcd_10e_v1 "$@"
