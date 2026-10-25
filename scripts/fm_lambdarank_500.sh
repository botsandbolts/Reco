#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python -u -m src.lambdarank_disk --models fm_all_bpr --pool 500 --threads 2 "$@"
