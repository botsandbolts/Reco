#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
if [[ -r /etc/ssl/certs/ca-certificates.crt ]]; then
  export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"
  export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}"
fi
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt duckdb sentence-transformers
python -u scripts/retrain_all_ml32m.py --recipe fm_lambdarank \
  --epochs 10 --steps 200 --threads 2 --backend sentence \
  --run-dir artifacts/retrain_runs/fm_lambdarank_500_v1 "$@"
