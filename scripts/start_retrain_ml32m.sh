#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_dir="${1:-artifacts/retrain_runs/fresh_ml32m_v1}"
mkdir -p "$run_dir"
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt duckdb sentence-transformers
# Background training survives terminal closure; a stopped Codespace stops compute.
nohup python -u scripts/retrain_all_ml32m.py --threads 2 --run-dir "$run_dir" \
  > "$run_dir/pipeline.log" 2>&1 < /dev/null &
echo "Pipeline PID: $!"
echo "Progress: tail -f $run_dir/pipeline.log"
echo "Detailed logs: $run_dir/*.log"
