# FM → 500 candidates → LambdaRank

Fresh run (no old checkpoints required):

```bash
bash scripts/train_fm_lambdarank_500.sh
```

This downloads/prepares MovieLens 32M, trains only temporal FM for 10 sampled
minibatch epochs (200 steps/epoch, batch size 64), freezes FM, retrieves its top
500 unseen eligible movies per user, and trains LightGBM LambdaRank. Two CPU
threads are used. Validation uses 20 users and 256 sampled candidates for fast
FM checkpoint selection, so those FM validation numbers are not full-catalog
metrics. No HOFM, DeepFM, BCD, SVD or BPR-MF training stage is launched.

A user with fewer than 500 eligible movies contributes all available movies.
Held-out positives are never injected. Compare original FM versus LambdaRank on
exactly the same candidate pools, using NDCG@10, Recall@10 and Hit Rate@10.

If the FM checkpoint already exists and is compatible, use the disk-only path:

```bash
bash scripts/fm_lambdarank_500.sh
```

The latter never trains FM, requires its weights and temporal feature cache,
and excludes all other checkpoints. Pass `--workspace PATH` if several trained
workspaces exist. Both launchers accept additional CLI options.

The fresh run is isolated under `artifacts/retrain_runs/fm_lambdarank_500_v1`.
Progress appears in the terminal at each stage; detailed stage logs are saved
in that directory. Result table:

```
artifacts/retrain_runs/fm_lambdarank_500_v1/code/artifacts/rerank/fm500/comparison/comparison.csv
```

Earlier successful stages are reused when rerunning with the same configuration.
The launcher uses the system certificate bundle when available and no explicit
bundle was supplied; HTTPS certificate verification stays enabled.

Ranker train/valid/test use distinct user cohorts from the original test period,
while base FM model selection uses the earlier original validation period. This
requires that no saved base checkpoint was selected on original test labels.
All features use the original training snapshot; results are for the retrieved
candidate pools and must not be equated to earlier full-catalog benchmarks.
