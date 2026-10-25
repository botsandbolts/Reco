# Fresh MovieLens 32M training

Run `bash scripts/start_retrain_ml32m.sh` from the project root. No old account,
pretrained checkpoint, or manual prediction export is required. The default uses
two CPU threads and the full 32M source dataset; optimizers use sampled minibatches,
not a full pass through all 32 million ratings per epoch. Each stage uses 15 epochs
and 800 steps per epoch; sampled validation selects checkpoints. Runtime on two
CPUs is not measured and may be substantial. Download, embedding generation,
preprocessing, SVD fitting and full-catalog score export also take time.

The launcher installs dependencies, runs in the background, and records progress.
A stopped Codespace does not continue training. Follow the status with:

```bash
tail -f artifacts/retrain_runs/fresh_ml32m_v1/pipeline.log
```

The original source tree and checkpoints are preserved. Research patches are
installed into `artifacts/retrain_runs/fresh_ml32m_v1/code`; all generated data,
weights and reports remain inside that isolated directory. New runs train without
old checkpoints; successful stages are skipped when the same run is resumed.
A failed, unfinished individual trainer stage restarts rather than resuming its
optimizer. Do not launch the same run twice. Do not delete its lock until you have
verified the previous process is no longer running. Use a new run directory to
repeat the experiment from scratch.

Training order:

1. Download MovieLens 32M and prepare disk-backed data.
2. Train temporal FM and DeepFM from random initialization.
3. Fine-tune a separate FM with 25% semi-hard negatives.
4. Add/train HOFM3 from that freshly trained FM.
5. Train temporal preference stages B, C, D, E from the new semi-hard FM.
6. Train HOFM3 + JointBCD and HOFM4 + JointBCD from the new HOFM3 and Stage D.
7. Fit chronological SVD and BPR baselines; derive popularity and title-profile
   semantic scores, then export common candidates and every frozen model's scores.
8. Select the top four frozen models by ranker-validation NDCG, train individual
   LambdaRank rerankers and a stacked ranker, compare against original scores,
   z-mean fusion, and RRF, and write CSV/Markdown plus JSON reports.

Semantic vectors use Sentence Transformers. No tag dataset is required: the
current downloader retrieves ratings and movies; absent tag inputs mean the
historical tag feature branches have no tag evidence. This is recorded by the
original trainers. Stage B-E heads are checkpointed; the comparison exports the
unified JointBCD models rather than four separate residual-head score columns.
This suite covers the recommendation family above; SASRec, BERT4Rec and TIGER
are not trained by this pipeline.

## Evaluation protocol

Base models fit the original global training period and select checkpoints using
its original validation period. To avoid reusing base-selection labels as ranker
supervision, the original TEST-period users are partitioned into independent
ranker train/early-stop/test cohorts (700/200/300). This is a new, explicitly named
protocol: it is not directly comparable with earlier published full-catalog NDCG.
Ranker train/valid/test labels share a period but have disjoint users. Historical
features and recommendation time stay fixed at the original training cutoff.

Every frozen retriever contributes its top 200 eligible movies to the common
label-blind pool; positives are never injected. Ranker-training queries without
both classes are skipped; evaluation keeps retrieval misses. `total_relevant`
retains all eligible relevant movies. Final NDCG/recall includes retrieval loss
but is evaluated only on the supplied pools. SVD/BPR and all neural models use
the same global training history. Query-bootstrap intervals are conditional on
this fixed training run and exclude training/selection uncertainty.

Outputs:

```
artifacts/retrain_runs/fresh_ml32m_v1/code/models/ml-32m/
artifacts/retrain_runs/fresh_ml32m_v1/code/artifacts/rerank/train.csv
artifacts/retrain_runs/fresh_ml32m_v1/code/artifacts/rerank/valid.csv
artifacts/retrain_runs/fresh_ml32m_v1/code/artifacts/rerank/test.csv
artifacts/retrain_runs/fresh_ml32m_v1/code/artifacts/rerank/evaluation_manifest.json
artifacts/retrain_runs/fresh_ml32m_v1/code/artifacts/rerank/comparison/comparison.csv
artifacts/retrain_runs/fresh_ml32m_v1/code/artifacts/rerank/comparison/comparison.md
```

Inspect the exact commands without starting compute:

```bash
python scripts/retrain_all_ml32m.py --plan
```

Use `--run-dir NEW_PATH`, `--epochs N`, `--steps N`, or `--threads N` with the
Python driver to change the budget. Any configuration change requires a new run
directory. The hash backend is only an explicit offline test alternative and is
not interchangeable with Sentence Transformer performance.
