# LambdaRank using existing disk checkpoints only

Run from the repository:

```bash
bash scripts/compare_disk_ml32m.sh
```

This is a separate entry point from the retraining pipeline. It never trains
FM, DeepFM, HOFM, BPR or SVD, and never downloads MovieLens or embeddings.
It trains only LightGBM LambdaRank rerankers. If another retraining process is
already running, this command does not stop it; stop that process separately
if you no longer want its training to continue.

The runner discovers existing supported FM/DeepFM/HOFM3 checkpoint directories,
unified HOFM3/HOFM4 + JointBCD checkpoints, and cached shared-chronology SVD/BPR
factors. A checkpoint must have its weights and metadata; source signatures,
cutoffs, feature backend, dimensions and strict tensor loading must agree.
Missing and incompatible checkpoints are excluded with reasons. No extra
popularity or semantic baseline is added to this comparison.

Every included saved model gets an original-score baseline and a single-model
LambdaRank reranker; the combined ranker uses every included score column.
There is no requirement to have HOFM4 or any other particular model, and no
top-four restriction. At least two valid models are needed for this comparative
stacking experiment. Other unsupported checkpoint families are not loaded.

The root workspace and `artifacts/retrain_runs/*/code` workspaces are inspected.
If several are available, pass the exact one rather than mixing their histories:

```bash
bash scripts/compare_disk_ml32m.sh --workspace artifacts/retrain_runs/fresh_ml32m_v1/code
```

For hashed feature caches use `--backend hash`. Inventory-only audit:

```bash
bash scripts/compare_disk_ml32m.sh --audit
```

The audit validates and loads weights but does not fit a ranker. If inference
modules are absent it installs the repository's existing source patches into a
separate code folder; it does not modify base checkpoints. Model fingerprints
and an inventory are written alongside each completed export. Outputs use a new
`artifacts/rerank/disk_<timestamp>` directory in the selected workspace.

Default cohorts: 700 ranker-training, 200 early-stopping, 300 held-out users.
These three user-disjoint cohorts come from the original test period, to avoid
reusing the earlier base-selection validation labels. Frozen inference uses the
original training cutoff. This requires that the original test period was not
used to select or tune the saved base checkpoints. Metadata cannot prove that
history, and the report records this limitation. Do not use checkpoints selected
on those test labels for a clean held-out claim.

All models retrieve into one label-blind union of their top 200 candidates. No
positive item is injected. Training queries with no informative labels are
omitted; validation/test retain missed positives in the denominator. Metrics
refer to these supplied pools, not a full-catalog comparison. Defaults can be
changed with `--train-users`, `--valid-users`, `--test-users`, and `--pool`.

If no compatible checkpoint/data pair is found, it stops and reports the
problem. It never fills the gap by retraining missing models.
