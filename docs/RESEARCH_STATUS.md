# Research status and experimental inventory

The repository contains two levels of reproducibility. Being explicit about this avoids claiming an experiment is production-ready simply because its source archive exists.

## Verified baseline

The 'src/' pipeline targets **MovieLens latest-small**. It trains popularity, TruncatedSVD, genre TF-IDF, a weighted hybrid, and LightGBM LambdaMART, with an optional PyTorch two-tower model. CI executes synthetic tests without downloading MovieLens. The Streamlit application reads trusted locally trained artifacts.

The base benchmark uses a per-user latest-positive holdout. It is **not** directly comparable with the global chronological holdouts used in the Temporal FM research.

## Archived research prototypes (integration in progress)

The 'research-patches/' directory preserves the progression of experiments as ZIP archives:

| Area | Source archives | Scientific status |
| --- | --- | --- |
| MovieLens 1M/32M scaling | 'movielens32m_upgrade.zip', 'movie32m_csv_fix.zip', 'unified_32m_benchmark_patch.zip' | DuckDB/Parquet, BPR and candidate retrieval prototypes |
| Generative retrieval | 'tiger_research_suite_complete.zip', 'tiger_decoder_fix.zip' | TIGER-*inspired* semantic-ID Transformer; decoder bug was patched; not a paper-identical reproduction |
| Temporal FM/DeepFM | 'temporal_hybrid_fm_patch.zip', 'temporal_fm_ranking_patch.zip', 'temporal_fm_pairwise_ranking_patch.zip' | Explicit rating prediction and top-K ranking are **different objectives** |
| Ranking improvements | 'fm_serious_training_patch.zip', 'temporal_fm_rating_aware_patch.zip', 'temporal_fm_regularization_patch.zip', 'temporal_fm_lambdamart_patch.zip' | Experimental BPR training, regularization and second-stage LambdaMART |
| Historical exploration | 'bpr_umap_research_patch.zip', 'streamlit_all_in_one_patch.zip', retired BERT4Rec/SASRec experiments (not included in active source) | Preserved for provenance only; **BERT4Rec/SASRec are not part of the intended active architecture** |

**Important:** these are *overlapping* patch snapshots. Blindly extracting all archives in an arbitrary order can overwrite code and produce invalid checkpoints. Their earlier benchmarks used differing user samples, holdouts, candidate pools and negative-sampling rules. A clean future release should consolidate them into one tested package, migrate the 32M pipeline, retire the obsolete sequence-model code, and run end-to-end integration tests before claiming the advanced application is supported from a fresh clone.

## Benchmark integrity checklist

1. Hold out interactions chronologically and create all features from data available at the prediction cutoff, including historical tags.
2. Ensure compared models share the **same eligible users, catalogue, candidate pool, positive labels, exclusion rules, K, and time cutoff**.
3. Evaluate retrieval Recall@200/500 separately from reranked NDCG@10; no reranker can recover an item absent from retrieval.
4. Distinguish full-catalogue ranking from sampled-negative validation; the latter usually yields much larger, incomparable NDCG.
5. Train candidate generators before the reranker labels, and reserve a genuinely held-out test cohort.
6. Keep model configurations, seeds, dataset signatures and checkpoint provenance with each report.
7. Report actual measured results, including popularity, SVD/BPR baselines, uncertainty intervals, and ablations; do not cherry-pick isolated runs.

No benchmark numbers have been inserted here without checked-in matching experiment artifacts.

## Consolidation plan

- Consolidate ZIP snapshots into the normal version-controlled 'temporal_fm/', 'generative/' and 'scalable/' trees.
- Make package/test imports and requirements deterministic.
- Fix cross-suite synthetic-fixture interference and verify all optional integration tests.
- Add an explicit common 32M chronological evaluation manifest and documented data-rights conditions.
- Provide an automated smoke dataset and CI job for representative 32M-path components.

Until these steps have passed, treat the advanced work as **research prototypes**, not production-ready features.