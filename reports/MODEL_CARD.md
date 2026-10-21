# Model card — Hybrid Movie Recommender

## Purpose
Research and noncommercial portfolio demonstration of full-catalogue
recommendation and comparative offline ranking. Not intended to make decisions
about people or be deployed in a commercial service.

## Data
MovieLens latest-small, GroupLens / University of Minnesota, anonymized
historical user ratings and movie genres. Source:
https://grouplens.org/datasets/movielens/latest/
No user demographics are included. Upstream terms prohibit commercial or
revenue-bearing use without permission and require attribution; see dataset README.

## Feedback and splits
Treat ratings >=4.0 as *observed positives*. Ratings below this threshold
are omitted from the implicit-positive training graph, not assumed to represent
exposure-level negatives. For users with >=4 positives, keep their last three
positive rating events (ordered by timestamp/movieId) for (1) learning-to-rank
training, (2) validation and (3) one final test event. All other earlier
positives train embedding and SVD models. Ineligible users remain train-only.
This is **per-user** chronology and not a single global time cutoff.

## Models
Popularity; TruncatedSVD; TF-IDF on movie genres; weighted hybrid; optional
PyTorch two-tower embeddings; LightGBM LambdaMART using one held later
positive per user and randomly sampled unobserved items. Optional frozen
sentence-transformer embeddings of movie titles/genres support cold-start.

## Evaluation
Rank all catalogue items per evaluated user, excluding training events plus
earlier held-out positives. Compute Recall@10 (equivalent to hit rate when there
is one held-out positive/user), NDCG@10, and catalogue coverage@10. Compare
model variants using the identical held-out events. Do not use the test set for
hyperparameter selection; adjust using validation only.

## Important limitations
* Movie ratings are not impressions. Missing ratings are *unknown*, not no-clicks.
* The sampled ranker negatives can contain unobserved relevant movies.
* MovieLens is historical and not representative of current streaming catalogues.
* Item genre and title metadata are shallow content features.
* Per-user time splits may leak global chronology across users (future global
  popularity is present in other users' earlier local sequences).
* Small data and a single sample test event per user limit conclusions.
* No A/B testing, counterfactual estimates, or live conversion uplift is claimed.
* Cold-start favorite-based recommendations are not neural personalization.
* Relative metric values depend on arbitrary dataset/split/training settings.

## Deployment/safety
Save artifacts with trusted serialization. `joblib.load`/pickle can execute
code; never load untrusted files. The public app should not allow writes to
model files. Avoid collecting visitor personal data or storing input history.

## Improvements
Global rolling time-window evaluation, historical impression data, in-batch
negatives, hard-negative mining with false-negative mitigations, semantic
synopses from properly licensed sources, Bayesian tuning on validation, ANN
retrieval for scale, and explicit experiment reproducibility.
