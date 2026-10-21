"""Add Sentence Transformer results to the *existing* MovieLens benchmark.

Run AFTER `python -m src.semantic` and AFTER the main training command.
This script does not retrain the model or overwrite its joblib artifact.

    python -m src.benchmark_semantic

Metrics for `semantic` and `svd_semantic` are written to artifacts/metrics.json
using exactly the same held-out MovieLens events and ranking exclusions as
other models. Existing model metrics are preserved.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

from .data import ensure_dataset, chronological_split
from .evaluation import evaluate_models, union_seen
from .recommender import HybridRecommender


class SemanticBenchmarkAdapter:
    """Supply semantic ranking scores to the existing, tested evaluator.

    The semantic user representation is computed ONLY from training movies;
    validation/test interactions are NEVER used to build the profile.
    """

    models = ["semantic", "svd_semantic"]

    def __init__(self, engine: HybridRecommender):
        if getattr(engine, "semantic_item", None) is None:
            raise RuntimeError(
                "No Sentence Transformer embeddings found in trained artifact. "
                "Run `python -m src.semantic` first."
            )
        self.engine = engine
        self.user_to_idx = engine.user_to_idx
        self.movie_to_idx = engine.movie_to_idx
        self.seen = engine.seen
        self.movie_ids = engine.movie_ids

        vectors = np.asarray(engine.semantic_item, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] != len(engine.movie_ids):
            raise ValueError("Embedding count does not match the movie catalogue")
        self.items = normalize(vectors, axis=1)

        # Per-user profiles depend solely on the training interaction matrix.
        self.users = normalize(
            np.asarray(engine.train_matrix @ self.items, dtype="float32"),
            axis=1,
        )
        self._cache = {}

    def _semantic_scores(self, uid):
        uid = int(uid)
        if uid not in self._cache:
            profile = self.users[self.user_to_idx[uid]]
            # Cosine scores lie in [-1, 1]; transform monotonically to [0, 1].
            self._cache[uid] = (1.0 + self.items @ profile) * 0.5
        return self._cache[uid]

    def score_user(self, user_id, model="semantic"):
        semantic = self._semantic_scores(user_id)
        if model == "semantic":
            return semantic
        if model == "svd_semantic":
            features = self.engine.features_for_user(user_id)
            # Fixed, illustrative weights, NOT tuned on the test split.
            return 0.50 * features[:, 1] + 0.40 * semantic + 0.10 * features[:, 0]
        raise ValueError(f"Unknown semantic scoring method: {model}")


def _check_training_history(engine, train):
    """Prevent accidentally comparing a model trained on different events."""
    expected = {}
    for row in train.itertuples(index=False):
        if int(row.movieId) in engine.movie_to_idx:
            expected.setdefault(int(row.userId), set()).add(engine.movie_to_idx[int(row.movieId)])
    if expected != engine.seen:
        raise ValueError(
            "Saved recommender has a different training split/dataset. "
            "Retrain the base model, run src.semantic, then rerun this benchmark."
        )


def run(data_dir="data", output_dir="artifacts", model_path=None):
    output = Path(output_dir)
    model_path = Path(model_path) if model_path else output / "recommender.joblib"
    metrics_path = output / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError("Existing metrics.json not found. Run `python -m src.train` first.")
    engine = HybridRecommender.load(model_path)
    adapter = SemanticBenchmarkAdapter(engine)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    k = int(metrics.get("k", 10))
    ratings, movies = ensure_dataset(data_dir)
    splits = chronological_split(ratings)
    _check_training_history(engine, splits.train)
    if metrics.get("counts", {}).get("training_positives") != len(splits.train):
        raise ValueError("Metrics are from a different training run. Please regenerate.")
    if metrics.get("catalog_size") != len(movies):
        raise ValueError("Metrics catalogue size differs from the current dataset.")

    # Reuse the same metric calculation AND exclusion policy as src.train.
    valid = evaluate_models(adapter, splits.valid, union_seen(splits.rank_train), k=k)
    test = evaluate_models(
        adapter, splits.test, union_seen(splits.rank_train, splits.valid), k=k
    )
    # This backup is created once, so a second run won't clobber the original.
    backup = output / "metrics_before_semantic.json"
    if not backup.exists():
        shutil.copy2(metrics_path, backup)

    metrics["validation"].update(valid)
    metrics["test"].update(test)
    metrics["semantic_benchmark_notes"] = (
        "Sentence Transformers all-MiniLM-L6-v2 encoded movie title+genre metadata. "
        "User semantic profiles come from training positives only. "
        "svd_semantic uses fixed untuned weights: 50% SVD, 40% semantic, 10% popularity. "
        "Metrics use the same full-catalogue, per-user chronological protocol "
        "as the other baselines; missing ratings are not negatives."
    )
    # Write atomically to reduce chance of a corrupt report on interruption.
    temp = metrics_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    temp.replace(metrics_path)
    print(f"Added results to {metrics_path}:\n")
    for model in adapter.models:
        print(
            f"  {model:15s} validation Recall@{k}={valid[model]['recall_at_k']}  "
            f"NDCG@{k}={valid[model]['ndcg_at_k']}  "
            f"test Recall@{k}={test[model]['recall_at_k']}  "
            f"NDCG@{k}={test[model]['ndcg_at_k']}"
        )
    print(f"Original metrics backup: {backup}")
    print("Restart or refresh Streamlit and open the Benchmark tab.")
    return valid, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--model-path", default=None)
    run(**vars(parser.parse_args()))
