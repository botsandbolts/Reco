"""Fast synthetic tests without downloading pretrained embeddings or datasets."""
import numpy as np
from sklearn.preprocessing import normalize
from src.benchmark_semantic import SemanticBenchmarkAdapter, _check_training_history
from src.recommender import HybridRecommender
from src.data import chronological_split
from src.evaluation import evaluate_models, union_seen
import pandas as pd
import pytest


def test_semantic_comparison_uses_training_only():
    movies = pd.DataFrame({
        "movieId": range(1, 15),
        "title": [f"Movie {i}" for i in range(1, 15)],
        "genres": (["Comedy", "Drama", "Action"] * 5)[:14],
    })
    events = [(u, m, 5.0, 100*u + m) for u in range(1, 5) for m in range(1, 10)]
    ratings = pd.DataFrame(events, columns=["userId", "movieId", "rating", "timestamp"])
    splits = chronological_split(ratings)
    model = HybridRecommender().fit(splits.train, movies, rank=2)
    rng = np.random.default_rng(4)
    model.semantic_item = normalize(rng.normal(size=(len(model.movie_ids), 8)).astype("float32"))
    adapter = SemanticBenchmarkAdapter(model)
    _check_training_history(model, splits.train)
    assert adapter.score_user(1, "semantic").shape == (14,)
    result = evaluate_models(adapter, splits.valid, union_seen(splits.rank_train), k=3)
    assert set(result) == {"semantic", "svd_semantic"}
    assert all(0 <= result[model]["recall_at_k"] <= 1 for model in result)
    assert "semantic" not in HybridRecommender().models  # no unintended baseline changes


def test_semantic_requires_embedding():
    rec = HybridRecommender()
    with pytest.raises(RuntimeError, match="Sentence Transformer"):
        SemanticBenchmarkAdapter(rec)
