"""Ranking must never back-fill top-K with excluded movie IDs."""
import pandas as pd
import pytest

from src.recommender import HybridRecommender
from src.evaluation import evaluate_models


def _engine():
    movies = pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5],
        "title": ["A", "B", "C", "D", "E"],
        "genres": ["Drama", "Comedy", "Drama", "Comedy", "Action"],
    })
    train = pd.DataFrame({
        "userId": [1, 1, 1, 1, 2, 2, 3, 3],
        "movieId": [1, 2, 3, 4, 1, 2, 3, 4],
    })
    return HybridRecommender().fit(train, movies, rank=1)


def test_topk_never_returns_seen_when_less_than_k_remain():
    model = _engine()
    assert model.recommend(1, k=100).movieId.tolist() == [5]
    assert model.recommend(1, k=100, extra_seen=[5]).empty


def test_cold_start_excludes_every_selected_movie():
    model = _engine()
    assert model.recommend_from_favorites([1, 2, 3, 4, 5], k=10).empty
    recommended = model.recommend_from_favorites([1, 2, 3, 4], k=10)
    assert recommended.movieId.tolist() == [5]


def test_invalid_k_is_rejected():
    with pytest.raises(ValueError, match="k must be >= 1"):
        _engine().recommend(1, k=0)


def test_evaluator_does_not_count_excluded_movies_as_recommended():
    model = _engine()
    held = pd.DataFrame({"userId": [1], "movieId": [5]})
    metrics = evaluate_models(model, held, k=100)
    assert metrics["popularity"]["recall_at_k"] == 1.0