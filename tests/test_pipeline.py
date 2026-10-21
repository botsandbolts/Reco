import numpy as np
import pandas as pd
import pytest
from src.data import chronological_split
from src.recommender import HybridRecommender
from src.evaluation import evaluate_models, union_seen


def fake_dataset():
    movies = pd.DataFrame({"movieId": np.arange(1, 15),
                           "title": [f"Movie {i}" for i in range(1, 15)],
                           "genres": (["Action|Adventure", "Drama", "Comedy|Romance"] * 5)[:14]})
    records = []
    for user_id in range(1, 7):
        for idx, movie_id in enumerate(range(1, 10)):
            records.append((user_id, movie_id, 4.0 if idx % 2 == 0 else 5.0,
                            user_id * 100 + idx))
    # An insufficient-positive user should remain train-only.
    records.extend([(7, 1, 4.0, 10), (7, 2, 4.0, 11), (7, 3, 1.0, 12)])
    ratings = pd.DataFrame(records, columns=["userId", "movieId", "rating", "timestamp"])
    return ratings, movies


def test_split_disjoint_and_temporal():
    ratings, _ = fake_dataset()
    s = chronological_split(ratings)
    assert len(s.valid) == 6 and len(s.test) == 6 and len(s.rank_train) == 6
    assert set(s.train.userId) == set(range(1, 8))
    for user_id in range(1, 7):
        prev = s.train[s.train.userId == user_id].timestamp.max()
        rank = s.rank_train[s.rank_train.userId == user_id].timestamp.iloc[0]
        val = s.valid[s.valid.userId == user_id].timestamp.iloc[0]
        test = s.test[s.test.userId == user_id].timestamp.iloc[0]
        assert prev < rank < val < test
    assert not set(zip(s.train.userId, s.train.movieId)) & set(zip(s.test.userId, s.test.movieId))


def test_training_recommends_and_evaluates():
    ratings, movies = fake_dataset()
    split = chronological_split(ratings)
    engine = HybridRecommender().fit(split.train, movies, rank=3)
    rec = engine.recommend(1, k=5)
    assert len(rec) == 5
    assert not set(rec.movieId) & set(split.train[split.train.userId == 1].movieId)
    cold = engine.recommend_from_favorites([1, 2], k=3)
    assert len(cold) == 3 and not set(cold.movieId) & {1, 2}
    valid = evaluate_models(engine, split.valid, union_seen(split.rank_train), k=3)
    assert valid["svd"]["users_evaluated"] == 6
    assert 0 <= valid["hybrid"]["ndcg_at_k"] <= 1
    with pytest.raises(ValueError):
        evaluate_models(engine, split.test, union_seen(split.test), k=3)


def test_ranker_and_serialization(tmp_path):
    ratings, movies = fake_dataset()
    split = chronological_split(ratings)
    engine = HybridRecommender().fit(split.train, movies, rank=3)
    engine.fit_ranker(split.rank_train, negatives=5)
    assert "lambdamart" in engine.models
    output = tmp_path / "model.joblib"
    engine.save(output)
    restored = HybridRecommender.load(output)
    assert np.allclose(engine.score_user(1, "lambdamart"),
                       restored.score_user(1, "lambdamart"))


def test_end_to_end_train_cli_with_synthetic_data(monkeypatch, tmp_path):
    from src import train as training
    ratings, movies = fake_dataset()
    monkeypatch.setattr(training, "ensure_dataset", lambda _: (ratings, movies))
    model, report = training.run(data_dir=str(tmp_path / "data"),
                                 output_dir=str(tmp_path / "artifacts"),
                                 neural=False, rank=3)
    assert (tmp_path / "artifacts/recommender.joblib").exists()
    assert (tmp_path / "artifacts/metrics.json").exists()
    assert report["counts"]["users_test"] == 6
    assert set(report["test"]) == set(model.models)


def test_neural_optional():
    pytest.importorskip("torch")
    ratings, movies = fake_dataset()
    split = chronological_split(ratings)
    engine = HybridRecommender().fit(split.train, movies, rank=3)
    engine.fit_neural(split.train, epochs=1, dim=8)
    assert engine.neural_user.shape == (7, 8)
    assert engine.neural_item.shape == (14, 8)
    assert len(engine.recommend(1, k=3, model="two_tower")) == 3
