"""CPU-friendly hybrid recommender; optional PyTorch two-tower and LambdaMART.

This is a portfolio research prototype, not a production personalization API.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize

FEATURE_NAMES = ["log_popularity", "svd_affinity", "genre_affinity", "neural_affinity"]


def _minmax(x):
    x = np.asarray(x, dtype="float32").ravel()
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)

def _top_finite_scores(scores, k):
    """Return at most k eligible results; never return excluded (-inf) movies.

    Handles catalogues where fewer than k items remain after filtering.
    Stable sorting preserves catalogue order when scores tie.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    scores = np.asarray(scores)
    eligible = np.flatnonzero(np.isfinite(scores))
    return eligible[np.argsort(-scores[eligible], kind="stable")[:k]]


class HybridRecommender:
    def __init__(self):
        self.user_to_idx = {}
        self.movie_to_idx = {}
        self.neural_user = None
        self.neural_item = None
        self.ranker = None
        self.semantic_item = None

    def fit(self, train: pd.DataFrame, movies: pd.DataFrame, rank=48):
        if train.empty:
            raise ValueError("Training interactions are empty")
        # Movie catalogue is independent from the ratings: unseen/genre-only
        # movies can appear in content-based recommendations.
        movies = movies.drop_duplicates("movieId").sort_values("movieId").reset_index(drop=True)
        self.movies = movies[["movieId", "title", "genres"]].copy()
        self.movie_ids = self.movies.movieId.to_numpy(dtype="int64")
        self.movie_to_idx = {int(v): idx for idx, v in enumerate(self.movie_ids)}
        self.user_ids = np.sort(train.userId.unique()).astype("int64")
        self.user_to_idx = {int(v): idx for idx, v in enumerate(self.user_ids)}
        self.seen = {int(u): set() for u in self.user_ids}
        rows, cols = [], []
        for u, movie in zip(train.userId, train.movieId):
            if int(movie) not in self.movie_to_idx:
                continue
            ui, mi = self.user_to_idx[int(u)], self.movie_to_idx[int(movie)]
            rows.append(ui)
            cols.append(mi)
            self.seen[int(u)].add(mi)
        if not rows:
            raise ValueError("No training events match the movie catalogue")
        n_users, n_movies = len(self.user_ids), len(self.movie_ids)
        self.train_matrix = csr_matrix((np.ones(len(rows), dtype="float32"), (rows, cols)),
                                       shape=(n_users, n_movies))
        self.train_matrix.data[:] = 1
        counts = np.asarray(self.train_matrix.sum(axis=0)).ravel()
        self.popularity = (np.log1p(counts) / max(np.log1p(counts.max()), 1)).astype("float32")

        n_components = min(rank, n_users - 1, n_movies - 1)
        if n_components < 1:
            raise ValueError("SVD requires >=2 users and >=2 movies")
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_latent = svd.fit_transform(self.train_matrix).astype("float32")
        self.item_latent = svd.components_.T.astype("float32")
        self.svd_explained_variance = float(np.sum(svd.explained_variance_ratio_))

        split_genres = [str(s).split("|") for s in self.movies.genres.fillna("")]
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w-]+\b")
        self.genre_tfidf = vectorizer.fit_transform(
            self.movies.genres.fillna("").str.replace("|", " ", regex=False)
        ).astype("float32")
        self.genre_encoder = MultiLabelBinarizer()
        self.genre_onehot = self.genre_encoder.fit_transform(split_genres).astype("float32")
        return self

    def fit_neural(self, train, epochs=4, dim=32):
        from .neural import train_two_tower
        pairs = np.array([(self.user_to_idx[int(u)], self.movie_to_idx[int(m)])
                          for u, m in zip(train.userId, train.movieId)
                          if int(u) in self.user_to_idx and int(m) in self.movie_to_idx], dtype="int64")
        self.neural_user, self.neural_item = train_two_tower(
            pairs, self.genre_onehot, len(self.user_ids), len(self.movie_ids), epochs, dim)
        return self

    def features_for_user(self, user_id: int) -> np.ndarray:
        if int(user_id) not in self.user_to_idx:
            raise KeyError(f"Unknown training user: {user_id}")
        u = self.user_to_idx[int(user_id)]
        raw_cf = self.item_latent @ self.user_latent[u]
        # Build an individual normalized genre profile from the user's
        # training positives, without touching validation/test ratings.
        profile = np.asarray(self.genre_tfidf[list(self.seen[int(user_id)])].mean(axis=0)).ravel()
        profile /= max(float(np.linalg.norm(profile)), 1e-10)
        raw_content = np.asarray(self.genre_tfidf @ profile).ravel()
        if self.neural_user is None:
            neural_score = np.zeros(len(self.movie_ids), dtype="float32")
        else:
            neural_score = ((self.neural_item @ self.neural_user[u]) + 1) / 2
        return np.column_stack([self.popularity, _minmax(raw_cf),
                                np.clip(raw_content, 0, 1), neural_score]).astype("float32")

    def fit_ranker(self, held_rank_train: pd.DataFrame, negatives=35, seed=42):
        """LambdaMART trained on a held-out *later* positive for each user.

        Unobserved sampled items are weak/implicit negatives; they are not
        known impressions or dislikes. All features use the early train data.
        """
        import lightgbm as lgb
        rng = np.random.default_rng(seed)
        x, y, groups = [], [], []
        for event in held_rank_train.itertuples(index=False):
            user_id, movie_id = int(event.userId), int(event.movieId)
            if user_id not in self.user_to_idx or movie_id not in self.movie_to_idx:
                continue
            positive_idx = self.movie_to_idx[movie_id]
            known = self.seen[user_id] | {positive_idx}
            candidates = np.setdiff1d(np.arange(len(self.movie_ids)),
                                      np.fromiter(known, dtype="int64"), assume_unique=False)
            if len(candidates) < 1:
                continue
            sampled = rng.choice(candidates, size=min(negatives, len(candidates)), replace=False)
            sample_idx = np.concatenate([[positive_idx], sampled])
            features = self.features_for_user(user_id)[sample_idx]
            x.append(features)
            y.extend([1] + [0] * len(sampled))
            groups.append(len(sample_idx))
        if not x:
            raise ValueError("No users are eligible for supervised ranker training")
        self.ranker = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=100, num_leaves=15,
            learning_rate=0.05, min_child_samples=10, random_state=seed,
            verbosity=-1, n_jobs=2)
        self.ranker.fit(pd.DataFrame(np.vstack(x), columns=FEATURE_NAMES),
                        np.array(y, dtype=int), group=groups)
        self.ranker_groups = len(groups)
        return self

    @property
    def models(self):
        names = ["popularity", "svd", "genres", "hybrid"]
        if self.neural_user is not None:
            names.extend(["two_tower", "hybrid_neural"])
        if self.ranker is not None:
            names.append("lambdamart")
        return names

    def score_user(self, user_id: int, model="hybrid") -> np.ndarray:
        f = self.features_for_user(user_id)
        indices = {"popularity": 0, "svd": 1, "genres": 2, "two_tower": 3}
        if model in indices:
            if model == "two_tower" and self.neural_user is None:
                raise ValueError("Neural model was not trained")
            return f[:, indices[model]]
        if model == "hybrid":
            # Fixed classical baseline, unchanged when a neural model is trained.
            return f @ np.array([0.1, 0.5, 0.4, 0], dtype="float32")
        if model == "hybrid_neural" and self.neural_user is not None:
            # Keep neural blend separate so experiments don't overwrite the baseline.
            return f @ np.array([0.1, 0.35, 0.25, 0.3], dtype="float32")
        if model == "lambdamart" and self.ranker is not None:
            return self.ranker.predict(pd.DataFrame(f, columns=FEATURE_NAMES)).astype("float32")
        raise ValueError(f"Unknown or untrained model: {model}")

    def recommend(self, user_id, k=10, model="hybrid", extra_seen=()):
        scores = np.array(self.score_user(int(user_id), model), copy=True)
        exclude = set(self.seen[int(user_id)])
        exclude.update(self.movie_to_idx[int(v)] for v in extra_seen if int(v) in self.movie_to_idx)
        if exclude:
            scores[list(exclude)] = -np.inf
        winners = _top_finite_scores(scores, k)
        return self._result(winners, scores)

    def recommend_from_favorites(self, movie_ids, k=10):
        """Cold-start demo: content + item SVD + popularity, *not* personalized LambdaMART."""
        indexes = [self.movie_to_idx[int(m)] for m in movie_ids if int(m) in self.movie_to_idx]
        if not indexes:
            raise ValueError("Choose at least one movie present in the catalogue")
        profile = np.asarray(self.genre_tfidf[indexes].mean(axis=0)).ravel()
        profile /= max(float(np.linalg.norm(profile)), 1e-10)
        genre_score = np.clip(np.asarray(self.genre_tfidf @ profile).ravel(), 0, 1)
        normalized_item = normalize(self.item_latent, axis=1)
        item_profile = normalized_item[indexes].mean(axis=0)
        item_profile /= max(float(np.linalg.norm(item_profile)), 1e-10)
        cf_score = _minmax(normalized_item @ item_profile)
        if self.semantic_item is not None:
            semantic_profile = self.semantic_item[indexes].mean(axis=0)
            semantic_profile /= max(float(np.linalg.norm(semantic_profile)), 1e-10)
            semantic_score = np.clip((self.semantic_item @ semantic_profile + 1) / 2, 0, 1)
            scores = (0.45 * genre_score + 0.15 * cf_score +
                      0.35 * semantic_score + 0.05 * self.popularity)
        else:
            scores = 0.65 * genre_score + 0.30 * cf_score + 0.05 * self.popularity
        scores[indexes] = -np.inf
        return self._result(_top_finite_scores(scores, k), scores)

    def _result(self, indexes, scores):
        result = self.movies.iloc[np.asarray(indexes)].copy()
        result["score"] = np.round(scores[indexes], 4)
        return result.reset_index(drop=True)

    def save(self, output: str | Path):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: str | Path):
        # Loading a joblib/pickle artifact executes code; use ONLY trusted files.
        return joblib.load(path)
