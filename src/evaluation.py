"""Full-catalogue, leave-last-out metrics on chronologically held events."""
from __future__ import annotations
import numpy as np


def evaluate_models(model, holdout, already_seen=None, k=10):
    already_seen = already_seen or {}
    results = {}
    for method in model.models:
        hits, ndcg, n = 0, 0.0, 0
        catalogue_coverage = set()
        for event in holdout.itertuples(index=False):
            uid, mid = int(event.userId), int(event.movieId)
            if uid not in model.user_to_idx or mid not in model.movie_to_idx:
                continue
            target = model.movie_to_idx[mid]
            if target in model.seen[uid] or mid in already_seen.get(uid, set()):
                raise ValueError("Evaluation leakage detected in exclusion history")
            scores = np.array(model.score_user(uid, method), copy=True)
            exclude = set(model.seen[uid])
            exclude.update(model.movie_to_idx[int(v)] for v in already_seen.get(uid, set())
                           if int(v) in model.movie_to_idx)
            if exclude:
                scores[list(exclude)] = -np.inf
            top = np.argsort(-scores, kind="stable")[:k]
            catalogue_coverage.update(int(x) for x in top)
            if target in top:
                hits += 1
                rank = int(np.flatnonzero(top == target)[0]) + 1
                ndcg += 1.0 / np.log2(rank + 1)
            n += 1
        results[method] = {
            "users_evaluated": n,
            "recall_at_k": round(hits / n, 5) if n else None,
            "ndcg_at_k": round(ndcg / n, 5) if n else None,
            "catalog_coverage_at_k": round(len(catalogue_coverage) / len(model.movie_ids), 5),
        }
    return results


def map_seen(df):
    return {int(u): set(map(int, group.movieId)) for u, group in df.groupby("userId")}


def union_seen(*dfs):
    combined = {}
    for df in dfs:
        for u, values in map_seen(df).items():
            combined.setdefault(u, set()).update(values)
    return combined
