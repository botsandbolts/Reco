"""Experimental BPR two-tower recommender with fresh mixed negative sampling.

Drop-in alternative to src/neural.py.  Trains on training interactions only.
This is an experimental implementation: compare to SVD using validation first.
"""
from __future__ import annotations
import numpy as np


def _sample_negatives(positive_users, user_seen, n_items, popularity_prob, rng):
    """Half popularity-biased and half uniform unobserved-item sampling."""
    n = len(positive_users)
    items = rng.integers(0, n_items, size=n, dtype=np.int64)
    biased = rng.random(n) < 0.5
    items[biased] = rng.choice(n_items, size=int(biased.sum()), p=popularity_prob)
    invalid = np.array([int(i) in user_seen[int(u)]
                        for u, i in zip(positive_users, items)], dtype=bool)
    retries = 0
    while invalid.any():
        if retries > 100:
            raise RuntimeError("Unable to sample unseen negatives")
        target = np.where(invalid)[0]
        candidates = rng.integers(0, n_items, size=len(target), dtype=np.int64)
        biased = rng.random(len(target)) < 0.5
        candidates[biased] = rng.choice(n_items, size=int(biased.sum()), p=popularity_prob)
        items[target] = candidates
        invalid[target] = [int(i) in user_seen[int(u)]
                           for u, i in zip(positive_users[target], candidates)]
        retries += 1
    return items


def train_two_tower(interactions, genres, n_users, n_items,
                    epochs=15, dim=32, seed=42, negatives_per_positive=3):
    """Return normalized user/item embeddings after pairwise-ranking training.

    Each epoch resamples a mix of popularity-biased and uniformly random
    non-observed movies per positive user/movie event.  This avoids repeatedly
    training against the same tiny set of easy uniform negatives.
    """
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "Install neural dependencies: python -m pip install -r requirements-neural.txt"
        ) from exc

    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be >= 1")
    interactions = np.asarray(interactions, dtype=np.int64)
    if interactions.ndim != 2 or interactions.shape[1] != 2 or len(interactions) == 0:
        raise ValueError("interactions must have shape (n, 2) with n > 0")
    if genres.shape[0] != n_items:
        raise ValueError("genres must contain one row per item")

    user_seen = {}
    for u, i in interactions:
        if not (0 <= u < n_users and 0 <= i < n_items):
            raise ValueError("out-of-range user/item index")
        user_seen.setdefault(int(u), set()).add(int(i))
    if any(len(items) >= n_items for items in user_seen.values()):
        raise ValueError("Cannot sample negatives from users with a full catalogue")

    np_rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    item_counts = np.bincount(interactions[:, 1], minlength=n_items).astype("float64")
    # Bias some negatives towards higher-engagement items; still sample broadly.
    popularity = np.power(item_counts + 0.1, 0.75)
    popularity /= popularity.sum()

    genre_values = np.asarray(genres, dtype=np.float32)
    genre_values /= np.maximum(
        np.linalg.norm(genre_values, axis=1, keepdims=True), 1.0
    )
    genre_t = torch.tensor(genre_values, dtype=torch.float32, device=device)

    class TwoTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.user_embedding = nn.Embedding(n_users, dim)
            self.item_embedding = nn.Embedding(n_items, dim)
            self.genre_projection = nn.Linear(genre_t.shape[1], dim, bias=False)
            nn.init.normal_(self.user_embedding.weight, std=0.08)
            nn.init.normal_(self.item_embedding.weight, std=0.08)

        def embed_user(self, user):
            return nn.functional.normalize(self.user_embedding(user), dim=1)

        def embed_item(self, item):
            v = self.item_embedding(item) + self.genre_projection(genre_t[item])
            return nn.functional.normalize(v, dim=1)

        def score(self, users, items):
            return 8.0 * (self.embed_user(users) * self.embed_item(items)).sum(dim=1)

    model = TwoTower().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    positive_users = np.repeat(interactions[:, 0], negatives_per_positive)
    positive_items = np.repeat(interactions[:, 1], negatives_per_positive)

    for epoch in range(epochs):
        negative_items = _sample_negatives(
            positive_users, user_seen, n_items, popularity, np_rng
        )
        triples = torch.from_numpy(
            np.column_stack([positive_users, positive_items, negative_items])
        )
        loader = DataLoader(TensorDataset(triples), batch_size=2048, shuffle=True)
        model.train()
        running_loss = 0.0
        processed = 0
        for (batch,) in loader:
            batch = batch.to(device)
            u, pos_i, neg_i = batch[:, 0], batch[:, 1], batch[:, 2]
            diff = model.score(u, pos_i) - model.score(u, neg_i)
            loss = nn.functional.softplus(-diff).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * len(u)
            processed += len(u)
        print(f"BPR two-tower epoch {epoch+1}/{epochs}: "
              f"training loss={running_loss/processed:.4f} device={device}")

    model.eval()
    with torch.no_grad():
        user_vectors = nn.functional.normalize(model.user_embedding.weight, dim=1)
        item_vectors = model.item_embedding.weight + model.genre_projection(genre_t)
        item_vectors = nn.functional.normalize(item_vectors, dim=1)
    return (user_vectors.cpu().numpy().astype("float32"),
            item_vectors.cpu().numpy().astype("float32"))
