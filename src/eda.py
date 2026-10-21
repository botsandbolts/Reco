"""Basic exploratory analysis and reproducible static charts."""
from pathlib import Path
import argparse
import pandas as pd
from .data import ensure_dataset, chronological_split


def run(data_dir="data", out_dir="reports"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ratings, movies = ensure_dataset(data_dir)
    split = chronological_split(ratings)
    print(f"Rows: {len(ratings):,}; unique users: {ratings.userId.nunique():,}; "
          f"rated movies: {ratings.movieId.nunique():,}; catalogue: {len(movies):,}")
    print("Rating counts:")
    print(ratings.rating.value_counts().sort_index().to_string())
    print("Chronological positive splits:")
    print({name: len(getattr(split, name)) for name in ('train', 'rank_train', 'valid', 'test')})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ratings.rating.hist(ax=axes[0], bins=10)
    axes[0].set(xlabel="Movie rating (stars)", ylabel="Interactions", title="Rating distribution")
    ratings.groupby('userId').size().hist(ax=axes[1], bins=40)
    axes[1].set(xlabel="Ratings per user", ylabel="Users", title="User activity")
    fig.tight_layout()
    fig.savefig(out / "eda.png", dpi=160)
    plt.close(fig)
    print(f"Saved {out / 'eda.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="reports")
    run(**vars(parser.parse_args()))
