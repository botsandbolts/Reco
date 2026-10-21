"""Download MovieLens and construct per-user chronological positive-only splits."""
from pathlib import Path
import io
from dataclasses import dataclass
import zipfile
import requests
import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


@dataclass
class Splits:
    train: pd.DataFrame
    rank_train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def ensure_dataset(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(data_dir)
    target = base / "ml-latest-small"
    movies_file = target / "movies.csv"
    ratings_file = target / "ratings.csv"
    if not (movies_file.is_file() and ratings_file.is_file()):
        base.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MovieLens Small from {URL}")
        response = requests.get(URL, timeout=90)
        response.raise_for_status()
        # Only extract files expected in the official archive.
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            for name in ("ml-latest-small/movies.csv", "ml-latest-small/ratings.csv"):
                dest = base / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(name))
    ratings = pd.read_csv(ratings_file)
    movies = pd.read_csv(movies_file)
    return ratings, movies


def chronological_split(ratings: pd.DataFrame, threshold: float = 4.0) -> Splits:
    """Hold out the final 3 positive events per user when >=4 positives.

    'Positive' means an observed rating >= threshold; other ratings are not
    treated as negative exposures. Users with <4 positives stay train-only.
    Timestamp ties are deterministically ordered by movieId.
    """
    required = {"userId", "movieId", "rating", "timestamp"}
    if not required.issubset(ratings.columns):
        raise ValueError(f"ratings must include {sorted(required)}")
    positive = ratings.loc[ratings.rating >= threshold].copy()
    positive = positive.sort_values(["userId", "timestamp", "movieId"], kind="stable")
    positive = positive.drop_duplicates(["userId", "movieId"], keep="last")
    position = positive.groupby("userId").cumcount()
    count = positive.groupby("userId")["userId"].transform("size")
    eligible = count >= 4
    train = positive.loc[~eligible | (position < count - 3)].copy()
    rank_train = positive.loc[eligible & (position == count - 3)].copy()
    valid = positive.loc[eligible & (position == count - 2)].copy()
    test = positive.loc[eligible & (position == count - 1)].copy()
    return Splits(train.reset_index(drop=True), rank_train.reset_index(drop=True),
                  valid.reset_index(drop=True), test.reset_index(drop=True))
