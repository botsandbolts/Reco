"""Optional: create sentence-transformer item embeddings for cold-start search.

Install `pip install -r requirements-optional-nlp.txt` and run
`python -m src.semantic`. This requires a model download, unlike standard train.
"""
import argparse
import numpy as np
from sklearn.preprocessing import normalize
from .recommender import HybridRecommender


def run(model_path="artifacts/recommender.joblib", checkpoint="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    rec = HybridRecommender.load(model_path)
    texts = [f"Movie: {title}. Genres: {genres.replace('|', ', ')}."
             for title, genres in zip(rec.movies.title, rec.movies.genres)]
    embedder = SentenceTransformer(checkpoint)
    vectors = embedder.encode(texts, batch_size=128, show_progress_bar=True,
                             convert_to_numpy=True, normalize_embeddings=True)
    rec.semantic_item = normalize(np.asarray(vectors, dtype="float32"))
    rec.save(model_path)
    print(f"Stored semantic embeddings for {len(texts)} films in {model_path}")
    print("NOTE: Title/genre-only text limits the value of semantic search.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="artifacts/recommender.joblib")
    parser.add_argument("--checkpoint", default="sentence-transformers/all-MiniLM-L6-v2")
    run(**vars(parser.parse_args()))
