"""Streamlit recommendation app. Deploy using `streamlit run app.py`."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from src.recommender import HybridRecommender

st.set_page_config(page_title="Hybrid Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Hybrid Movie Recommender")
st.caption("Research demo • MovieLens latest-small / GroupLens • No commercial use without permission")

ARTIFACT = Path("artifacts/recommender.joblib")
if not ARTIFACT.exists():
    st.info("The interactive demo is in setup mode: no trusted trained model artifact was found.")
    st.markdown("""
    ### What this project demonstrates
    - **Recommendation:** popularity, collaborative SVD, genre similarity, and hybrid ranking.
    - **Learning to rank:** LightGBM LambdaMART; optional PyTorch two-tower model.
    - **Evaluation:** chronological positive holdouts, full-catalogue Recall@10 and NDCG@10.
    - **Transparency:** training scripts, automated tests, and a methodology/model card.

    **Run locally or in Codespaces:**
    """)
    st.code("python -m pip install -r requirements-dev.txt\npython -m src.train\nstreamlit run app.py", language="bash")
    st.caption("MovieLens data are downloaded by the training command and have separate "
               "GroupLens conditions. No benchmark result is shown without a generated report.")
    st.stop()

@st.cache_resource(show_spinner="Loading recommendation engine …")
def load_engine():
    return HybridRecommender.load(ARTIFACT)

engine = load_engine()
explore, favorites, benchmark, methodology = st.tabs(
    ["Known user", "Choose favorites (cold start)", "Benchmark", "Methodology"])

with explore:
    st.subheader("Recommendations for an existing MovieLens user")
    left, right = st.columns([1, 3])
    with left:
        user_id = st.selectbox("Anonymous training user", engine.user_ids.tolist())
        model_name = st.selectbox("Recommendation method", engine.models,
                                  index=engine.models.index("hybrid"))
        n = st.slider("How many recommendations?", 5, 20, 10)
    with right:
        st.write("**Movies in this user's training history**")
        seen_ids = sorted(engine.seen[int(user_id)])
        st.dataframe(engine.movies.iloc[seen_ids][["title", "genres"]].head(12),
                     hide_index=True, use_container_width=True)
    st.write("**Top recommendations**")
    st.dataframe(engine.recommend(user_id, k=n, model=model_name),
                 hide_index=True, use_container_width=True)
    st.caption("The training history deliberately omits ranker-training, validation "
               "and test positives, which are kept separate during evaluation.")

with favorites:
    st.subheader("Recommendations without a user account")
    st.write("Select movies you enjoyed. The system combines genre similarity, "
             "item embeddings, and a small popularity prior. It does **not** "
             "claim to use the known-user ranking model for new users.")
    labels = {f"{row.title} (ID {row.movieId})": int(row.movieId)
              for row in engine.movies.itertuples(index=False)}
    selections = st.multiselect("Favorite movies", options=list(labels),
                                default=[name for name in labels if name.startswith("Toy Story (1995)")][:1],
                                max_selections=7)
    if selections:
        st.dataframe(engine.recommend_from_favorites([labels[k] for k in selections], k=10),
                     hide_index=True, use_container_width=True)
    else:
        st.info("Choose at least one title.")

with benchmark:
    metrics_file = Path("artifacts/metrics.json")
    st.subheader("Out-of-sample ranking benchmark")
    if metrics_file.exists():
        content = json.loads(metrics_file.read_text(encoding="utf-8"))
        split = st.radio("Split", ["test", "validation"], horizontal=True)
        df = pd.DataFrame(content[split]).T
        st.dataframe(df, use_container_width=True)
        if "ndcg_at_k" in df:
            st.bar_chart(df["ndcg_at_k"])
        st.caption(f"K={content['k']}; positive rating threshold "
                   f"{content['positive_rating_threshold']}. Full-catalogue ranking.")
    else:
        st.warning("metrics.json missing. Train the recommender to generate it.")

with methodology:
    st.markdown("""
    **Models:** popularity, TruncatedSVD collaborative filtering, TF-IDF genres,
    weighted hybrid, optional PyTorch two-tower, and LightGBM LambdaMART.

    **Split:** for users with at least four positive ratings (≥4 stars), reserve
    each user's last three positives in time order for ranker training, validation,
    and test. The model only sees earlier positives.

    **Ranking:** evaluate every non-excluded catalogue item, not just sampled
    negatives. Holdouts are one positive per user, so Recall@K is hit rate@K.

    **Limitations:** ratings are not impression logs; unobserved items are not
    verified dislikes. Historical MovieLens data and per-user chronological
    splits can still exhibit popularity, selection, and cross-user temporal bias.
    Offline metrics do not measure live business uplift. Cold-start uses item
    metadata, not a learned identity for the visitor.

    **Attribution:** [MovieLens / GroupLens](https://grouplens.org/datasets/movielens/)
    at the University of Minnesota. The data's license forbids commercial or
    revenue-bearing use without permission.
    """)
