# Hybrid Movie Recommender — End-to-End Data Science Portfolio

**Full runnable research project:** data acquisition → chronological validation →
classical collaborative filtering → modern two-tower recommendation → learning to
rank → held-out evaluation → interactive Streamlit application → GitHub/Cloud deployment.

The repository is designed for **GitHub Codespaces** (browser IDE), **Kaggle** or
**Colab** (optional GPU), and **Streamlit Community Cloud** (free hosted demo,
subject to free-tier limits). Start with **MovieLens latest-small**. No API key
or paid infrastructure is needed. MovieLens itself has noncommercial-use
restrictions; see [Data rights](#data-rights-and-attribution).

## Contents

```
app.py                          # Interactive demonstration
src/data.py                     # Download and chronological split
src/eda.py                      # Exploratory charts
src/recommender.py              # SVD + content + hybrid + LambdaMART
src/neural.py                   # Optional PyTorch two-tower training
src/semantic.py                 # Optional sentence-transformer item embeddings
src/evaluation.py               # Full-catalogue Recall, NDCG, coverage
src/train.py                    # Training + evaluation CLI
notebooks/01_exploration.ipynb  # Data inspection
notebooks/02_results.ipynb      # Reproducible evaluation reports
tests/test_pipeline.py          # Synthetic tests, data-free
reports/MODEL_CARD.md           # Methodology and limitations
.github/workflows/tests.yml     # GitHub Actions CI
requirements.txt                # Base Python environment
requirements-neural.txt         # Adds PyTorch
requirements-optional-nlp.txt   # Adds sentence-transformers
requirements-dev.txt            # Adds pytest/matplotlib
Dockerfile                      # Optional container deployment
```

## Step 1 — Create a free browser-based development environment

1. Create/sign into a free account at https://github.com/.
2. Create a repository named `hybrid-recommender` (public when ready to share).
3. In the repository choose **Code → Codespaces → Create codespace on main**.
4. Download the ZIP distributed with these instructions, and upload/drag the
   **contents** of the `hybrid-recommender` folder to the repository. You can use
   **Add file → Upload files** on GitHub's website, then open Codespaces.
5. Open a Codespaces terminal (**Terminal → New Terminal**). Check that you are
   in the directory containing `requirements.txt`.

No local Python installation is required. You can also unzip the project in
Colab/Kaggle, but Codespaces is simplest for editing a multi-file repository.

## Step 2 — Install Python dependencies

Use **Python 3.12** where possible (Community Cloud supports choosing it):

```bash
python --version
python -m pip install -r requirements-dev.txt
```

This installs NumPy, Pandas, SciPy, scikit-learn, LightGBM, Streamlit, pytest,
matplotlib, and related libraries.

**To train the neural model** (CPU works; GPU is faster):

```bash
python -m pip install -r requirements-neural.txt
```

If you use Kaggle, PyTorch may already be installed. GPU availability depends
on the provider's current quotas. The default training command **does not**
require PyTorch.

## Step 3 — Test the code before downloading any data

```bash
python -m pytest -q
```

The unit/integration tests generate a tiny synthetic ratings dataset. They
check temporal data separation, the absence of rated movies among
recommendations, evaluation behavior, LambdaMART, and model serialization.
A dedicated optional neural test runs only if PyTorch is installed.

## Step 4 — Fetch the public MovieLens dataset and examine it

```bash
python -m src.eda
```

The script automatically downloads the ~1 MB `ml-latest-small.zip` archive
from GroupLens, reads its official `ratings.csv` and `movies.csv`, displays
rating distributions, counts users/movies, and saves `reports/eda.png`.
The raw data are saved below `data/` (excluded from Git).

Dataset: https://grouplens.org/datasets/movielens/latest/

### How the project treats ratings

A rating >= **4 stars** counts as a positive interaction. A 1–3.5 rating is
**not** used as a confirmed negative impression; this is a conservative
positive-only proxy, not an implicit click dataset. For each user with at least
four positives, hold their last three positives aside in chronological order:

```
Earlier positives                   Last three positives
[         FIT SVD / TWO TOWER       ][ranker training][validation][test]
```

Later ranker training learns to order an observed positive above sampled
unobserved items. Validation guides development. The test is held back for the
final report. This is per-user chronological splitting, **not** a global time
cutoff (see Model Card for bias implications).

## Step 5 — Train the classical + LambdaMART recommender

```bash
python -m src.train
```

Runs end-to-end:

- Counts popularity of positive ratings.
- Fits truncated SVD of the binary user–item interaction matrix.
- Fits a TF-IDF movie-genre representation.
- Constructs per-user content profiles from **training** ratings only.
- Trains a LightGBM LambdaMART ranker on the ranker-training holdout with
  sampled *unobserved* alternatives.
- Evaluates **all available catalogue movies** on validation/test (no sampled
  negatives during evaluation), excluding previously observed positives.
- Writes `artifacts/recommender.joblib` and `artifacts/metrics.json`.

The script prints Recall@10, NDCG@10 and catalogue coverage@10 for
`popularity`, `svd`, `genres`, `hybrid`, and `lambdamart`.

To run only the simple baselines, skip LambdaMART:

```bash
python -m src.train --no-ranker
```

## Step 6 — Train a two-tower neural recommender

Once the classical system works, train the more demanding model:

```bash
python -m src.train --neural --epochs 4
```

The PyTorch two-tower architecture has:

- **User tower:** trainable `nn.Embedding(user_id, dimension)`.
- **Item tower:** trainable item ID embeddings + linear projection of movie
  genre multi-hot vectors.
- **Similarity:** normalized dot product of the two towers.
- **Loss:** binary cross-entropy with one randomly sampled *unobserved* item
  per positive (provisional negative, not a verified dislike).

The resulting neural user/item arrays are exported as compact NumPy arrays in
the trusted serialized artifact; the **deployed app does not need PyTorch**.

Re-running training overwrites model and metrics artifacts. Capture previous
metrics if you wish to compare two distinct training runs.

To move neural training to Kaggle, upload or clone the project there and run
`!python -m src.train --neural --epochs 4` in a notebook cell with the GPU
accelerator enabled. Download the resulting `artifacts/` directory and upload
it back to your GitHub repository. The small dataset should also work on CPU.

## Step 7 — Optional language-model item embeddings

This is an **optional** NLP extension for visitors without a known user ID:

```bash
python -m pip install -r requirements-optional-nlp.txt
python -m src.semantic
```

This downloads `sentence-transformers/all-MiniLM-L6-v2`, embeds movie titles
and genres, and stores item vectors in the artifact. The cold-start engine
then combines genre TF-IDF, item SVD, semantic similarity, and popularity.
This is **not** LLM answer generation or RAG; with only titles and genres, the
semantic benefit may be limited. Only obtain plot synopses with suitable rights
if you decide to expand the item text later.

## Step 8 — Launch the interactive application

```bash
streamlit run app.py
```

In Codespaces, open the forwarded port **8501** (Ports panel). The app contains:

1. **Known user:** choose an anonymous MovieLens training user, select the
   baseline/neural/hybrid/LambdaMART method, and view recommendations.
2. **Cold start:** select favorite films without a user ID; this uses an
   item-item similarity blend rather than pretending to infer a user tower.
3. **Benchmark:** view measured validation and final test ranking metrics.
4. **Methodology:** explain the data, evaluation and limitations to recruiters.

The app **loads pre-trained artifacts**. It never retrains inside the Streamlit
process, which is safer for free-tier CPU/RAM limits.

## Step 9 — Publish the project on GitHub

Inside Codespaces:

```bash
git status
git add .
git commit -m "Build reproducible hybrid recommendation system"
git push
```

The `.gitignore` intentionally excludes `data/` and `artifacts/`. Do **not**
commit raw data by default. For the live demonstration, you need the generated
model and metric artifacts. If you are comfortable distributing derived models
under the **MovieLens dataset's research/noncommercial conditions**, explicitly
add only those two files:

```bash
git add -f artifacts/recommender.joblib artifacts/metrics.json
git commit -m "Add noncommercial demo model and evaluation summary"
git push
```

Never use `joblib.load` with a model supplied by someone you do not trust.

## Step 10 — Deploy publicly, free

1. Open https://share.streamlit.io/ and authenticate with GitHub.
2. Select **Create app** and choose your GitHub repository.
3. Select the `main` branch and **`app.py`** as the entry-point file.
4. Select **Python 3.12** in Advanced settings if available.
5. Deploy. The service installs packages from `requirements.txt` and starts
   your app using the trusted committed artifact.

Resource limits and free plans can change. If the deployment reports no
model file, confirm `artifacts/recommender.joblib` exists on the branch selected.

## Step 11 — Prepare the portfolio write-up

Document:

1. **Problem** — rank unseen movies for users using interactions and content.
2. **Hypotheses** — SVD > popularity; neural improves retrieval; learning to
   rank improves NDCG over a fixed weighted hybrid.
3. **Data** — counts and distributions; label choice and chronological splits.
4. **Methods** — baselines, factorization, two-tower, LambdaMART.
5. **Validation** — full-catalogue Recall@10/NDCG@10/coverage, no test tuning.
6. **Ablations** — repeat training with and without neural features; discuss
   cold-start users/items and popularity/selection bias.
7. **Engineering** — reproducible scripts, CI tests, portable app artifact.
8. **Limitations** — missing-not-at-random feedback and no online A/B test.

The honest story is stronger than a flashy accuracy number. Include measured
figures only after actually running the scripts. See `reports/MODEL_CARD.md`.

## What the supplied code does not claim

- **MovieLens 32M is not supported out of the box.** The prototype's dense
  per-user score arrays and LambdaMART full-catalogue inference are tuned for
  Small. To scale: convert to Parquet, batch with PyArrow/Polars, use more
  efficient sparse implicit ALS/mini-batch training, ANN retrieval (FAISS or
  ScaNN), then rerank only the top ~200 retrieved items.
- The two-tower has simple random negatives rather than adaptive in-batch or
  hard negatives; these are sensible next experiments.
- It does not use user demographic features, external API keys, paid LLMs,
  real-time streaming logs, or proven business causal uplift.
- TruncatedSVD on a binary interaction matrix is **not** an alternating least
  squares implicit-feedback objective. It is a solid lightweight baseline.

## Data rights and attribution

Data: **MovieLens latest-small**, GroupLens Research, University of Minnesota.
Source: https://grouplens.org/datasets/movielens/latest/
License: https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html

GroupLens requires citation/attribution, forbids claims of endorsement, and
requires permission for commercial or revenue-bearing use. Derived data that
are redistributed must respect the upstream terms. This project is a
noncommercial research/portfolio demonstration and is **not** endorsed by
GroupLens or the University of Minnesota.

Original project code is offered under the MIT License (see `LICENSE`); this
license does not override the dataset's separate terms.
