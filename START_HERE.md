# Start here — 10 practical steps

**What you receive:** complete Python source, reproducible training scripts,
classic and neural recommenders, LambdaMART reranking, automated tests,
MovieLens data downloader, a Streamlit demonstration, and a research model card.
Everything runs in a browser development environment, using free plans within
quotas. No paid LLM API is required.

## Import the ZIP into GitHub Codespaces

1. Download `hybrid_recommender_complete.zip` from ChatGPT.
2. Go to https://github.com/new and create a new GitHub repository, with
   **Add a README file** checked so the `main` branch exists.
3. Select **Code → Codespaces → Create codespace on main**.
4. In the browser-based VS Code Explorer, drag-and-drop the ZIP into the root
   of the new repository. (Use a desktop browser if drag/drop on mobile is hard.)
5. In the Codespaces **Terminal → New Terminal**, enter:

```bash
unzip hybrid_recommender_complete.zip
cp -R hybrid-recommender/. .
rm -rf hybrid-recommender
rm hybrid_recommender_complete.zip
python -m pip install -r requirements-dev.txt
```

**Check that `app.py` and `requirements.txt` appear at the repository root.**
Codespaces comes with Python pre-installed; choose a Python 3.12 Codespace if
possible for consistency with Streamlit's deployment environment.

## Run the project

```bash
# 1. Validate without using any internet datasets
python -m pytest -q

# 2. Download and examine MovieLens latest-small
python -m src.eda

# 3. Train popularity + SVD + TF-IDF + LambdaMART
python -m src.train

# 4. Launch the demo; open forwarded port 8501
streamlit run app.py
```

## Optional, more impressive version

```bash
# Add PyTorch, retrain with a genuine two-tower model
python -m pip install -r requirements-neural.txt
python -m src.train --neural --epochs 4

# Optional NLP feature, after training
python -m pip install -r requirements-optional-nlp.txt
python -m src.semantic

# Relaunch demo
streamlit run app.py
```

## Share your work

```bash
# In your Codespaces repo

git add .
git commit -m "Add end-to-end hybrid recommendation project"
git push

# Only after confirming the MovieLens conditions for your noncommercial demo:
git add -f artifacts/recommender.joblib artifacts/metrics.json
git commit -m "Include trained noncommercial demo artifacts"
git push
```

Visit https://share.streamlit.io/, choose the GitHub repository, select
**`app.py`**, choose Python 3.12 in advanced settings and deploy.

Read `README.md` for the full 11-step walk-through and `reports/MODEL_CARD.md`
for methodology, scientific limitations and the upstream dataset terms.

**Note:** This project was automatically tested using synthetic data; the
MovieLens download and Streamlit cloud deployment must be run in your own online
environment, which has network access and the relevant installed packages.
