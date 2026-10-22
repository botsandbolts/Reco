# Start here

This is the **maintained MovieLens Small** entry point. The 1M/32M FM, TIGER and scalable research prototypes are currently retained separately under [research-patches](research-patches/) and described in [Research status](docs/RESEARCH_STATUS.md).

## Run in GitHub Codespaces

Open this repository in Codespaces and use its terminal:

```bash
cd /workspaces/Reco
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m src.train
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open port **8501** in Codespaces. The data downloader is part of 'src.train'; no dataset needs to be committed. Generated models and metric reports live locally in 'artifacts/'.

Without model artifacts, the Streamlit app displays a setup/overview page instead of claiming that a demo is already trained. Keep artifacts from untrusted sources out of the app.

## Explore optional components

```bash
python -m pip install -r requirements-neural.txt
python -m src.train --neural --epochs 4
python -m pip install -r requirements-optional-nlp.txt
python -m src.semantic
python -m src.benchmark_semantic
```

Learn about scientific caveats in [reports/MODEL_CARD.md](reports/MODEL_CARD.md). Do **not** mix NDCG values from the original per-user split with the chronological 32M experiments unless they have been re-evaluated on one matching cohort.