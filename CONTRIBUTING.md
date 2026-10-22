# Contributing and reproducibility

Pull requests are welcome for bug fixes, tests and experimentally justified ranking improvements.

1. Use Python 3.12; install with 'python -m pip install -r requirements-dev.txt'.
2. Run 'python -m pytest -q' and 'python -m compileall -q src tests app.py'.
3. Do not commit local datasets, secret files, checkpoints or generated metrics without reviewing their licenses and provenance.
4. Include a synthetic regression test for each behavioral bug.
5. When claiming improved recommendation quality, record the same holdout cohort, date/cutoff, unseen-item exclusions, candidate protocol, K, seed, model and config for both runs.
6. Keep test data out of ranking feature construction. Never call an unrated movie a confirmed negative.
7. Explain any changes to the data split or metric denominator in the PR.

Security: load only trusted model files, since pickle/joblib and PyTorch checkpoint loading can run code. Report potential vulnerabilities privately to the repository maintainer rather than putting exploits or secrets in an issue.