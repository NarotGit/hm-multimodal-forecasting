# Decision Log

Status values: **provisional** (may change with evidence) or **final**.

## D-001: Repository location and Git workflow (final)
- Repository: `C:\Users\Narot\Documents\GitHub\hm-multimodal-forecasting`, remote on GitHub, default branch `main`.
- Raw H&M CSVs are stored **outside** the repository for now. Whether to copy them into `data/raw/` or reference the external folder via config is decided in Stage 2.
- `.gitignore` blocks `articles.csv`, `customers.csv`, `transactions_train.csv` anywhere in the repo. Any committed test sample must use a different name or format.
- Consequence: raw data and generated artifacts can never be committed by accident.

## D-002: Python version and dependency policy (provisional)
- **Python 3.12** (supported range in `pyproject.toml`: `>=3.12,<3.14`).
- Evidence: PyPI metadata checked on 2026-11-01. NumPy 2.5 requires `>=3.12`; Polars, PyArrow, PyTorch, Streamlit, scikit-learn and the other planned packages all declare support for 3.12. PyPI metadata does not prove that Windows wheels install cleanly; that is verified by the install in Step 2.
- Alternatives: 3.13 (works with everything listed, but newer and less proven for deep-learning wheels); 3.11 (excluded by NumPy 2.5).
- **Core dependencies only at this stage:** numpy, polars, pyarrow, pyyaml (+ pytest for development). Each has a lower bound at the version checked and an upper bound at the next major version.
- PyTorch, sentence-transformers, MLflow, Streamlit and Plotly are deliberately **not** installed yet. They are added in the stage that first needs them, with a compatibility check at that time (PyTorch especially: CPU vs CUDA wheels depend on the user's hardware).
- `requirements-lock.txt` records the exact versions that were actually installed and tested on the user's Windows machine.
- Consequence: a small, fast, low-risk environment now; the dependency set grows only with justification.
