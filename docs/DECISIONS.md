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

## D-003: Compute budget (provisional)
- The development machine has no NVIDIA GPU or other GPU, so neural training starts on **CPU only**.
- Plan: benchmark a small model and the embedding job on CPU before deciding whether external compute is needed. No external compute is committed to.
- Evidence so far (2026-11-10): profiling all 31.8 M transaction rows took 17.3 s, 3.6 s and 1.8 s for the three lazy scans on this machine; the heaviest scan peaked at about 7.3 GB, the simple per-date and per-article aggregations at about 1.3 GB and 0.9 GB. Data preparation is therefore feasible on this CPU machine. Neural training and embedding speed are still unmeasured.
- Revisit at Stage 6 (text embeddings) and Stage 9 (neural training) using measured runtime and memory.

## D-004: Raw-data location via git-ignored local config (final)
- The path to the raw CSVs is stored in `configs/local.yaml` (git-ignored; template in `configs/local.example.yaml`), not in code or committed config.
- Expected file and column specification is version-controlled in `configs/data.yaml`. It records what we expect from the Kaggle description; it is not evidence about the real files until the inventory is run.
- Validation policy (Stage 2): ERROR blocks later stages (required file missing/empty/unreadable, required column missing, duplicate headers); WARNING is reported but does not block (optional file missing, extra columns, implausible file size).
- Consequence: the repo is portable across machines and never contains absolute paths or raw data.

## D-005: Profile tables as text first (final)
- `articles.csv` is profiled with every column read as text (no schema inference), so nothing is coerced before we have measured how many values would fail integer conversion or lose information (for example leading zeros).
- Severity policy for the articles profile: key missing/non-integer/duplicate = ERROR (blocks later stages); leading-zero key, non-integer values in other integer columns, non-one-to-one code/name pairs and fully duplicated rows = WARNING; missing-value rates, whitespace and text-length statistics = INFO (they describe the data and are handled by later explicit rules).
- Consequence: dtype choices for the canonical dataset (Stage 4) will be based on measured evidence, not assumed.

## D-006: article_id is a 10-character text key; join by exact string equality (final)
- Evidence (2026-11-10): in both `articles.csv` and `transactions_train.csv` every `article_id` is a 10-character zero-padded digit string (articles: 105,542 of 105,542; transactions: 31,788,324 of 31,788,324 rows, all with a leading zero). 0 transaction rows lack a match in articles by exact comparison.
- Rule: keep `article_id` as text (Utf8) in every table and join by exact equality. Never cast to an integer, which would drop the leading zero.
- The zero-padding check in the transactions profile and `article_id_width` in `configs/data.yaml` remain as a safeguard but are not needed for the current files.
- Alternatives considered: converting to integer on both sides (rejected: needless information loss and a risk of mismatches); zero-padding transactions (not needed: already padded).

## D-007: Validation gate policy (final)
- One command, `python -m hm_forecast.data.validate`, runs the inventory, articles profile and transactions profile and writes `artifacts/validation/validation_summary.json`.
- If the inventory has blocking errors the remaining stages are skipped; otherwise every stage runs so all problems are visible at once.
- Exit codes: 0 passed, 1 blocking data errors, 2 a stage could not run. Only reports written by the current run are read.
- Trial runs (`--n-rows`) are not available in the gate: a gate validates the whole file. `--skip-transactions` exists for quick checks and is recorded in the summary as `full_validation: false`.
- The summary records file sizes and modification times, optional SHA-256 hashes (`--sha256`, slow for the 3.5 GB file) and the Git commit, so a validation result can be tied to exact data and code.
- Consequence: later stages (starting with Stage 4) call this gate before any expensive processing.
