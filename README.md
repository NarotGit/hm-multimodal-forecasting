# Multimodal Probabilistic Forecasting of Retail Purchase Activity

Dissertation project: **Multimodal Probabilistic Forecasting of Retail Purchase Activity Using Product Metadata and Text Embeddings.**

> **Status: Stage 1 (project initialization) in progress.** This repository currently contains only the project skeleton, environment definition and documentation. No data pipeline, models, or results exist yet.

## Research objective

Develop and evaluate a neural model that produces probabilistic (quantile) forecasts of daily purchase activity at the H&M article level, and test whether text embeddings of product names and descriptions improve forecast quality and uncertainty calibration compared with an otherwise comparable model without text.

Text benefit is treated as a **hypothesis, not an assumption**. A negative or mixed result is an acceptable finding if the experimental design is sound.

### Research questions

- **RQ1:** How well can a neural model forecast daily article-level purchase-event counts probabilistically?
- **RQ2:** Do product-text embeddings improve probabilistic forecasts beyond numerical and structured product features?
- **RQ3:** Where is text most useful, particularly for articles with sparse or shorter purchase histories?
- **RQ4:** Are the predicted intervals calibrated and informative?

## Target definition and limitations

The target is the **daily purchase-event count per `article_id`**: the number of transaction rows for that article on that date.

- It is a count of recorded purchase events, **not** units sold and **not** necessarily distinct orders (the data has no quantity or order-ID column).
- There is no inventory information, so a zero is **not** evidence of zero demand and the project does not estimate lost sales.
- Article launch and withdrawal dates are absent, so missing article-days are not automatically treated as zeros. An eligibility and observation-window policy will be decided after exploratory analysis.
- Initial forecast horizon: 7 days.

## Dataset

[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) (Kaggle). Planned files: `transactions_train.csv`, `articles.csv`; `customers.csv` only if later justified. Product images are out of scope.

- The dataset was released for a recommendation task; using it for count forecasting is an adaptation made by this project.
- Use is subject to the Kaggle competition rules and data terms. Accept them on Kaggle before downloading.
- **Raw data is never committed to Git.** It lives in `data/raw/` (git-ignored) and is obtained by each user directly from Kaggle.

## Planned technology stack

Python, Polars (lazy API), Parquet/PyArrow, NumPy, PyTorch, Sentence Transformers, MLflow (local), Streamlit + Plotly, pytest, Git/GitHub. Exact Python and package versions will be chosen after compatibility checks and recorded in `docs/DECISIONS.md`.

Out of scope: Spark, Beam, TFX, Airflow, image models, LLM fine-tuning, paid cloud infrastructure.

## How development proceeds

Work is incremental: one small, verifiable step at a time. Each step has stated acceptance criteria that are checked against observed results before moving on. The planned stages are:

1. Project initialization
2. Data access and validation
3. Exploratory data analysis and target design
4. Canonical dataset construction
5. Feature engineering
6. Text embeddings
7. Temporal splitting and backtesting
8. Baselines
9. Neural forecasting
10. Evaluation and calibration
11. Experiment tracking
12. Batch inference
13. Dashboard
14. Final analysis and handover

Only the items marked done below exist in this repository.

## Setup (Windows PowerShell)

Requires Python 3.12 (see `docs/DECISIONS.md`, D-002).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

Only the core data stack (NumPy, Polars, PyArrow, PyYAML) is installed so far. Further dependencies are added when the stage that needs them begins. Always use `python -m pip ...` rather than bare `pip`.

Tell the project where the raw H&M CSVs are stored:

```powershell
Copy-Item configs\local.example.yaml configs\local.yaml
# edit configs\local.yaml and set raw_data_dir
python -m hm_forecast.data.inventory
```

Run all raw-data checks at once (inventory, articles profile, transactions profile; about 30 s on the full data):

```powershell
python -m hm_forecast.data.validate
```

Exit code 0 = passed, 1 = blocking data errors, 2 = a stage could not run. The combined summary is written to `artifacts/validation/validation_summary.json`.

## Current repository layout

```
configs/          data.yaml (expected files/columns), local.example.yaml; local.yaml is git-ignored
data/raw/         downloaded H&M CSVs (git-ignored)
data/processed/   generated Parquet tables (git-ignored)
data/embeddings/  cached article embeddings (git-ignored)
data/sample/      tiny committed samples for tests/demos (empty for now)
docs/             project documentation (added as needed)
notebooks/        exploratory analysis (empty for now)
reports/          figures and tables for the dissertation (empty for now)
src/hm_forecast/  Python package (empty apart from version for now)
tests/            automated tests (environment smoke tests only)
pyproject.toml    project metadata and bounded dependencies
```

## Progress

- [x] Stage 1, Step 1: repository skeleton, `.gitignore`, `.gitattributes`, README
- [x] Stage 1, Step 2: Python 3.12 environment, `pyproject.toml`, smoke tests, `docs/DECISIONS.md`
- [x] Stage 1 complete (Stage 1 closed after Step 2 was verified)
- [x] Stage 2, Step 1: raw-file inventory (`python -m hm_forecast.data.inventory`)
- [x] Stage 2, Step 2: articles profile (`python -m hm_forecast.data.profile_articles`)
- [x] Stage 2, Step 3: transactions profile (`python -m hm_forecast.data.profile_transactions`)
- [ ] Stage 2, Step 4: single validation gate (`python -m hm_forecast.data.validate`)
- [ ] Stage 3 onward: not started

## Integrity rules

- Chronological splits only; no random row-level splitting for the primary evaluation.
- The final test period is never used for model selection.
- No future counts, prices, or customer aggregates as prediction inputs.
- No fabricated results: every reported number must come from a recorded run.
