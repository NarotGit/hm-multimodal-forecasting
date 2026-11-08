# Data Quality Log

Findings are recorded with the evidence that produced them. Everything below comes from runs on the real files at the time stated. Items marked **open** are not yet resolved.

## Stage 2, Step 1: raw-file inventory (verified 2026-11-05)
Command: `python -m hm_forecast.data.inventory`

| File | Size | Columns | Result |
|---|---|---|---|
| transactions_train.csv | 3,488.0 MB | 5 | OK |
| articles.csv | 36.1 MB | 25 | OK |
| customers.csv | 207.1 MB | 7 | OK (optional file; not used unless justified) |

Headers match `configs/data.yaml`. File sizes are close to the expected sizes. Not checked at this boundary: rows, dtypes, values.

## Stage 2, Step 2: articles.csv profile (verified 2026-11-08)
Command: `python -m hm_forecast.data.profile_articles` (all columns read as text)

| Finding | Evidence | Severity | Consequence |
|---|---|---|---|
| Key is clean | 105,542 rows; 105,542 unique `article_id`; 0 null, 0 non-integer, 0 duplicate | none | `article_id` is a valid unique key for articles |
| Key is a zero-padded 10-digit string | all 105,542 values start with 0; length 10-10 | warning | Integer conversion would drop the zero. Keep as text or pad explicitly. Join rule decided in D-006 **after** the transactions profile shows its `article_id` format (**open**) |
| Code to name is one-to-one in one direction only | 0 codes with several names; names with several codes: department_name 22, product_type_name 1, section_name 1 | warning | Encode categories from numeric codes, not names. Examples are in `artifacts/validation/articles_profile.json` (**open**: not yet reviewed) |
| `detail_desc` missing | 416 of 105,542 missing/blank (0.394%) | info | Needs an explicit missing-text rule at Stage 6 |
| `prod_name` missing | 0 | info | none |
| Text is shared across articles | `detail_desc`: 43,404 unique among 105,126 usable; `prod_name`: 45,875 unique | info | Embed unique texts and map back (Stage 6). Relevant to RQ3 |
| Text length | `detail_desc` median 132, p95 266, max 764 characters; `prod_name` median 15, max 30 | info | Check against the encoder's token limit at Stage 6 |
| Articles per product_code | 47,224 groups; min 1, median 1, max 75 | info | Variants share product codes; relevant to RQ3 |

## Open questions
1. How is `article_id` formatted in `transactions_train.csv`, and do all transaction keys exist in `articles.csv`? (next step)
2. Which names are shared across several codes (department_name especially)?
