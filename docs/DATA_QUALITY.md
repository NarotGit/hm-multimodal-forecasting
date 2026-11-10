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
| Key is a zero-padded 10-digit string | all 105,542 values start with 0; length 10-10 | warning | Integer conversion would drop the zero. Keep as text. Resolved by D-006 (2026-11-10): transactions use the same 10-character format, so join by exact string equality |
| Code to name is one-to-one in one direction only | 0 codes with several names; names with several codes: department_name 22, product_type_name 1, section_name 1. Examples reviewed 2026-11-08: department_name "Shirt" (4 codes), "Trouser" (7), "Knitwear" (11), "Blazer" (2), "Jersey" (5); product_type_name "Umbrella" (codes 532, 83); section_name "Ladies Other" (codes 17, 4) | warning | Names are generic labels reused across different codes, so they are not identifiers. Encode categories from the numeric codes. The codes are the finer-grained identifier (each code maps to exactly one name) |
| `detail_desc` missing | 416 of 105,542 missing/blank (0.394%) | info | Needs an explicit missing-text rule at Stage 6 |
| `prod_name` missing | 0 | info | none |
| Text is shared across articles | `detail_desc`: 43,404 unique among 105,126 usable; `prod_name`: 45,875 unique | info | Embed unique texts and map back (Stage 6). Relevant to RQ3 |
| Text length | `detail_desc` median 132, p95 266, max 764 characters; `prod_name` median 15, max 30 | info | Check against the encoder's token limit at Stage 6 |
| Articles per product_code | 47,224 groups; min 1, median 1, max 75 | info | Variants share product codes; relevant to RQ3 |

## Stage 2, Step 3: transactions_train.csv profile (verified 2026-11-10)
Command: `python -m hm_forecast.data.profile_transactions` (full file, lazy streaming scans, all columns read as text)

| Finding | Evidence | Severity | Consequence |
|---|---|---|---|
| Size and period | 31,788,324 rows; 1,362,281 distinct customers; 2018-09-20 to 2020-09-22 | info | Matches the expected scale of the dataset |
| No missing calendar dates | 734 distinct dates = 734 calendar days; 0 missing | none | Every day has at least one transaction row (this says nothing about individual articles) |
| Daily volume varies a lot | rows/day: min 12,760; median 39,515.5; max 198,622 | info | Strong day-to-day variation, to be examined in Stage 3 (possible promotion/seasonal effects) |
| `article_id` format matches articles.csv | all 31,788,324 rows have length 10 and a leading zero; 0 rows unmatched by exact string comparison | none | Join on the exact 10-character text key; no normalisation needed (D-006). Never cast to integer |
| Every transaction key exists in articles | 0 unmatched rows | none | No orphan transactions |
| Articles with no transactions | 995 of 105,542 (0.94%) have none; 104,547 have at least one | info | Cannot be forecast from their own history; eligibility to be decided in Stage 3 |
| Price | min 1.69e-05; max 0.5915; mean 0.0278; 0 non-positive, 0 non-numeric | none | Values look normalised, not currency; interpret with care |
| Sales channel | 1: 9,408,462 (29.6%); 2: 22,379,862 (70.4%); no other values | none | Both channels present; channel split to be explored in Stage 3 |
| Repeated identical rows | about 2,974,905 rows (9.4%) repeat an earlier row (64-bit row hash, approximate) | info | Legitimate repeat purchase events, kept: the target is a count of rows |
| Trial run is not representative | first 2,000,000 rows touched only 35,103 articles, channel 1 share 31.6%, mean price 0.0298 | info | Trial runs describe the start of the file only; the early assortment looks smaller than the full catalogue (to examine in Stage 3) |

### Measured cost on the development machine (CPU only, no GPU)
| Scan | Time | Peak process memory |
|---|---|---|
| Global statistics (includes distinct customers and row hashes) | 17.31 s | 7,263 MB |
| Rows per date | 3.64 s | 1,341 MB |
| Rows per article | 1.83 s | 923 MB |

Peak memory is the process resident memory sampled during each scan, so later scans include memory still held from earlier ones. The global-statistics scan is the expensive one; a plain per-date or per-article aggregation is cheap, which is what Stage 4 needs.

## Not yet checked
- Whether `customers.csv` ids cover the transaction `customer_id`s (customers are optional and not used for now).
- Per-article activity, sparsity, article launch and withdrawal patterns (Stage 3).

## Open questions
1. Which articles are eligible for forecasting, and how are days without sales represented (Stage 3)?
2. Why does daily volume spike (max 198,622 vs median 39,515.5), and does it shift the distribution between train and test periods (Stage 3)?

Resolved: article_id format in transactions and key coverage (see Step 3 above, 2026-11-10); which names are shared across several codes (see articles table, 2026-11-08).
