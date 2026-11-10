"""Profile of ``transactions_train.csv`` (about 3.5 GB, ~31.8 M rows expected).

Third data-quality boundary. The file is **never loaded into memory as a whole**:
it is scanned lazily with the Polars streaming engine, every column is read as
text (no type inference), and each question is answered by a small aggregation.
Three scans are made:

1. global statistics (nulls, formats, dates, prices, channels, row hashes),
2. rows per calendar date,
3. rows per ``article_id`` (to measure key coverage against ``articles.csv``).

Policy
------
ERROR   (blocks later stages): no rows, missing/non-integer article_id, missing or
        unparseable t_dat, missing or non-numeric price, any transaction whose
        article_id has no match in articles.csv (after the documented
        zero-padding rule).
WARNING: non-positive prices, unexpected sales_channel_id values, calendar dates
        inside the observed range that have no rows.
INFO:   article_id format differences between the two files, articles that never
        appear in transactions, approximate duplicate-row count.

Exact duplicate rows are **not** defects: one customer can buy the same article
twice on the same day, and each row is a purchase event for our target. The
duplicate count uses a 64-bit row hash, so it is approximate.

Usage:  python -m hm_forecast.data.profile_transactions --n-rows 2000000   (trial run)
        python -m hm_forecast.data.profile_transactions                      (full file)
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from hm_forecast.config import DEFAULT_DATA_CONFIG, REPO_ROOT, load_yaml, resolve_raw_data_dir
from hm_forecast.data.inventory import ERROR, WARNING, Issue
from hm_forecast.data.profile_articles import INFO, INTEGER_PATTERN

DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "validation" / "transactions_profile.json"
DEFAULT_OUTPUT_SAMPLE = REPO_ROOT / "artifacts" / "validation" / "transactions_profile_trial.json"
DATE_FORMAT = "%Y-%m-%d"
MAX_EXAMPLES = 5
COLUMNS = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]


class PeakMemory:
    """Sample this process's resident memory in a background thread (needs psutil).

    ``peak_mb`` stays ``None`` when psutil is not installed.
    """

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak_mb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self, process) -> None:
        while not self._stop.is_set():
            rss = process.memory_info().rss / 1e6
            self.peak_mb = rss if self.peak_mb is None else max(self.peak_mb, rss)
            self._stop.wait(self.interval)

    def __enter__(self):
        try:
            import psutil
        except ImportError:
            return self
        self._thread = threading.Thread(target=self._run, args=(psutil.Process(),), daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()
        return False


@contextmanager
def benchmark(store: dict, label: str):
    """Record wall-clock seconds and peak memory (MB) of a block into ``store[label]``."""
    start = time.perf_counter()
    with PeakMemory() as mem:
        yield
    store[label] = {
        "seconds": round(time.perf_counter() - start, 2),
        "peak_memory_mb": None if mem.peak_mb is None else round(mem.peak_mb, 1),
    }


def scan_transactions(path: Path, n_rows: int | None = None) -> pl.LazyFrame:
    """Lazy scan with every column as text. Nothing is read until a query is collected.

    ``n_rows`` limits the scan to the first rows of the file (used for trial runs).
    """
    return pl.scan_csv(path, infer_schema=False, n_rows=n_rows)


def _blank(col: str) -> pl.Expr:
    return (pl.col(col).str.strip_chars() == "").sum()


def global_stats(lf: pl.LazyFrame, expected_channels: list[str]) -> dict:
    """Scan 1: one pass computing all file-level statistics."""
    date = pl.col("t_dat").str.to_date(DATE_FORMAT, strict=False)
    price = pl.col("price").cast(pl.Float64, strict=False)
    article = pl.col("article_id")
    channel = pl.col("sales_channel_id")

    exprs = [
        pl.len().alias("n_rows"),
        # t_dat
        pl.col("t_dat").null_count().alias("t_dat_null"),
        _blank("t_dat").alias("t_dat_blank"),
        (pl.col("t_dat").is_not_null() & date.is_null()).sum().alias("t_dat_unparseable"),
        date.min().alias("t_dat_min"),
        date.max().alias("t_dat_max"),
        # customer_id
        pl.col("customer_id").null_count().alias("customer_null"),
        _blank("customer_id").alias("customer_blank"),
        pl.col("customer_id").str.len_chars().min().alias("customer_len_min"),
        pl.col("customer_id").str.len_chars().max().alias("customer_len_max"),
        pl.col("customer_id").n_unique().alias("customer_n_unique"),
        # article_id
        article.null_count().alias("article_null"),
        _blank("article_id").alias("article_blank"),
        (article.is_not_null() & ~article.str.contains(INTEGER_PATTERN)).sum().alias("article_non_integer"),
        article.str.len_chars().min().alias("article_len_min"),
        article.str.len_chars().max().alias("article_len_max"),
        article.str.contains(r"^0\d").sum().alias("article_leading_zero"),
        # price
        pl.col("price").null_count().alias("price_null"),
        (pl.col("price").is_not_null() & price.is_null()).sum().alias("price_non_numeric"),
        (price <= 0).sum().alias("price_non_positive"),
        price.min().alias("price_min"),
        price.max().alias("price_max"),
        price.mean().alias("price_mean"),
        # sales channel
        channel.null_count().alias("channel_null"),
        (channel.is_not_null() & ~channel.is_in(expected_channels)).sum().alias("channel_unexpected"),
        # approximate duplicate rows (64-bit hash of the whole row)
        pl.struct(COLUMNS).hash().n_unique().alias("n_unique_row_hashes"),
    ]
    exprs += [(channel == v).sum().alias(f"channel_{v}") for v in expected_channels]
    return lf.select(exprs).collect(engine="streaming").row(0, named=True)


def daily_counts(lf: pl.LazyFrame) -> pl.DataFrame:
    """Scan 2: number of rows per parsed date (a null date means unparseable/missing)."""
    return (
        lf.select(pl.col("t_dat").str.to_date(DATE_FORMAT, strict=False).alias("date"))
        .group_by("date")
        .agg(pl.len().alias("n_rows"))
        .sort("date")
        .collect(engine="streaming")
    )


def date_coverage(daily: pl.DataFrame, max_listed: int = 20) -> dict:
    """Which calendar days between the first and last date have no rows at all."""
    valid = daily.drop_nulls("date")
    if valid.is_empty():
        return {"n_distinct_dates": 0}
    first, last = valid["date"].min(), valid["date"].max()
    calendar = pl.date_range(first, last, interval="1d", eager=True)
    missing = calendar.filter(~calendar.is_in(valid["date"].implode()))
    rows = valid["n_rows"]
    return {
        "first_date": str(first),
        "last_date": str(last),
        "n_calendar_days": calendar.len(),
        "n_distinct_dates": valid.height,
        "n_missing_dates": missing.len(),
        "missing_dates_listed": [str(d) for d in missing.head(max_listed)],
        "rows_per_day_min": int(rows.min()),
        "rows_per_day_median": float(rows.median()),
        "rows_per_day_max": int(rows.max()),
    }


def article_counts(lf: pl.LazyFrame) -> pl.DataFrame:
    """Scan 3: rows per article_id string."""
    return lf.group_by("article_id").agg(pl.len().alias("n_rows")).collect(engine="streaming")


def _coverage(counts: pl.DataFrame, key_column: str, article_ids: pl.Series) -> dict:
    matched = counts[key_column].is_in(article_ids.implode())
    unmatched = counts.filter(~matched)
    return {
        "n_distinct_transaction_ids": counts.height,
        "n_distinct_matched": int(matched.sum()),
        "n_distinct_unmatched": unmatched.height,
        "n_rows_unmatched": int(unmatched["n_rows"].sum() or 0),
        "unmatched_examples": unmatched.head(MAX_EXAMPLES)[key_column].to_list(),
    }


def key_coverage(counts: pl.DataFrame, article_ids: pl.Series, width: int) -> dict:
    """Compare transaction article_ids with articles.csv ids, as-is and zero-padded."""
    padded = counts.with_columns(pl.col("article_id").str.zfill(width).alias("article_id_padded"))
    exact = _coverage(padded, "article_id", article_ids)
    zero_padded = _coverage(padded, "article_id_padded", article_ids)
    sold = padded.filter(padded["article_id_padded"].is_in(article_ids.implode()))["article_id_padded"]
    return {
        "articles_n_ids": article_ids.len(),
        "exact_match": exact,
        f"zero_padded_to_{width}": zero_padded,
        "n_articles_without_transactions": int((~article_ids.is_in(sold.implode())).sum()),
    }


def build_profile(
    lf: pl.LazyFrame,
    article_ids: pl.Series,
    cfg: dict,
    benchmarks: dict | None = None,
    partial_rows: int | None = None,
) -> dict:
    """Run the three scans and classify findings (policy in the module docstring)."""
    bench = benchmarks if benchmarks is not None else {}
    channels = list(cfg.get("expected_channel_values", ["1", "2"]))
    width = int(cfg.get("article_id_width", 10))

    with benchmark(bench, "scan_1_global_stats"):
        g = global_stats(lf, channels)
    with benchmark(bench, "scan_2_daily_counts"):
        daily = daily_counts(lf)
    with benchmark(bench, "scan_3_article_counts"):
        counts = article_counts(lf)

    dates = date_coverage(daily, int(cfg.get("max_missing_dates_listed", 20)))
    coverage = key_coverage(counts, article_ids, width)
    padded = coverage[f"zero_padded_to_{width}"]
    issues: list[Issue] = []

    if g["n_rows"] == 0:
        issues.append(Issue(ERROR, "NO_ROWS", "the file contains no data rows"))
    if g["article_null"] or g["article_blank"]:
        issues.append(Issue(ERROR, "ARTICLE_ID_MISSING", f"{g['article_null']} null and {g['article_blank']} blank article_id values"))
    if g["article_non_integer"]:
        issues.append(Issue(ERROR, "ARTICLE_ID_NON_INTEGER", f"{g['article_non_integer']} article_id values are not integer-like"))
    if g["t_dat_null"] or g["t_dat_blank"] or g["t_dat_unparseable"]:
        issues.append(Issue(ERROR, "DATE_INVALID", f"t_dat: {g['t_dat_null']} null, {g['t_dat_blank']} blank, {g['t_dat_unparseable']} not matching {DATE_FORMAT}"))
    if g["price_null"] or g["price_non_numeric"]:
        issues.append(Issue(ERROR, "PRICE_INVALID", f"price: {g['price_null']} null, {g['price_non_numeric']} non-numeric"))
    if padded["n_rows_unmatched"]:
        issues.append(
            Issue(
                ERROR,
                "ARTICLE_KEY_UNMATCHED",
                f"{padded['n_rows_unmatched']} rows ({padded['n_distinct_unmatched']} distinct ids) have no match in articles.csv "
                f"even after zero-padding to {width}; examples: {padded['unmatched_examples']}",
            )
        )
    if g["price_non_positive"]:
        issues.append(Issue(WARNING, "PRICE_NON_POSITIVE", f"{g['price_non_positive']} rows have price <= 0"))
    if g["channel_unexpected"] or g["channel_null"]:
        issues.append(Issue(WARNING, "CHANNEL_UNEXPECTED", f"{g['channel_unexpected']} unexpected and {g['channel_null']} null sales_channel_id values (expected {channels})"))
    if dates.get("n_missing_dates"):
        issues.append(Issue(WARNING, "DATE_GAPS", f"{dates['n_missing_dates']} calendar days between {dates['first_date']} and {dates['last_date']} have no rows; first: {dates['missing_dates_listed'][:5]}"))

    exact_rows = coverage["exact_match"]["n_rows_unmatched"]
    if exact_rows and not padded["n_rows_unmatched"]:
        issues.append(Issue(INFO, "ARTICLE_ID_FORMAT_DIFFERS", f"as written, {exact_rows} rows do not match articles.csv; all match after zero-padding to {width} (transactions length {g['article_len_min']}-{g['article_len_max']})"))
    if coverage["n_articles_without_transactions"]:
        issues.append(Issue(INFO, "ARTICLES_NEVER_SOLD", f"{coverage['n_articles_without_transactions']} articles in articles.csv have no transaction"))
    approx_dups = g["n_rows"] - g["n_unique_row_hashes"]
    if approx_dups:
        issues.append(Issue(INFO, "DUPLICATE_ROWS_APPROX", f"about {approx_dups} rows repeat an earlier row (64-bit hash); these are legitimate repeat purchase events, not defects"))

    if partial_rows is not None:
        issues.insert(
            0,
            Issue(INFO, "PARTIAL_SCAN", f"only the first {partial_rows:,} rows were scanned: a trial run, NOT a validation of the full file"),
        )

    g = {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in g.items()}
    return {
        "global": g,
        "approx_duplicate_rows": approx_dups,
        "dates": dates,
        "article_key_coverage": coverage,
        "benchmarks": bench,
        "partial_scan_rows": partial_rows,
        "issues": [vars(i) for i in issues],
    }


def has_blocking_errors(profile: dict) -> bool:
    return any(i["severity"] == ERROR for i in profile["issues"])


def format_summary(profile: dict) -> str:
    g, d, k = profile["global"], profile["dates"], profile["article_key_coverage"]
    lines = [
        f"Rows: {g['n_rows']:,}   Distinct customers: {g['customer_n_unique']:,}",
        f"Dates: {d.get('first_date')} to {d.get('last_date')}  distinct={d.get('n_distinct_dates')} "
        f"calendar={d.get('n_calendar_days')} missing={d.get('n_missing_dates')}  "
        f"rows/day min/median/max={d.get('rows_per_day_min')}/{d.get('rows_per_day_median')}/{d.get('rows_per_day_max')}",
        f"article_id (as written): length {g['article_len_min']}-{g['article_len_max']}, leading zeros in {g['article_leading_zero']:,} rows",
        f"Coverage vs articles.csv: exact match unmatched rows={k['exact_match']['n_rows_unmatched']:,}; "
        + "; ".join(f"{name} unmatched rows={v['n_rows_unmatched']:,}" for name, v in k.items() if name.startswith("zero_padded")),
        f"Articles with at least one transaction: {k['articles_n_ids'] - k['n_articles_without_transactions']:,} of {k['articles_n_ids']:,}",
        f"price: min={g['price_min']} max={g['price_max']} mean={g['price_mean']:.6f} non_positive={g['price_non_positive']}",
        "sales_channel_id: " + ", ".join(f"{key.split('_', 1)[1]}={val:,}" for key, val in g.items() if key.startswith("channel_") and key[8:].isdigit()),
        "Benchmarks: " + ", ".join(f"{k_}: {v['seconds']}s peak {v['peak_memory_mb']} MB" for k_, v in profile["benchmarks"].items()),
        "Issues:",
    ]
    lines += [f"  {i['severity'].upper():7} {i['code']}: {i['message']}" for i in profile["issues"]] or ["  none"]
    lines.append("Result: " + ("BLOCKING ERRORS FOUND" if has_blocking_errors(profile) else "no blocking errors"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile transactions_train.csv with lazy streaming scans.")
    parser.add_argument("--raw-dir", type=Path, help="Override raw_data_dir from configs/local.yaml.")
    parser.add_argument("--config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--output", type=Path, default=None, help="Report path (default depends on --n-rows).")
    parser.add_argument(
        "--n-rows",
        type=int,
        default=None,
        help="Trial run: scan only the first N rows (also measures time and memory). Omit for the full file.",
    )
    args = parser.parse_args(argv)
    if args.n_rows is not None and args.n_rows <= 0:
        parser.error("--n-rows must be a positive integer")
    output = args.output or (DEFAULT_OUTPUT_SAMPLE if args.n_rows else DEFAULT_OUTPUT)

    try:
        raw_dir = args.raw_dir if args.raw_dir else resolve_raw_data_dir()
        config = load_yaml(args.config)
        cfg = config["profiling"]["transactions"]
        tx_path, art_path = raw_dir / "transactions_train.csv", raw_dir / "articles.csv"
        for p in (tx_path, art_path):
            if not p.is_file():
                raise FileNotFoundError(f"{p.name} not found in {raw_dir}")
        article_ids = pl.read_csv(art_path, infer_schema=False, columns=["article_id"])["article_id"].drop_nulls()
        profile = build_profile(scan_transactions(tx_path, args.n_rows), article_ids, cfg, partial_rows=args.n_rows)
    except (FileNotFoundError, ValueError, KeyError, OSError, pl.exceptions.PolarsError) as exc:
        print(f"Could not profile transactions_train.csv: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(f"Source: {tx_path}")
    print(format_summary(profile))
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": str(tx_path),
        "blocking_errors": has_blocking_errors(profile),
        **profile,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report written to: {output}")
    return 1 if has_blocking_errors(profile) else 0


if __name__ == "__main__":
    sys.exit(main())
