"""Stage 3, Step 1: per-article daily purchase-count EDA.

Measures the structure of the target (purchase events per ``article_id`` per day)
so that the Stage 3 decisions (eligibility rule, missing article-day policy,
split dates) can be made from evidence. This module **only measures**. It does
not filter articles, does not zero-fill, and does not choose any threshold.

The transactions file is never loaded whole. It is scanned lazily with the Polars
streaming engine (every column read as text, as in the Stage 2 profile) and three
small aggregations are collected:

1. per article: events, distinct active days, first and last sale date,
2. per calendar date: events, distinct active articles, events per sales channel,
3. per article-day: histogram of the daily count among days that have a record.

Everything else is derived from those three small tables.

Terminology (kept strict on purpose)
------------------------------------
* "record" / "non-zero cell": an (article, date) pair with at least one row.
* "span": first sale date to last sale date of one article, inclusive.
* "record density": active days divided by span days. It is computed **inside
  each article's own span only**. A missing day inside the span is *not* shown to
  be a zero-demand day (there is no inventory information); it is simply a day
  without a record. Days outside the span are not counted at all.

Checks (exit code 1 if any fails)
---------------------------------
All three scans must agree on the total number of rows, channel counts must add
up, and no date may be unparseable. For a full-file run the totals are also
compared with the Stage 2 validation profile when that report exists.

Outputs, written to ``artifacts/eda/article_daily/`` (git-ignored):
``summary.json``, ``article_stats.parquet``, ``daily_stats.csv``,
``count_histogram.csv``, ``monthly_summary.csv``. The article table is Parquet,
not CSV, so the zero-padded ``article_id`` text survives round trips (Excel would
strip the leading zeros of a CSV).

Usage:  python -m hm_forecast.eda.article_daily --n-rows 2000000   (trial run)
        python -m hm_forecast.eda.article_daily                    (full file)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from hm_forecast.config import DEFAULT_DATA_CONFIG, REPO_ROOT, load_yaml, resolve_raw_data_dir
from hm_forecast.data.profile_transactions import DATE_FORMAT, benchmark, scan_transactions

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "eda" / "article_daily"
DEFAULT_OUTPUT_DIR_SAMPLE = REPO_ROOT / "artifacts" / "eda" / "article_daily_trial"
DEFAULT_REFERENCE_PROFILE = REPO_ROOT / "artifacts" / "validation" / "transactions_profile.json"

PASS, FAIL, SKIPPED = "pass", "fail", "skipped"

DEFAULTS = {
    "expected_channel_values": ["1", "2"],
    "spike_window_days": 29,  # centred rolling-median window used as the "normal" level of a day
    "top_spike_days": 15,
    "count_histogram_cap": 20,  # article-day counts above this are pooled in one bucket
    "quantiles": [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
    "min_threshold_values": [1, 2, 5, 10, 30, 100],
    "recent_windows_days": [7, 28],
}


def settings(config: dict | None) -> dict:
    """Defaults overridden by ``eda.article_daily`` in the data config, if present."""
    merged = dict(DEFAULTS)
    section = ((config or {}).get("eda") or {}).get("article_daily") or {}
    merged.update(section)
    if int(merged["spike_window_days"]) < 1 or int(merged["count_histogram_cap"]) < 1:
        raise ValueError("spike_window_days and count_histogram_cap must be at least 1")
    return merged


# --------------------------------------------------------------------------- scans


def _prepared(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Only the three columns needed, with the date parsed. Unparseable dates become null."""
    return lf.select(
        pl.col("article_id"),
        pl.col("t_dat").str.to_date(DATE_FORMAT, strict=False).alias("date"),
        pl.col("sales_channel_id").alias("channel"),
    )


def article_stats_scan(lf: pl.LazyFrame) -> pl.DataFrame:
    """Scan 1: one row per article that has at least one row with a valid date."""
    return (
        _prepared(lf)
        .filter(pl.col("date").is_not_null())
        .group_by("article_id")
        .agg(
            pl.len().alias("n_events"),
            pl.col("date").n_unique().alias("n_active_days"),
            pl.col("date").min().alias("first_date"),
            pl.col("date").max().alias("last_date"),
        )
        .sort("article_id")
        .collect(engine="streaming")
    )


def daily_stats_scan(lf: pl.LazyFrame, channels: list[str]) -> pl.DataFrame:
    """Scan 2: one row per date seen in the file. A null date holds unparseable rows."""
    aggs = [
        pl.len().alias("n_events"),
        pl.col("article_id").n_unique().alias("n_active_articles"),
    ]
    aggs += [(pl.col("channel") == v).sum().alias(f"channel_{v}") for v in channels]
    aggs.append((pl.col("channel").is_null() | ~pl.col("channel").is_in(channels)).sum().alias("channel_other"))
    return _prepared(lf).group_by("date").agg(aggs).sort("date").collect(engine="streaming")


def count_histogram_scan(lf: pl.LazyFrame) -> pl.DataFrame:
    """Scan 3: how many article-days (with a record) have exactly k events."""
    return (
        _prepared(lf)
        .filter(pl.col("date").is_not_null())
        .group_by("article_id", "date")
        .agg(pl.len().alias("n_events"))
        .group_by("n_events")
        .agg(pl.len().alias("n_article_days"))
        .sort("n_events")
        .collect(engine="streaming")
    )


# ----------------------------------------------------------------- derived tables


def with_span_columns(articles: pl.DataFrame) -> pl.DataFrame:
    """Add span length and record density (density is relative to the article's own span)."""
    return articles.with_columns(
        ((pl.col("last_date") - pl.col("first_date")).dt.total_days() + 1).alias("span_days")
    ).with_columns((pl.col("n_active_days") / pl.col("span_days")).alias("record_density"))


def complete_daily_table(daily_raw: pl.DataFrame, articles: pl.DataFrame, channels: list[str]) -> pl.DataFrame:
    """Daily table over the **full calendar range**, so days without any row are explicit.

    Adds, per day, ``n_in_span``: the number of articles whose first sale date is on
    or before the day and whose last sale date is on or after it (the articles that
    "exist" in the data that day), and ``active_share_of_in_span``.
    """
    valid = daily_raw.drop_nulls("date")
    calendar = pl.DataFrame({"date": pl.date_range(valid["date"].min(), valid["date"].max(), interval="1d", eager=True)})
    count_cols = ["n_events", "n_active_articles", *[f"channel_{v}" for v in channels], "channel_other"]
    starts = articles.group_by("first_date").agg(pl.len().alias("n_first_sales")).rename({"first_date": "date"})
    ends = articles.group_by("last_date").agg(pl.len().alias("n_last_sales")).rename({"last_date": "date"})
    out = (
        calendar.join(valid, on="date", how="left")
        .join(starts, on="date", how="left")
        .join(ends, on="date", how="left")
        .with_columns(pl.col(*count_cols, "n_first_sales", "n_last_sales").fill_null(0))
        .sort("date")
    )
    return out.with_columns(
        (pl.col("n_first_sales").cum_sum() - pl.col("n_last_sales").cum_sum().shift(1, fill_value=0)).alias("n_in_span"),
        pl.col("date").dt.weekday().alias("weekday"),  # 1 = Monday ... 7 = Sunday
    ).with_columns((pl.col("n_active_articles") / pl.col("n_in_span")).alias("active_share_of_in_span"))


def add_spike_columns(daily: pl.DataFrame, window: int) -> pl.DataFrame:
    """Compare each day's volume with the centred rolling median of its neighbours."""
    return daily.with_columns(
        pl.col("n_events").cast(pl.Float64).rolling_median(window_size=window, center=True, min_samples=1).alias("rolling_median_events")
    ).with_columns((pl.col("n_events") / pl.col("rolling_median_events")).alias("ratio_to_rolling_median"))


def monthly_table(daily: pl.DataFrame, articles: pl.DataFrame, channels: list[str]) -> pl.DataFrame:
    """Events, channel mix, and assortment entries/exits per calendar month."""
    month = pl.col("date").dt.truncate("1mo").alias("month")
    channel_cols = [f"channel_{v}" for v in channels]
    sums = [pl.col(c).sum() for c in ["n_events", *channel_cols, "channel_other"]]
    m = daily.group_by(month).agg(
        *sums,
        pl.col("n_active_articles").mean().alias("mean_daily_active_articles"),
        pl.len().alias("n_days"),
    )
    m = m.with_columns([(pl.col(c) / pl.col("n_events")).alias(f"{c}_share") for c in channel_cols])
    first = articles.group_by(pl.col("first_date").dt.truncate("1mo").alias("month")).agg(pl.len().alias("n_articles_first_sold"))
    last = articles.group_by(pl.col("last_date").dt.truncate("1mo").alias("month")).agg(pl.len().alias("n_articles_last_sold"))
    return (
        m.join(first, on="month", how="left")
        .join(last, on="month", how="left")
        .with_columns(pl.col("n_articles_first_sold", "n_articles_last_sold").fill_null(0))
        .sort("month")
    )


# ---------------------------------------------------------------------- summaries


def describe(series: pl.Series, quantiles: list[float]) -> dict:
    """Count, min, mean, max and the requested quantiles (labelled p1 ... p99)."""
    s = series.drop_nulls()
    if s.is_empty():
        return {"n": 0}
    out = {"n": s.len(), "min": s.min(), "mean": float(s.mean()), "max": s.max()}
    out.update({f"p{round(q * 100):d}": s.quantile(q, interpolation="linear") for q in quantiles})
    return out


def histogram_summary(hist: pl.DataFrame, cap: int) -> dict:
    """Distribution of the daily count among article-days that have a record."""
    total = int(hist["n_article_days"].sum())
    exact = hist.filter(pl.col("n_events") <= cap)
    pooled = hist.filter(pl.col("n_events") > cap)
    events = int((hist["n_events"].cast(pl.Int64) * hist["n_article_days"].cast(pl.Int64)).sum())
    share_1 = int(hist.filter(pl.col("n_events") == 1)["n_article_days"].sum()) / total if total else None
    return {
        "n_article_days_with_record": total,
        "n_events": events,
        "mean_events_per_article_day": events / total if total else None,
        "max_events_in_one_article_day": int(hist["n_events"].max()) if total else None,
        "share_article_days_with_exactly_1_event": share_1,
        "cap": cap,
        "exact": {str(k): int(v) for k, v in zip(exact["n_events"], exact["n_article_days"])},
        "pooled_above_cap": int(pooled["n_article_days"].sum()),
    }


def threshold_table(articles: pl.DataFrame, thresholds: list[int]) -> list[dict]:
    """How many articles have at least k events / at least k active days (no choice is made here)."""
    n = articles.height
    rows = []
    for k in thresholds:
        e = int((articles["n_events"] >= k).sum())
        d = int((articles["n_active_days"] >= k).sum())
        rows.append({"k": k, "n_articles_events_ge_k": e, "share_events_ge_k": e / n, "n_articles_active_days_ge_k": d, "share_active_days_ge_k": d / n})
    return rows


def lifecycle_summary(articles: pl.DataFrame, first: object, last: object, windows: list[int]) -> dict:
    """Left/right censoring and recent activity, relative to the observed date range."""
    out = {
        "first_sale_on_first_date_of_data": int((articles["first_date"] == first).sum()),
        "last_sale_on_last_date_of_data": int((articles["last_date"] == last).sum()),
    }
    for w in windows:
        # first_date < first + w days  <=>  first_date - w days < first
        out[f"first_sale_within_first_{w}_days"] = int((articles["first_date"].dt.offset_by(f"-{w}d") < first).sum())
        # last_date > last - w days  <=>  last_date + w days > last
        out[f"last_sale_within_last_{w}_days"] = int((articles["last_date"].dt.offset_by(f"{w}d") > last).sum())
    out["n_articles_single_active_day"] = int((articles["n_active_days"] == 1).sum())
    return out


def channel_summary(daily: pl.DataFrame, monthly: pl.DataFrame, channels: list[str]) -> dict:
    total = int(daily["n_events"].sum())
    overall = {v: int(daily[f"channel_{v}"].sum()) for v in channels}
    out = {
        "overall_events": {**overall, "other": int(daily["channel_other"].sum())},
        "overall_share": {v: (n / total if total else None) for v, n in overall.items()},
    }
    out["monthly_share_min_max"] = {
        v: {"min": float(monthly[f"channel_{v}_share"].min()), "max": float(monthly[f"channel_{v}_share"].max())} for v in channels
    }
    return out


def run_checks(
    n_rows_total: int,
    n_rows_bad_date: int,
    articles: pl.DataFrame,
    daily: pl.DataFrame,
    hist: pl.DataFrame,
    reference: dict | None,
    reference_note: str | None,
    partial: bool,
) -> list[dict]:
    """Internal-consistency checks plus an optional comparison with the Stage 2 profile."""
    by_article = int(articles["n_events"].sum())
    by_day = int(daily["n_events"].sum())
    by_hist = int((hist["n_events"].cast(pl.Int64) * hist["n_article_days"].cast(pl.Int64)).sum())
    channel_cols = [c for c in daily.columns if c.startswith("channel_")]
    by_channel = int(daily.select(pl.sum_horizontal(channel_cols).sum()).item())
    checks = [
        _check("no_unparseable_dates", n_rows_bad_date == 0, f"{n_rows_bad_date} rows with a missing or unparseable t_dat"),
        _check("article_total_equals_daily_total", by_article == by_day, f"by article {by_article:,}; by date {by_day:,}"),
        _check("histogram_total_equals_daily_total", by_hist == by_day, f"by article-day histogram {by_hist:,}; by date {by_day:,}"),
        _check("channel_counts_add_up", by_channel == by_day, f"channels {by_channel:,}; events {by_day:,}"),
        _check("rows_total_equals_valid_plus_bad", n_rows_total == by_day + n_rows_bad_date, f"all rows {n_rows_total:,}; valid {by_day:,}; bad date {n_rows_bad_date:,}"),
    ]
    if partial:
        checks.append(_check("reference_profile", None, "trial run (--n-rows): totals not compared with the full-file profile"))
    elif reference is None:
        checks.append(_check("reference_profile", None, reference_note or "no Stage 2 transactions profile found"))
    else:
        n_days = daily.filter(pl.col("n_events") > 0).height
        pairs = {
            "n_rows": (n_rows_total, reference["n_rows"]),
            "n_distinct_dates": (n_days, reference["n_distinct_dates"]),
            "n_calendar_days": (daily.height, reference["n_calendar_days"]),
            "n_articles_with_sales": (articles.height, reference["n_articles_with_sales"]),
        }
        for name, (ours, theirs) in pairs.items():
            checks.append(_check(f"matches_stage2_profile_{name}", ours == theirs, f"EDA {ours:,}; Stage 2 profile {theirs:,}"))
    return checks


def _check(name: str, ok: bool | None, detail: str) -> dict:
    return {"name": name, "status": SKIPPED if ok is None else (PASS if ok else FAIL), "detail": detail}


def load_reference_profile(path: Path | None) -> tuple[dict | None, str | None]:
    """Totals from the Stage 2 full-file profile, or ``(None, reason)`` if unusable."""
    if path is None:
        return None, "reference comparison disabled (--no-reference)"
    if not Path(path).is_file():
        return None, f"no Stage 2 profile at {path}"
    try:
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        if p.get("partial_scan_rows") is not None:
            return None, f"{path} is a trial-run profile, not a full-file profile"
        return {
            "n_rows": int(p["global"]["n_rows"]),
            "n_distinct_dates": int(p["dates"]["n_distinct_dates"]),
            "n_calendar_days": int(p["dates"]["n_calendar_days"]),
            "n_articles_with_sales": int(p["article_key_coverage"]["exact_match"]["n_distinct_transaction_ids"]),
        }, None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, f"could not read {path}: {type(exc).__name__}: {exc}"


# ------------------------------------------------------------------------ the run


@dataclass
class ArticleDailyEda:
    summary: dict
    article_stats: pl.DataFrame
    daily_stats: pl.DataFrame
    count_histogram: pl.DataFrame
    monthly_summary: pl.DataFrame


def build_eda(
    lf: pl.LazyFrame,
    cfg: dict,
    reference: dict | None = None,
    reference_note: str | None = None,
    partial_rows: int | None = None,
    benchmarks: dict | None = None,
) -> ArticleDailyEda:
    """Run the three scans, derive the tables and the summary, and run the checks."""
    bench = benchmarks if benchmarks is not None else {}
    channels = [str(v) for v in cfg["expected_channel_values"]]
    qs = [float(q) for q in cfg["quantiles"]]

    with benchmark(bench, "scan_1_article_stats"):
        articles = article_stats_scan(lf)
    with benchmark(bench, "scan_2_daily_stats"):
        daily_raw = daily_stats_scan(lf, channels)
    with benchmark(bench, "scan_3_article_day_count_histogram"):
        hist = count_histogram_scan(lf)

    bad = daily_raw.filter(pl.col("date").is_null())
    n_bad = int(bad["n_events"].sum()) if bad.height else 0
    n_total = int(daily_raw["n_events"].sum())
    if daily_raw.drop_nulls("date").is_empty():
        raise ValueError("no rows with a valid t_dat: nothing to analyse")

    articles = with_span_columns(articles)
    daily = add_spike_columns(complete_daily_table(daily_raw, articles, channels), int(cfg["spike_window_days"]))
    monthly = monthly_table(daily, articles, channels)

    first, last = daily["date"].min(), daily["date"].max()
    n_articles, n_days = articles.height, daily.height
    nonzero = int(hist["n_article_days"].sum())
    within_span = int(articles["span_days"].sum())
    full_grid = n_articles * n_days
    top = (
        daily.sort("n_events", descending=True)
        .head(int(cfg["top_spike_days"]))
        .select("date", "weekday", "n_events", "rolling_median_events", "ratio_to_rolling_median")
    )

    checks = run_checks(n_total, n_bad, articles, daily, hist, reference, reference_note, partial_rows is not None)
    summary = {
        "scope": {
            "n_rows_total": n_total,
            "n_rows_unparseable_date": n_bad,
            "n_articles_with_sales": n_articles,
            "n_calendar_days": n_days,
            "n_days_without_any_row": int((daily["n_events"] == 0).sum()),
            "first_date": str(first),
            "last_date": str(last),
            "partial_scan_rows": partial_rows,
        },
        "grid_sparsity": {
            "note": "within-span cells exclude days before an article's first sale and after its last sale; a missing day inside the span is a day without a record, not proven zero demand",
            "n_article_days_with_record": nonzero,
            "n_cells_within_article_spans": within_span,
            "n_cells_full_grid": full_grid,
            "record_share_of_within_span_cells": nonzero / within_span,
            "record_share_of_full_grid": nonzero / full_grid,
            "within_span_cells_without_record": within_span - nonzero,
        },
        "per_article": {
            "n_events": describe(articles["n_events"], qs),
            "n_active_days": describe(articles["n_active_days"], qs),
            "span_days": describe(articles["span_days"], qs),
            "record_density_within_span": describe(articles["record_density"], qs),
        },
        "min_history_thresholds": threshold_table(articles, [int(k) for k in cfg["min_threshold_values"]]),
        "article_day_count_distribution": histogram_summary(hist, int(cfg["count_histogram_cap"])),
        "lifecycle": lifecycle_summary(articles, first, last, [int(w) for w in cfg["recent_windows_days"]]),
        "daily_volume": {
            "events_per_day": describe(daily["n_events"], qs),
            "spike_window_days": int(cfg["spike_window_days"]),
            "n_days_ratio_ge_2": int((daily["ratio_to_rolling_median"] >= 2).sum()),
            "top_days": [
                {
                    "date": str(r["date"]),
                    "weekday": r["weekday"],
                    "n_events": r["n_events"],
                    "rolling_median_events": r["rolling_median_events"],
                    "ratio_to_rolling_median": r["ratio_to_rolling_median"],
                }
                for r in top.iter_rows(named=True)
            ],
        },
        "assortment": {
            "active_articles_per_day": describe(daily["n_active_articles"], qs),
            "articles_in_span_per_day": describe(daily["n_in_span"], qs),
            "active_share_of_in_span_per_day": describe(daily["active_share_of_in_span"], qs),
            "active_articles_first_date": int(daily["n_active_articles"][0]),
            "active_articles_last_date": int(daily["n_active_articles"][-1]),
        },
        "channel": channel_summary(daily, monthly, channels),
        "checks": checks,
        "checks_failed": [c["name"] for c in checks if c["status"] == FAIL],
        "benchmarks": bench,
    }
    return ArticleDailyEda(summary, articles, daily, hist, monthly)


def has_failed_checks(result: ArticleDailyEda) -> bool:
    return bool(result.summary["checks_failed"])


def write_outputs(result: ArticleDailyEda, out_dir: Path, source: Path | None = None) -> list[Path]:
    """Write the summary and the small tables. Returns the paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "article_stats": out_dir / "article_stats.parquet",
        "daily": out_dir / "daily_stats.csv",
        "histogram": out_dir / "count_histogram.csv",
        "monthly": out_dir / "monthly_summary.csv",
        "summary": out_dir / "summary.json",
    }
    result.article_stats.write_parquet(paths["article_stats"])
    result.daily_stats.write_csv(paths["daily"])
    result.count_histogram.write_csv(paths["histogram"])
    result.monthly_summary.write_csv(paths["monthly"])
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": None if source is None else str(source),
        "checks_failed": result.summary["checks_failed"],
        **result.summary,
    }
    paths["summary"].write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return list(paths.values())


def format_summary(result: ArticleDailyEda) -> str:
    s = result.summary
    sc, g, pa, h, lc, dv = s["scope"], s["grid_sparsity"], s["per_article"], s["article_day_count_distribution"], s["lifecycle"], s["daily_volume"]
    a = s["assortment"]
    lines = [
        f"Rows: {sc['n_rows_total']:,}   Articles with sales: {sc['n_articles_with_sales']:,}   "
        f"Dates: {sc['first_date']} to {sc['last_date']} ({sc['n_calendar_days']} calendar days, {sc['n_days_without_any_row']} without any row)",
        f"Article-days with a record: {g['n_article_days_with_record']:,}  |  cells inside article spans: {g['n_cells_within_article_spans']:,}  |  full grid: {g['n_cells_full_grid']:,}",
        f"Record share: {g['record_share_of_within_span_cells']:.1%} of within-span cells, {g['record_share_of_full_grid']:.1%} of the full grid",
        f"Events per article: median {pa['n_events']['p50']:,.0f}, p10 {pa['n_events']['p10']:,.0f}, p90 {pa['n_events']['p90']:,.0f}, max {pa['n_events']['max']:,}",
        f"Active days per article: median {pa['n_active_days']['p50']:,.0f}; record density within span: median {pa['record_density_within_span']['p50']:.2f}, p10 {pa['record_density_within_span']['p10']:.2f}",
        f"Events per article-day with a record: mean {h['mean_events_per_article_day']:.2f}; exactly 1 event in {h['share_article_days_with_exactly_1_event']:.1%} of them; max {h['max_events_in_one_article_day']:,}",
        f"Lifecycle: first sale on first data date: {lc['first_sale_on_first_date_of_data']:,}; last sale on last data date: {lc['last_sale_on_last_date_of_data']:,}; single active day: {lc['n_articles_single_active_day']:,}",
        f"Active articles per day: first date {a['active_articles_first_date']:,}, last date {a['active_articles_last_date']:,}, median {a['active_articles_per_day']['p50']:,.0f}",
        f"Daily volume: median {dv['events_per_day']['p50']:,.0f}, max {dv['events_per_day']['max']:,}; days >= 2x rolling median: {dv['n_days_ratio_ge_2']}",
        "Top volume days: " + "; ".join(f"{d['date']} ({d['n_events']:,}, x{d['ratio_to_rolling_median']:.1f})" for d in dv["top_days"][:5]),
        "Channel share overall: " + ", ".join(f"{k}={v:.1%}" for k, v in s["channel"]["overall_share"].items()),
        "Benchmarks: " + ", ".join(f"{k}: {v['seconds']}s peak {v['peak_memory_mb']} MB" for k, v in s["benchmarks"].items()),
        "Checks:",
    ]
    lines += [f"  {c['status'].upper():7} {c['name']}: {c['detail']}" for c in s["checks"]]
    lines.append("Result: " + ("CHECK FAILURES" if s["checks_failed"] else "all checks passed"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-article daily purchase-count EDA (Stage 3, Step 1).")
    parser.add_argument("--raw-dir", type=Path, help="Override raw_data_dir from configs/local.yaml.")
    parser.add_argument("--config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None, help="Output folder (default depends on --n-rows).")
    parser.add_argument("--n-rows", type=int, default=None, help="Trial run: scan only the first N rows.")
    parser.add_argument("--reference-profile", type=Path, default=DEFAULT_REFERENCE_PROFILE, help="Stage 2 transactions profile to compare totals with.")
    parser.add_argument("--no-reference", action="store_true", help="Skip the comparison with the Stage 2 profile.")
    args = parser.parse_args(argv)
    if args.n_rows is not None and args.n_rows <= 0:
        parser.error("--n-rows must be a positive integer")
    out_dir = args.output_dir or (DEFAULT_OUTPUT_DIR_SAMPLE if args.n_rows else DEFAULT_OUTPUT_DIR)

    try:
        raw_dir = args.raw_dir if args.raw_dir else resolve_raw_data_dir()
        tx_path = raw_dir / "transactions_train.csv"
        if not tx_path.is_file():
            raise FileNotFoundError(f"{tx_path.name} not found in {raw_dir}")
        cfg = settings(load_yaml(args.config))
        reference, note = load_reference_profile(None if args.no_reference else args.reference_profile)
        result = build_eda(scan_transactions(tx_path, args.n_rows), cfg, reference, note, partial_rows=args.n_rows)
    except (FileNotFoundError, ValueError, KeyError, OSError, pl.exceptions.PolarsError) as exc:
        print(f"Could not run the article-daily EDA: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(f"Source: {tx_path}")
    print(format_summary(result))
    written = write_outputs(result, out_dir, tx_path)
    print("Written: " + ", ".join(str(p) for p in written))
    return 1 if has_failed_checks(result) else 0


if __name__ == "__main__":
    sys.exit(main())
