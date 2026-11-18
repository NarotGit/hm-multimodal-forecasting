"""Tests for the per-article daily EDA (Stage 3, Step 1). Synthetic data only.

The main fixture is small enough to verify by hand (10 calendar days, 3 articles):

    article    rows per day with a record                    events  active  span
    ...101     Jan 1 x2, Jan 2 x1, Jan 4 x3                     6       3      4 (Jan 1-4)
    ...102     Jan 3 x1, Jan 7 x1, Jan 10 x2                    4       3      8 (Jan 3-10)
    ...103     Jan 5 x1                                         1       1      1 (Jan 5)

Days without any row: Jan 6, Jan 8, Jan 9.
"""

import csv
import json
from datetime import date

import polars as pl
import pytest

from hm_forecast.data import profile_transactions as pt
from hm_forecast.eda import article_daily as ad

HEADER = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
A, B, C = "0000000101", "0000000102", "0000000103"

ROWS = [
    ("2020-01-01", "c1", A, "0.1", "1"),
    ("2020-01-01", "c2", A, "0.1", "2"),
    ("2020-01-02", "c1", A, "0.1", "1"),
    ("2020-01-04", "c1", A, "0.1", "1"),
    ("2020-01-04", "c2", A, "0.1", "1"),
    ("2020-01-04", "c3", A, "0.1", "1"),
    ("2020-01-03", "c3", B, "0.1", "1"),
    ("2020-01-07", "c3", B, "0.1", "2"),
    ("2020-01-10", "c3", B, "0.1", "2"),
    ("2020-01-10", "c4", B, "0.1", "3"),  # unexpected channel
    ("2020-01-05", "c5", C, "0.1", "1"),
]

CFG = {**ad.DEFAULTS, "spike_window_days": 3, "count_histogram_cap": 20, "recent_windows_days": [2], "min_threshold_values": [1, 3, 5]}


def write_tx(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    return path


def eda(tmp_path, rows=ROWS, cfg=CFG, **kwargs):
    path = write_tx(tmp_path / "tx.csv", rows)
    return ad.build_eda(pt.scan_transactions(path), cfg, **kwargs)


def status(result, name):
    return next(c["status"] for c in result.summary["checks"] if c["name"] == name)


# ------------------------------------------------------------------ per article


def test_article_stats_hand_computed(tmp_path):
    r = eda(tmp_path)
    rows = {x["article_id"]: x for x in r.article_stats.iter_rows(named=True)}
    assert set(rows) == {A, B, C}
    assert (rows[A]["n_events"], rows[A]["n_active_days"], rows[A]["span_days"]) == (6, 3, 4)
    assert (rows[B]["n_events"], rows[B]["n_active_days"], rows[B]["span_days"]) == (4, 3, 8)
    assert (rows[C]["n_events"], rows[C]["n_active_days"], rows[C]["span_days"]) == (1, 1, 1)
    assert rows[A]["first_date"] == date(2020, 1, 1) and rows[A]["last_date"] == date(2020, 1, 4)
    assert rows[A]["record_density"] == pytest.approx(0.75)
    assert rows[B]["record_density"] == pytest.approx(3 / 8)
    assert rows[C]["record_density"] == pytest.approx(1.0)


def test_article_id_stays_text_with_leading_zeros_in_parquet(tmp_path):
    r = eda(tmp_path)
    ad.write_outputs(r, tmp_path / "out")
    back = pl.read_parquet(tmp_path / "out" / "article_stats.parquet")
    assert back.schema["article_id"] == pl.String
    assert sorted(back["article_id"]) == [A, B, C]


def test_no_zero_filled_article_days(tmp_path):
    """Only articles with sales appear, and no row is created for an article-day without a record."""
    r = eda(tmp_path)
    assert r.article_stats.height == 3
    assert r.summary["grid_sparsity"]["n_article_days_with_record"] == 7  # not 3 articles x 10 days


# ----------------------------------------------------------------------- sparsity


def test_grid_sparsity_counts(tmp_path):
    g = eda(tmp_path).summary["grid_sparsity"]
    assert g["n_article_days_with_record"] == 7
    assert g["n_cells_within_article_spans"] == 4 + 8 + 1
    assert g["n_cells_full_grid"] == 3 * 10
    assert g["within_span_cells_without_record"] == 6
    assert g["record_share_of_within_span_cells"] == pytest.approx(7 / 13)
    assert g["record_share_of_full_grid"] == pytest.approx(7 / 30)


def test_count_histogram_exact(tmp_path):
    r = eda(tmp_path)
    assert dict(zip(r.count_histogram["n_events"], r.count_histogram["n_article_days"])) == {1: 4, 2: 2, 3: 1}
    h = r.summary["article_day_count_distribution"]
    assert h["n_events"] == 11 and h["n_article_days_with_record"] == 7
    assert h["max_events_in_one_article_day"] == 3
    assert h["share_article_days_with_exactly_1_event"] == pytest.approx(4 / 7)


def test_count_histogram_cap_pools_large_counts(tmp_path):
    h = eda(tmp_path, cfg={**CFG, "count_histogram_cap": 2}).summary["article_day_count_distribution"]
    assert h["exact"] == {"1": 4, "2": 2} and h["pooled_above_cap"] == 1


def test_thresholds_report_without_choosing(tmp_path):
    t = {row["k"]: row for row in eda(tmp_path).summary["min_history_thresholds"]}
    assert t[1]["n_articles_events_ge_k"] == 3
    assert t[3]["n_articles_events_ge_k"] == 2 and t[3]["n_articles_active_days_ge_k"] == 2
    assert t[5]["n_articles_events_ge_k"] == 1 and t[5]["n_articles_active_days_ge_k"] == 0


def test_per_article_distribution_labels_and_values(tmp_path):
    d = eda(tmp_path).summary["per_article"]["n_events"]
    assert d["n"] == 3 and d["min"] == 1 and d["max"] == 6 and d["p50"] == 4
    assert d["mean"] == pytest.approx(11 / 3)
    assert {"p1", "p5", "p10", "p25", "p75", "p90", "p95", "p99"} <= set(d)


# -------------------------------------------------------------------------- daily


def test_daily_table_covers_full_calendar_with_explicit_zero_days(tmp_path):
    d = eda(tmp_path).daily_stats
    assert d.height == 10
    assert d["date"].to_list() == [date(2020, 1, day) for day in range(1, 11)]
    zero_days = d.filter(pl.col("n_events") == 0)["date"].to_list()
    assert zero_days == [date(2020, 1, 6), date(2020, 1, 8), date(2020, 1, 9)]
    assert d["n_events"].to_list() == [2, 1, 1, 3, 1, 0, 1, 0, 0, 2]
    assert d["n_active_articles"].to_list() == [1, 1, 1, 1, 1, 0, 1, 0, 0, 1]


def test_articles_in_span_per_day(tmp_path):
    d = eda(tmp_path).daily_stats
    assert d["n_in_span"].to_list() == [1, 1, 2, 2, 2, 1, 1, 1, 1, 1]
    assert d["active_share_of_in_span"].to_list()[2] == pytest.approx(0.5)  # Jan 3: 1 active of 2 in span
    assert d["active_share_of_in_span"].to_list()[5] == 0.0  # Jan 6: 0 active of 1 in span


def test_summary_scope_and_assortment(tmp_path):
    s = eda(tmp_path).summary
    assert s["scope"]["n_rows_total"] == 11 and s["scope"]["n_articles_with_sales"] == 3
    assert s["scope"]["n_calendar_days"] == 10 and s["scope"]["n_days_without_any_row"] == 3
    assert s["scope"]["first_date"] == "2020-01-01" and s["scope"]["last_date"] == "2020-01-10"
    assert s["assortment"]["active_articles_first_date"] == 1 and s["assortment"]["active_articles_last_date"] == 1


def test_lifecycle_censoring_and_recent_windows(tmp_path):
    lc = eda(tmp_path).summary["lifecycle"]
    assert lc["first_sale_on_first_date_of_data"] == 1  # ...101 on Jan 1
    assert lc["last_sale_on_last_date_of_data"] == 1  # ...102 on Jan 10
    assert lc["first_sale_within_first_2_days"] == 1  # first sale before Jan 3: only ...101
    assert lc["last_sale_within_last_2_days"] == 1  # last sale after Jan 8: only ...102
    assert lc["n_articles_single_active_day"] == 1  # ...103


def test_channel_counts_and_other(tmp_path):
    r = eda(tmp_path)
    c = r.summary["channel"]
    assert c["overall_events"] == {"1": 7, "2": 3, "other": 1}
    assert c["overall_share"]["1"] == pytest.approx(7 / 11)
    d = r.daily_stats
    assert (d["channel_1"] + d["channel_2"] + d["channel_other"]).to_list() == d["n_events"].to_list()


def test_spike_day_is_found(tmp_path):
    rows = [(f"2020-01-{d:02d}", f"c{i}", A, "0.1", "1") for d in range(1, 10) for i in range(10 if d == 5 else 1)]
    r = eda(tmp_path, rows=rows, cfg={**CFG, "spike_window_days": 5})
    top = r.summary["daily_volume"]["top_days"][0]
    assert top["date"] == "2020-01-05" and top["n_events"] == 10
    assert top["ratio_to_rolling_median"] == pytest.approx(10.0)
    assert r.summary["daily_volume"]["n_days_ratio_ge_2"] == 1


def test_monthly_table_mix_and_entries_exits(tmp_path):
    rows = [
        ("2020-01-30", "c1", A, "0.1", "1"),
        ("2020-01-31", "c1", A, "0.1", "2"),
        ("2020-02-01", "c1", A, "0.1", "1"),
        ("2020-02-02", "c2", B, "0.1", "1"),
    ]
    m = eda(tmp_path, rows=rows).monthly_summary
    jan, feb = (m.filter(pl.col("month") == date(2020, mth, 1)).row(0, named=True) for mth in (1, 2))
    assert jan["n_events"] == 2 and jan["channel_1_share"] == pytest.approx(0.5)
    assert feb["n_events"] == 2 and feb["channel_1_share"] == pytest.approx(1.0)
    assert (jan["n_articles_first_sold"], feb["n_articles_first_sold"]) == (1, 1)  # A in Jan, B in Feb
    assert (jan["n_articles_last_sold"], feb["n_articles_last_sold"]) == (0, 2)


# ------------------------------------------------------------------------- checks


def test_internal_checks_pass_and_reference_skipped_without_profile(tmp_path):
    r = eda(tmp_path)
    assert not ad.has_failed_checks(r)
    assert status(r, "article_total_equals_daily_total") == ad.PASS
    assert status(r, "histogram_total_equals_daily_total") == ad.PASS
    assert status(r, "channel_counts_add_up") == ad.PASS
    assert status(r, "reference_profile") == ad.SKIPPED


def test_reference_profile_match_and_mismatch(tmp_path):
    ref = {"n_rows": 11, "n_distinct_dates": 7, "n_calendar_days": 10, "n_articles_with_sales": 3}
    ok = eda(tmp_path, reference=ref)
    assert not ad.has_failed_checks(ok) and status(ok, "matches_stage2_profile_n_rows") == ad.PASS
    bad = eda(tmp_path, reference={**ref, "n_rows": 12})
    assert ad.has_failed_checks(bad) and bad.summary["checks_failed"] == ["matches_stage2_profile_n_rows"]


def test_reference_not_compared_for_trial_runs(tmp_path):
    ref = {"n_rows": 999, "n_distinct_dates": 1, "n_calendar_days": 1, "n_articles_with_sales": 1}
    r = eda(tmp_path, reference=ref, partial_rows=5)
    assert not ad.has_failed_checks(r) and status(r, "reference_profile") == ad.SKIPPED
    assert r.summary["scope"]["partial_scan_rows"] == 5


def test_unparseable_dates_fail_the_check_but_do_not_break_totals(tmp_path):
    r = eda(tmp_path, rows=ROWS + [("01/02/2020", "c9", A, "0.1", "1"), ("", "c9", B, "0.1", "1")])
    assert r.summary["scope"]["n_rows_unparseable_date"] == 2 and r.summary["scope"]["n_rows_total"] == 13
    assert r.summary["checks_failed"] == ["no_unparseable_dates"]
    assert r.article_stats["n_events"].sum() == 11  # bad-date rows excluded, not guessed


def test_empty_file_raises(tmp_path):
    with pytest.raises(ValueError):
        eda(tmp_path, rows=[])


def test_load_reference_profile(tmp_path):
    full = {
        "partial_scan_rows": None,
        "global": {"n_rows": 11},
        "dates": {"n_distinct_dates": 7, "n_calendar_days": 10},
        "article_key_coverage": {"exact_match": {"n_distinct_transaction_ids": 3}},
    }
    p = tmp_path / "p.json"
    p.write_text(json.dumps(full), encoding="utf-8")
    ref, note = ad.load_reference_profile(p)
    assert ref == {"n_rows": 11, "n_distinct_dates": 7, "n_calendar_days": 10, "n_articles_with_sales": 3} and note is None
    p.write_text(json.dumps({**full, "partial_scan_rows": 5}), encoding="utf-8")
    assert ad.load_reference_profile(p)[0] is None
    p.write_text("not json", encoding="utf-8")
    assert ad.load_reference_profile(p)[0] is None
    assert ad.load_reference_profile(tmp_path / "missing.json")[0] is None
    assert ad.load_reference_profile(None)[0] is None


# ------------------------------------------------------------------------ settings


def test_settings_defaults_override_and_validation():
    assert ad.settings({}) == ad.DEFAULTS and ad.settings(None) == ad.DEFAULTS
    assert ad.settings({"eda": {"article_daily": {"count_histogram_cap": 5}}})["count_histogram_cap"] == 5
    with pytest.raises(ValueError):
        ad.settings({"eda": {"article_daily": {"spike_window_days": 0}}})


def test_repo_config_has_matching_eda_section():
    from hm_forecast.config import DEFAULT_DATA_CONFIG, load_yaml

    section = load_yaml(DEFAULT_DATA_CONFIG)["eda"]["article_daily"]
    assert set(section) <= set(ad.DEFAULTS)


# ---------------------------------------------------------------------------- CLI


def _raw(tmp_path, rows=ROWS):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_tx(raw / "transactions_train.csv", rows)
    return raw


def test_cli_end_to_end_writes_all_outputs(tmp_path):
    out = tmp_path / "out"
    assert ad.main(["--raw-dir", str(_raw(tmp_path)), "--output-dir", str(out), "--no-reference"]) == 0
    for name in ["summary.json", "article_stats.parquet", "daily_stats.csv", "count_histogram.csv", "monthly_summary.csv"]:
        assert (out / name).is_file(), name
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["scope"]["n_rows_total"] == 11 and summary["checks_failed"] == []
    assert summary["benchmarks"].keys() == {"scan_1_article_stats", "scan_2_daily_stats", "scan_3_article_day_count_histogram"}


def test_cli_exit_1_when_reference_profile_disagrees(tmp_path):
    ref = {
        "partial_scan_rows": None,
        "global": {"n_rows": 12},
        "dates": {"n_distinct_dates": 7, "n_calendar_days": 10},
        "article_key_coverage": {"exact_match": {"n_distinct_transaction_ids": 3}},
    }
    ref_path = tmp_path / "ref.json"
    ref_path.write_text(json.dumps(ref), encoding="utf-8")
    args = ["--raw-dir", str(_raw(tmp_path)), "--output-dir", str(tmp_path / "out"), "--reference-profile", str(ref_path)]
    assert ad.main(args) == 1
    ref["global"]["n_rows"] = 11
    ref_path.write_text(json.dumps(ref), encoding="utf-8")
    assert ad.main(args) == 0


def test_cli_exit_2_when_file_missing_or_empty_or_config_missing(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert ad.main(["--raw-dir", str(empty), "--output-dir", str(tmp_path / "o")]) == 2
    raw = _raw(tmp_path)
    assert ad.main(["--raw-dir", str(raw), "--output-dir", str(tmp_path / "o"), "--config", str(tmp_path / "nope.yaml")]) == 2
    raw2 = tmp_path / "raw2"
    raw2.mkdir()
    write_tx(raw2 / "transactions_train.csv", [])
    assert ad.main(["--raw-dir", str(raw2), "--output-dir", str(tmp_path / "o")]) == 2


def test_cli_trial_run_uses_separate_default_folder(tmp_path, monkeypatch):
    full_dir, trial_dir = tmp_path / "full", tmp_path / "trial"
    monkeypatch.setattr(ad, "DEFAULT_OUTPUT_DIR", full_dir)
    monkeypatch.setattr(ad, "DEFAULT_OUTPUT_DIR_SAMPLE", trial_dir)
    assert ad.main(["--raw-dir", str(_raw(tmp_path)), "--n-rows", "4", "--no-reference"]) == 0
    assert (trial_dir / "summary.json").is_file() and not full_dir.exists()
    assert json.loads((trial_dir / "summary.json").read_text(encoding="utf-8"))["scope"]["n_rows_total"] == 4


def test_cli_rejects_non_positive_n_rows(tmp_path):
    with pytest.raises(SystemExit):
        ad.main(["--raw-dir", str(tmp_path), "--n-rows", "0"])
