"""Tests for the transactions profile. Synthetic data only."""

import csv
import json

import polars as pl

from hm_forecast.data import profile_transactions as pt

CFG = {"expected_channel_values": ["1", "2"], "article_id_width": 10, "max_missing_dates_listed": 20}
ARTICLES = pl.Series(["0000000101", "0000000102", "0000000103"])
HEADER = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]


def write_tx(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    return path


GOOD_ROWS = [
    ("2020-01-01", "c1", "101", "0.05", "1"),
    ("2020-01-01", "c2", "101", "0.05", "2"),
    ("2020-01-02", "c1", "102", "0.10", "1"),
    ("2020-01-04", "c3", "102", "0.10", "1"),  # 2020-01-03 has no rows
]


def profile(tmp_path, rows, articles=ARTICLES, cfg=CFG):
    path = write_tx(tmp_path / "tx.csv", rows)
    return pt.build_profile(pt.scan_transactions(path), articles, cfg)


def codes(p, severity=None):
    return {i["code"] for i in p["issues"] if severity is None or i["severity"] == severity}


def test_global_stats_basic_counts(tmp_path):
    p = profile(tmp_path, GOOD_ROWS)
    g = p["global"]
    assert g["n_rows"] == 4 and g["customer_n_unique"] == 3
    assert g["t_dat_min"] == "2020-01-01" and g["t_dat_max"] == "2020-01-04"
    assert g["channel_1"] == 3 and g["channel_2"] == 1
    assert g["price_min"] == 0.05 and g["price_max"] == 0.10
    assert g["article_len_min"] == g["article_len_max"] == 3


def test_unpadded_ids_match_after_padding_and_flag_info_only(tmp_path):
    p = profile(tmp_path, GOOD_ROWS)
    cov = p["article_key_coverage"]
    assert cov["exact_match"]["n_rows_unmatched"] == 4  # '101' != '0000000101'
    assert cov["zero_padded_to_10"]["n_rows_unmatched"] == 0
    assert "ARTICLE_ID_FORMAT_DIFFERS" in codes(p, "info")
    assert not pt.has_blocking_errors(p)


def test_already_padded_ids_match_exactly(tmp_path):
    rows = [("2020-01-01", "c1", "0000000101", "0.05", "1")]
    p = profile(tmp_path, rows)
    assert p["article_key_coverage"]["exact_match"]["n_rows_unmatched"] == 0
    assert "ARTICLE_ID_FORMAT_DIFFERS" not in codes(p)


def test_unmatched_article_is_blocking_error(tmp_path):
    rows = GOOD_ROWS + [("2020-01-04", "c9", "999", "0.1", "1")]
    p = profile(tmp_path, rows)
    assert "ARTICLE_KEY_UNMATCHED" in codes(p, "error")
    assert p["article_key_coverage"]["zero_padded_to_10"]["n_rows_unmatched"] == 1


def test_articles_never_sold_counted(tmp_path):
    p = profile(tmp_path, GOOD_ROWS)  # article 103 never appears
    assert p["article_key_coverage"]["n_articles_without_transactions"] == 1
    assert "ARTICLES_NEVER_SOLD" in codes(p, "info")


def test_date_gaps_reported(tmp_path):
    d = profile(tmp_path, GOOD_ROWS)["dates"]
    assert d["n_calendar_days"] == 4 and d["n_distinct_dates"] == 3
    assert d["n_missing_dates"] == 1 and d["missing_dates_listed"] == ["2020-01-03"]
    assert d["rows_per_day_max"] == 2


def test_date_gap_is_warning(tmp_path):
    assert "DATE_GAPS" in codes(profile(tmp_path, GOOD_ROWS), "warning")


def test_invalid_dates_are_errors(tmp_path):
    rows = GOOD_ROWS + [("01/02/2020", "c1", "101", "0.1", "1"), ("", "c1", "101", "0.1", "1")]
    p = profile(tmp_path, rows)
    assert "DATE_INVALID" in codes(p, "error")
    assert p["global"]["t_dat_unparseable"] >= 1


def test_price_problems(tmp_path):
    rows = GOOD_ROWS + [("2020-01-04", "c1", "101", "abc", "1"), ("2020-01-04", "c1", "101", "0", "1")]
    p = profile(tmp_path, rows)
    assert "PRICE_INVALID" in codes(p, "error") and "PRICE_NON_POSITIVE" in codes(p, "warning")
    assert p["global"]["price_non_numeric"] == 1


def test_scientific_notation_price_is_numeric(tmp_path):
    rows = [("2020-01-01", "c1", "101", "1.6e-05", "1")]
    assert profile(tmp_path, rows)["global"]["price_non_numeric"] == 0


def test_non_integer_and_missing_article_id_are_errors(tmp_path):
    rows = GOOD_ROWS + [("2020-01-04", "c1", "10x", "0.1", "1"), ("2020-01-04", "c1", "", "0.1", "1")]
    p = profile(tmp_path, rows)
    assert {"ARTICLE_ID_NON_INTEGER", "ARTICLE_ID_MISSING"} <= codes(p, "error")


def test_unexpected_channel_is_warning(tmp_path):
    rows = GOOD_ROWS + [("2020-01-04", "c1", "101", "0.1", "3")]
    p = profile(tmp_path, rows)
    assert "CHANNEL_UNEXPECTED" in codes(p, "warning") and p["global"]["channel_unexpected"] == 1


def test_duplicate_rows_are_info_not_defects(tmp_path):
    rows = GOOD_ROWS + [GOOD_ROWS[0], GOOD_ROWS[0]]
    p = profile(tmp_path, rows)
    assert p["approx_duplicate_rows"] == 2
    assert "DUPLICATE_ROWS_APPROX" in codes(p, "info") and not pt.has_blocking_errors(p)


def test_empty_file_reports_no_rows(tmp_path):
    p = profile(tmp_path, [])
    assert "NO_ROWS" in codes(p, "error")


def test_benchmarks_recorded_for_each_scan(tmp_path):
    b = profile(tmp_path, GOOD_ROWS)["benchmarks"]
    assert set(b) == {"scan_1_global_stats", "scan_2_daily_counts", "scan_3_article_counts"}
    assert all(v["seconds"] >= 0 for v in b.values())


def test_peak_memory_tracker_runs():
    with pt.PeakMemory() as mem:
        _ = [0] * 1000
    assert mem.peak_mb is None or mem.peak_mb > 0


def _write_raw(raw, tx_rows):
    raw.mkdir()
    write_tx(raw / "transactions_train.csv", tx_rows)
    with (raw / "articles.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["article_id", "prod_name"])
        w.writerows([["0000000101", "a"], ["0000000102", "b"], ["0000000103", "c"]])


def test_cli_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    _write_raw(raw, GOOD_ROWS)
    out = tmp_path / "out.json"
    assert pt.main(["--raw-dir", str(raw), "--output", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["global"]["n_rows"] == 4 and report["blocking_errors"] is False


def test_cli_exit_1_on_unmatched_key(tmp_path):
    raw = tmp_path / "raw"
    _write_raw(raw, GOOD_ROWS + [("2020-01-04", "c9", "999", "0.1", "1")])
    assert pt.main(["--raw-dir", str(raw), "--output", str(tmp_path / "o.json")]) == 1


def test_cli_exit_2_when_files_missing_or_malformed(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    assert pt.main(["--raw-dir", str(raw), "--output", str(tmp_path / "o.json")]) == 2
    _write_raw(tmp_path / "raw2", GOOD_ROWS)
    (tmp_path / "raw2" / "transactions_train.csv").write_text(
        ",".join(HEADER) + "\n2020-01-01,c1,101,0.1,1,EXTRA\n", encoding="utf-8"
    )
    assert pt.main(["--raw-dir", str(tmp_path / "raw2"), "--output", str(tmp_path / "o.json")]) == 2


def test_n_rows_limits_scan_and_flags_partial(tmp_path):
    path = write_tx(tmp_path / "tx.csv", GOOD_ROWS)
    p = pt.build_profile(pt.scan_transactions(path, n_rows=2), ARTICLES, CFG, partial_rows=2)
    assert p["global"]["n_rows"] == 2 and p["partial_scan_rows"] == 2
    assert "PARTIAL_SCAN" in codes(p, "info")


def test_cli_trial_run_writes_separate_report(tmp_path):
    raw = tmp_path / "raw"
    _write_raw(raw, GOOD_ROWS)
    out = tmp_path / "trial.json"
    assert pt.main(["--raw-dir", str(raw), "--n-rows", "2", "--output", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["partial_scan_rows"] == 2


def test_cli_rejects_non_positive_n_rows(tmp_path):
    import pytest

    with pytest.raises(SystemExit):
        pt.main(["--raw-dir", str(tmp_path), "--n-rows", "0"])
