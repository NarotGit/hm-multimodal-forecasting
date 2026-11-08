"""Tests for the articles profile. Synthetic data only."""

import json

import polars as pl
import pytest

from hm_forecast.config import DEFAULT_DATA_CONFIG, load_yaml
from hm_forecast.data import profile_articles as pa

SPEC = {
    "columns": {
        "article_id": "integer",
        "product_code": "integer",
        "type_no": "integer",
        "type_name": "string",
        "detail_desc": "string",
    }
}
PROFILING = {
    "key_column": "article_id",
    "group_column": "product_code",
    "text_columns": ["detail_desc"],
    "code_name_pairs": [["type_no", "type_name"]],
}


def make_df(rows):
    cols = ["article_id", "product_code", "type_no", "type_name", "detail_desc"]
    return pl.DataFrame(rows, schema=cols, orient="row").cast(pl.Utf8)


GOOD = [
    ("101", "10", "1", "Shirt", "A nice shirt"),
    ("102", "10", "1", "Shirt", "A nice shirt"),
    ("103", "11", "2", "Dress", "Long dress"),
    ("104", "12", "2", "Dress", None),
]


def codes(profile, severity=None):
    return {i["code"] for i in profile["issues"] if severity is None or i["severity"] == severity}


def test_clean_table_has_no_blocking_or_warning_issues():
    p = pa.profile_articles(make_df(GOOD), SPEC, PROFILING)
    assert not pa.has_blocking_errors(p)
    assert codes(p, "error") == set() and codes(p, "warning") == set()
    assert p["n_rows"] == 4 and p["key"]["n_unique"] == 4


def test_duplicate_key_is_error():
    rows = GOOD + [("101", "13", "3", "Skirt", "x")]
    p = pa.profile_articles(make_df(rows), SPEC, PROFILING)
    assert "KEY_DUPLICATE" in codes(p, "error") and p["key"]["n_duplicate_rows"] == 1


def test_missing_and_non_integer_key_are_errors():
    rows = GOOD + [(None, "13", "3", "Skirt", "x"), ("abc", "14", "3", "Skirt", "x")]
    p = pa.profile_articles(make_df(rows), SPEC, PROFILING)
    assert {"KEY_MISSING", "KEY_NON_INTEGER"} <= codes(p, "error")


def test_leading_zero_key_is_warning():
    rows = [("0101", "10", "1", "Shirt", "x")] + GOOD[1:]
    p = pa.profile_articles(make_df(rows), SPEC, PROFILING)
    assert "KEY_LEADING_ZERO" in codes(p, "warning") and not pa.has_blocking_errors(p)


def test_negative_integer_is_integer_like():
    rows = [("101", "10", "-1", "Unknown", "x")]
    p = pa.profile_articles(make_df(rows), SPEC, PROFILING)
    assert p["columns"]["type_no"]["n_non_integer"] == 0


def test_non_integer_in_other_integer_column_is_warning():
    rows = [("101", "10", "x1", "Shirt", "x")] + GOOD[1:]
    p = pa.profile_articles(make_df(rows), SPEC, PROFILING)
    assert "INTEGER_COLUMN_NON_INTEGER" in codes(p, "warning")


def test_text_coverage_and_lengths():
    rows = [
        ("1", "1", "1", "A", "abcd"),
        ("2", "1", "1", "A", "  ab  "),  # stripped length 2, has edge whitespace
        ("3", "2", "1", "A", "   "),  # blank
        ("4", "2", "1", "A", None),  # null
    ]
    p = pa.profile_articles(make_df(rows), SPEC, PROFILING)
    t = p["text"]["detail_desc"]
    assert t["n_usable"] == 2 and t["n_missing_or_blank"] == 2
    assert t["pct_missing_or_blank"] == 50.0
    assert t["length_min"] == 2 and t["length_max"] == 4
    assert "EDGE_WHITESPACE" in codes(p, "info")


def test_code_name_inconsistency_detected():
    rows = [
        ("1", "1", "1", "Shirt", "x"),
        ("2", "1", "1", "Blouse", "x"),  # code 1 -> two names
        ("3", "2", "2", "Dress", "x"),
        ("4", "2", "3", "Dress", "x"),  # name Dress -> two codes
    ]
    p = pa.profile_articles(make_df(rows), SPEC, PROFILING)
    c = p["code_name_consistency"][0]
    assert c["n_codes_with_multiple_names"] == 1 and c["n_names_with_multiple_codes"] == 1
    assert "CODE_NAME_INCONSISTENT" in codes(p, "warning")


def test_code_name_missing_values_are_excluded_not_flagged():
    rows = [("1", "1", "1", "Shirt", "x"), ("2", "1", "1", None, "x")]
    p = pa.profile_articles(make_df(rows), SPEC, PROFILING)
    c = p["code_name_consistency"][0]
    assert c["n_rows_excluded_for_missing"] == 1 and c["n_codes_with_multiple_names"] == 0


def test_product_group_sizes():
    g = pa.profile_articles(make_df(GOOD), SPEC, PROFILING)["product_groups"]
    assert g["n_groups"] == 3
    assert g["articles_per_group_max"] == 2 and g["n_groups_with_single_article"] == 2


def test_full_duplicate_rows_warning():
    rows = GOOD + [GOOD[0]]
    p = pa.profile_articles(make_df(rows), SPEC, PROFILING)
    assert p["n_full_duplicate_rows"] == 2 and "FULL_ROW_DUPLICATES" in codes(p, "warning")


def test_missing_expected_column_raises():
    df = make_df(GOOD).drop("type_name")
    with pytest.raises(ValueError):
        pa.profile_articles(df, SPEC, PROFILING)


def test_read_articles_keeps_everything_as_text(tmp_path):
    path = tmp_path / "articles.csv"
    path.write_text('article_id,desc\n0108775015,"a, b"\n', encoding="utf-8")
    df = pa.read_articles(path)
    assert df.schema["article_id"] == pl.Utf8
    assert df["article_id"][0] == "0108775015"  # leading zero preserved
    assert df["desc"][0] == "a, b"  # quoted comma handled


def _write_real_schema_csv(path, duplicate_key=False):
    spec = load_yaml(DEFAULT_DATA_CONFIG)["files"]["articles.csv"]["columns"]
    cols = list(spec)
    rows = []
    for i in range(3):
        row = {c: ("1" if str(spec[c]).startswith("integer") else "x") for c in cols}
        row["article_id"] = "100" if (duplicate_key and i < 2) else str(100 + i)
        rows.append(",".join(row[c] for c in cols))
    path.write_text(",".join(cols) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_cli_end_to_end_with_project_config(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_real_schema_csv(raw / "articles.csv")
    out = tmp_path / "out.json"
    assert pa.main(["--raw-dir", str(raw), "--output", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["n_rows"] == 3 and report["blocking_errors"] is False


def test_cli_exit_1_on_blocking_error(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_real_schema_csv(raw / "articles.csv", duplicate_key=True)
    assert pa.main(["--raw-dir", str(raw), "--output", str(tmp_path / "o.json")]) == 1


def test_cli_exit_2_when_file_missing_or_unreadable(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    assert pa.main(["--raw-dir", str(raw), "--output", str(tmp_path / "o.json")]) == 2
    (raw / "articles.csv").write_bytes(b"article_id\n\xff\xfe\xfa\n")
    assert pa.main(["--raw-dir", str(raw), "--output", str(tmp_path / "o.json")]) == 2
