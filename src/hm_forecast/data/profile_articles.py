"""Content profile of ``articles.csv``.

Second data-quality boundary. ``articles.csv`` is small, so it is read fully, but
**every column is read as text** (no type inference). This way nothing is coerced
or silently altered, and we can measure exactly how many values would fail an
integer conversion or lose information (for example leading zeros).

Policy
------
ERROR   (blocks later stages): key column has missing values, non-integer values
        or duplicates.
WARNING: key with leading zeros, non-integer values in other integer columns,
        code/name pairs that are not one-to-one, fully duplicated rows.
INFO:   missing-value rates, whitespace at value edges, text length statistics.
        These describe the data; they are not defects by themselves.

Usage:  python -m hm_forecast.data.profile_articles
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from hm_forecast.config import DEFAULT_DATA_CONFIG, REPO_ROOT, load_yaml, resolve_raw_data_dir
from hm_forecast.data.inventory import ERROR, WARNING, Issue

INFO = "info"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "validation" / "articles_profile.json"
INTEGER_PATTERN = r"^-?\d+$"
MAX_EXAMPLES = 5


def read_articles(path: Path) -> pl.DataFrame:
    """Read the CSV with every column as text (no schema inference)."""
    return pl.read_csv(path, infer_schema=False)


def _number(value):
    """Convert a Polars scalar to a plain JSON-friendly number (or None)."""
    if value is None:
        return None
    return round(float(value), 2)


def column_stats(series: pl.Series) -> dict:
    """Missingness and whitespace statistics for one text column."""
    n = series.len()
    stripped = series.str.strip_chars()
    n_null = series.null_count()
    n_blank = int((stripped == "").sum() or 0)
    n_edge_ws = int((series != stripped).sum() or 0)
    return {
        "n_null": n_null,
        "n_blank_whitespace": n_blank,
        "pct_missing_or_blank": round(100 * (n_null + n_blank) / n, 3) if n else None,
        "n_values_with_edge_whitespace": n_edge_ws,
        "n_unique": series.drop_nulls().n_unique(),
    }


def non_integer_count(series: pl.Series) -> int:
    """Number of non-missing values that are not integer-like."""
    present = series.drop_nulls()
    return int((~present.str.contains(INTEGER_PATTERN)).sum() or 0)


def key_stats(series: pl.Series) -> dict:
    """Integrity statistics for the article key."""
    present = series.drop_nulls()
    lengths = present.str.len_chars()
    return {
        "n_null": series.null_count(),
        "n_blank_whitespace": int((series.str.strip_chars() == "").sum() or 0),
        "n_non_integer": non_integer_count(series),
        "n_unique": present.n_unique(),
        "n_duplicate_rows": present.len() - present.n_unique(),
        "n_leading_zero": int(present.str.contains(r"^0\d").sum() or 0),
        "min_length": int(lengths.min()) if present.len() else None,
        "max_length": int(lengths.max()) if present.len() else None,
    }


def text_stats(series: pl.Series) -> dict:
    """Coverage and length of a free-text column (length in characters, after stripping)."""
    n = series.len()
    stripped = series.str.strip_chars()
    usable = stripped.filter(stripped.is_not_null() & (stripped != ""))
    lengths = usable.str.len_chars()
    return {
        "n_rows": n,
        "n_usable": usable.len(),
        "n_missing_or_blank": n - usable.len(),
        "pct_missing_or_blank": round(100 * (n - usable.len()) / n, 3) if n else None,
        "n_unique_texts": usable.n_unique(),
        "length_min": int(lengths.min()) if usable.len() else None,
        "length_median": _number(lengths.median()) if usable.len() else None,
        "length_p95": _number(lengths.quantile(0.95)) if usable.len() else None,
        "length_max": int(lengths.max()) if usable.len() else None,
    }


def code_name_consistency(df: pl.DataFrame, code: str, name: str) -> dict:
    """Check that a code column and its name column map one-to-one."""
    both = df.select([code, name])
    pairs = both.drop_nulls().unique()
    n_excluded = both.height - both.drop_nulls().height

    names_per_code = (
        pairs.group_by(code).agg(pl.col(name).sort().alias("names")).filter(pl.col("names").list.len() > 1)
    )
    codes_per_name = (
        pairs.group_by(name).agg(pl.col(code).sort().alias("codes")).filter(pl.col("codes").list.len() > 1)
    )
    return {
        "code_column": code,
        "name_column": name,
        "n_distinct_pairs": pairs.height,
        "n_rows_excluded_for_missing": n_excluded,
        "n_codes_with_multiple_names": names_per_code.height,
        "n_names_with_multiple_codes": codes_per_name.height,
        "examples_codes_with_multiple_names": names_per_code.head(MAX_EXAMPLES).to_dicts(),
        "examples_names_with_multiple_codes": codes_per_name.head(MAX_EXAMPLES).to_dicts(),
    }


def group_sizes(df: pl.DataFrame, group_column: str) -> dict:
    """How many articles share each value of ``group_column`` (for example product_code)."""
    sizes = df.drop_nulls(group_column).group_by(group_column).len()["len"]
    return {
        "group_column": group_column,
        "n_groups": sizes.len(),
        "articles_per_group_min": int(sizes.min()),
        "articles_per_group_median": _number(sizes.median()),
        "articles_per_group_max": int(sizes.max()),
        "n_groups_with_single_article": int((sizes == 1).sum()),
    }


def profile_articles(df: pl.DataFrame, spec: dict, profiling: dict) -> dict:
    """Build the full profile and the list of issues for an all-text articles table."""
    key = profiling["key_column"]
    expected = list(spec["columns"])
    integer_columns = [c for c, d in spec["columns"].items() if str(d).startswith("integer")]
    issues: list[Issue] = []

    absent = [c for c in expected if c not in df.columns]
    if absent:
        raise ValueError(f"columns required for profiling are missing: {absent}")

    columns = {c: column_stats(df[c]) for c in df.columns}
    for c in integer_columns:
        columns[c]["n_non_integer"] = non_integer_count(df[c])
    ks = key_stats(df[key])
    texts = {c: text_stats(df[c]) for c in profiling.get("text_columns", [])}
    pairs = [code_name_consistency(df, c, n) for c, n in profiling.get("code_name_pairs", [])]
    groups = group_sizes(df, profiling["group_column"]) if profiling.get("group_column") else None
    n_full_duplicates = int(df.is_duplicated().sum())

    # ---- issue classification (policy in the module docstring) ----
    if ks["n_null"] or ks["n_blank_whitespace"]:
        issues.append(Issue(ERROR, "KEY_MISSING", f"{key}: {ks['n_null']} null and {ks['n_blank_whitespace']} blank values"))
    if ks["n_non_integer"]:
        issues.append(Issue(ERROR, "KEY_NON_INTEGER", f"{key}: {ks['n_non_integer']} values are not integer-like"))
    if ks["n_duplicate_rows"]:
        issues.append(Issue(ERROR, "KEY_DUPLICATE", f"{key}: {ks['n_duplicate_rows']} duplicated key values"))
    if ks["n_leading_zero"]:
        issues.append(Issue(WARNING, "KEY_LEADING_ZERO", f"{key}: {ks['n_leading_zero']} values start with 0; integer conversion would drop it"))
    for c in integer_columns:
        if c != key and columns[c]["n_non_integer"]:
            issues.append(Issue(WARNING, "INTEGER_COLUMN_NON_INTEGER", f"{c}: {columns[c]['n_non_integer']} values are not integer-like"))
    for p in pairs:
        if p["n_codes_with_multiple_names"] or p["n_names_with_multiple_codes"]:
            issues.append(
                Issue(
                    WARNING,
                    "CODE_NAME_INCONSISTENT",
                    f"{p['code_column']}/{p['name_column']}: {p['n_codes_with_multiple_names']} codes with several names, "
                    f"{p['n_names_with_multiple_codes']} names with several codes",
                )
            )
    if n_full_duplicates:
        issues.append(Issue(WARNING, "FULL_ROW_DUPLICATES", f"{n_full_duplicates} rows are exact duplicates of another row"))

    for c, s in columns.items():
        if s["n_null"] or s["n_blank_whitespace"]:
            issues.append(Issue(INFO, "MISSING_VALUES", f"{c}: {s['n_null'] + s['n_blank_whitespace']} missing/blank ({s['pct_missing_or_blank']}%)"))
    ws = [c for c, s in columns.items() if s["n_values_with_edge_whitespace"]]
    if ws:
        issues.append(Issue(INFO, "EDGE_WHITESPACE", f"leading/trailing whitespace present in: {ws}"))

    return {
        "n_rows": df.height,
        "n_columns": df.width,
        "n_full_duplicate_rows": n_full_duplicates,
        "key": {"column": key, **ks},
        "columns": columns,
        "text": texts,
        "code_name_consistency": pairs,
        "product_groups": groups,
        "issues": [vars(i) for i in issues],
    }


def has_blocking_errors(profile: dict) -> bool:
    return any(i["severity"] == ERROR for i in profile["issues"])


def format_summary(profile: dict) -> str:
    k = profile["key"]
    lines = [
        f"Rows: {profile['n_rows']:,}   Columns: {profile['n_columns']}",
        f"Key {k['column']}: unique={k['n_unique']:,} duplicates={k['n_duplicate_rows']} "
        f"null={k['n_null']} non_integer={k['n_non_integer']} leading_zero={k['n_leading_zero']} "
        f"length={k['min_length']}-{k['max_length']}",
    ]
    g = profile["product_groups"]
    if g:
        lines.append(
            f"{g['group_column']}: {g['n_groups']:,} groups; articles per group "
            f"min/median/max = {g['articles_per_group_min']}/{g['articles_per_group_median']}/{g['articles_per_group_max']}"
        )
    for c, t in profile["text"].items():
        lines.append(
            f"Text {c}: usable={t['n_usable']:,} missing/blank={t['n_missing_or_blank']:,} "
            f"({t['pct_missing_or_blank']}%) unique={t['n_unique_texts']:,} "
            f"length median/p95/max = {t['length_median']}/{t['length_p95']}/{t['length_max']}"
        )
    lines.append("Issues:")
    for i in profile["issues"]:
        lines.append(f"  {i['severity'].upper():7} {i['code']}: {i['message']}")
    if not profile["issues"]:
        lines.append("  none")
    lines.append("Result: " + ("BLOCKING ERRORS FOUND" if has_blocking_errors(profile) else "no blocking errors"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile articles.csv (all columns read as text).")
    parser.add_argument("--raw-dir", type=Path, help="Override raw_data_dir from configs/local.yaml.")
    parser.add_argument("--config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    try:
        raw_dir = args.raw_dir if args.raw_dir else resolve_raw_data_dir()
        config = load_yaml(args.config)
        spec = config["files"]["articles.csv"]
        profiling = config["profiling"]["articles"]
        path = raw_dir / "articles.csv"
        if not path.is_file():
            raise FileNotFoundError(f"articles.csv not found in {raw_dir}")
        df = read_articles(path)
        profile = profile_articles(df, spec, profiling)
    except (FileNotFoundError, ValueError, KeyError, OSError, pl.exceptions.PolarsError) as exc:
        print(f"Could not profile articles.csv: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(f"Source: {path}")
    print(format_summary(profile))
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": str(path),
        "blocking_errors": has_blocking_errors(profile),
        **profile,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report written to: {args.output}")
    return 1 if has_blocking_errors(profile) else 0


if __name__ == "__main__":
    sys.exit(main())
