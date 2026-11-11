"""Tests for the validation gate. Synthetic data only."""

import csv
import hashlib
import json

import pytest

from hm_forecast.config import DEFAULT_DATA_CONFIG, load_yaml
from hm_forecast.data import validate as gate

TX_HEADER = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
ARTICLE_IDS = ["0000000101", "0000000102", "0000000103"]


def write_articles(path, ids=ARTICLE_IDS):
    spec = load_yaml(DEFAULT_DATA_CONFIG)["files"]["articles.csv"]["columns"]
    cols = list(spec)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, article_id in enumerate(ids):
            row = {c: ("1" if str(spec[c]).startswith("integer") else "x") for c in cols}
            row["article_id"] = article_id
            row["product_code"] = str(i)
            w.writerow([row[c] for c in cols])


def write_transactions(path, article_ids=("0000000101", "0000000102")):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(TX_HEADER)
        for day, article_id in enumerate(article_ids, start=1):
            w.writerow([f"2020-01-0{day}", "c1", article_id, "0.05", "1"])


@pytest.fixture
def raw(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    write_articles(d / "articles.csv")
    write_transactions(d / "transactions_train.csv")
    return d


def run(raw, tmp_path, *extra):
    out = tmp_path / "out"
    code = gate.main(["--raw-dir", str(raw), "--output-dir", str(out), *extra])
    summary_path = out / gate.SUMMARY_NAME
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    return code, summary


def statuses(summary):
    return {s["name"]: s["status"] for s in summary["stages"]}


def test_all_stages_pass(raw, tmp_path):
    code, summary = run(raw, tmp_path)
    assert code == 0 and summary["overall"] == "passed" and summary["blocking_errors"] is False
    assert statuses(summary) == {"inventory": "passed", "articles_profile": "passed", "transactions_profile": "passed"}
    assert summary["full_validation"] is True


def test_summary_counts_issues_by_severity(raw, tmp_path):
    _, summary = run(raw, tmp_path)
    inv = next(s for s in summary["stages"] if s["name"] == "inventory")
    # tiny files vs expected sizes -> warnings; missing optional customers.csv -> warning
    assert inv["counts"]["errors"] == 0 and inv["counts"]["warnings"] >= 1


def test_inventory_failure_skips_remaining_stages(raw, tmp_path):
    (raw / "transactions_train.csv").unlink()
    code, summary = run(raw, tmp_path)
    assert code == 1 and summary["overall"] == "failed"
    assert statuses(summary) == {"inventory": "failed", "articles_profile": "skipped", "transactions_profile": "skipped"}
    skipped = [s for s in summary["stages"] if s["status"] == "skipped"]
    assert all("inventory did not pass" in s["note"] for s in skipped)


def test_articles_failure_does_not_stop_transactions(raw, tmp_path):
    write_articles(raw / "articles.csv", ids=["0000000101", "0000000101", "0000000103"])  # duplicate key
    code, summary = run(raw, tmp_path)
    assert code == 1
    assert statuses(summary)["articles_profile"] == "failed"
    assert statuses(summary)["transactions_profile"] != "skipped"


def test_transactions_unmatched_key_fails_gate(raw, tmp_path):
    write_transactions(raw / "transactions_train.csv", article_ids=("0000000101", "0000000999"))
    code, summary = run(raw, tmp_path)
    assert code == 1 and statuses(summary)["transactions_profile"] == "failed"


def test_skip_transactions_is_recorded_as_not_full_validation(raw, tmp_path):
    code, summary = run(raw, tmp_path, "--skip-transactions")
    assert code == 0 and summary["full_validation"] is False
    assert statuses(summary)["transactions_profile"] == "skipped"
    assert not (tmp_path / "out" / "transactions_profile.json").exists()


def test_bad_raw_dir_is_configuration_error(tmp_path):
    code, summary = run(tmp_path / "does_not_exist", tmp_path)
    assert code == 2 and summary is None


def test_undecodable_articles_is_caught_by_inventory(raw, tmp_path):
    (raw / "articles.csv").write_bytes(b"article_id\n\xff\xfe\xfa\n")
    code, summary = run(raw, tmp_path)
    assert code == 1 and summary["overall"] == "failed"
    assert statuses(summary) == {"inventory": "failed", "articles_profile": "skipped", "transactions_profile": "skipped"}


def test_overall_exit_code_precedence():
    assert gate.overall_exit_code([{"exit_code": 0}, {"exit_code": None}]) == 0
    assert gate.overall_exit_code([{"exit_code": 0}, {"exit_code": 1}]) == 1
    assert gate.overall_exit_code([{"exit_code": 1}, {"exit_code": 2}]) == 2


def test_file_fingerprint_without_and_with_hash(tmp_path):
    p = tmp_path / "f.csv"
    p.write_bytes(b"abc,def\n1,2\n")
    plain = gate.file_fingerprint(p)
    assert plain["size_bytes"] == 12 and "sha256" not in plain
    hashed = gate.file_fingerprint(p, sha256=True)
    assert hashed["sha256"] == hashlib.sha256(b"abc,def\n1,2\n").hexdigest()


def test_summary_records_data_files_and_revision(raw, tmp_path):
    _, summary = run(raw, tmp_path, "--sha256")
    assert set(summary["data_files"]) == {"transactions_train.csv", "articles.csv"}  # customers absent
    assert all("sha256" in v for v in summary["data_files"].values())
    assert "code_revision" in summary  # a dict inside a Git checkout, otherwise None


def test_count_issues_handles_missing_report(tmp_path):
    assert gate.count_issues(tmp_path / "absent.json") is None
