"""Tests for the raw-file inventory and config helpers. Synthetic data only."""

import json

import pytest

from hm_forecast.config import load_yaml, resolve_raw_data_dir, DEFAULT_DATA_CONFIG
from hm_forecast.data import inventory as inv

SPEC = {
    "required": True,
    "approx_size_mb": None,
    "columns": {"a": "int", "b": "string"},
}


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def codes(result):
    return {i.code for i in result.issues}


def test_valid_file_has_no_issues(tmp_path):
    write(tmp_path / "f.csv", "a,b\n1,x\n")
    r = inv.inventory_file(tmp_path, "f.csv", SPEC, 2.0)
    assert r.exists and r.columns == ["a", "b"] and r.issues == []


def test_missing_required_file_is_error(tmp_path):
    r = inv.inventory_file(tmp_path, "nope.csv", SPEC, 2.0)
    assert not r.exists and r.has_errors and "FILE_MISSING" in codes(r)


def test_missing_optional_file_is_warning_only(tmp_path):
    r = inv.inventory_file(tmp_path, "nope.csv", {**SPEC, "required": False}, 2.0)
    assert not r.has_errors and "FILE_MISSING" in codes(r)


def test_empty_file_is_error(tmp_path):
    write(tmp_path / "f.csv", "")
    assert "EMPTY_FILE" in codes(inv.inventory_file(tmp_path, "f.csv", SPEC, 2.0))


def test_missing_column_is_error(tmp_path):
    write(tmp_path / "f.csv", "a\n1\n")
    r = inv.inventory_file(tmp_path, "f.csv", SPEC, 2.0)
    assert r.has_errors and r.missing_columns == ["b"]


def test_unexpected_column_is_warning_not_error(tmp_path):
    write(tmp_path / "f.csv", "a,b,c\n1,x,9\n")
    r = inv.inventory_file(tmp_path, "f.csv", SPEC, 2.0)
    assert not r.has_errors and r.unexpected_columns == ["c"]


def test_duplicate_header_is_error(tmp_path):
    write(tmp_path / "f.csv", "a,a,b\n1,2,x\n")
    assert "DUPLICATE_COLUMNS" in codes(inv.inventory_file(tmp_path, "f.csv", SPEC, 2.0))


def test_column_order_does_not_matter(tmp_path):
    write(tmp_path / "f.csv", "b,a\nx,1\n")
    assert inv.inventory_file(tmp_path, "f.csv", SPEC, 2.0).issues == []


def test_utf8_bom_is_tolerated(tmp_path):
    (tmp_path / "f.csv").write_bytes(b"\xef\xbb\xbfa,b\n1,x\n")
    r = inv.inventory_file(tmp_path, "f.csv", SPEC, 2.0)
    assert r.columns == ["a", "b"] and r.issues == []


def test_undecodable_header_is_error(tmp_path):
    (tmp_path / "f.csv").write_bytes(b"\xff\xfe\x00a,b\n")
    assert "HEADER_UNREADABLE" in codes(inv.inventory_file(tmp_path, "f.csv", SPEC, 2.0))


def test_size_far_from_expected_is_warning(tmp_path):
    write(tmp_path / "f.csv", "a,b\n1,x\n")  # a few bytes vs expected 100 MB
    r = inv.inventory_file(tmp_path, "f.csv", {**SPEC, "approx_size_mb": 100}, 2.0)
    assert "SIZE_UNEXPECTED" in codes(r) and not r.has_errors


def test_directory_instead_of_file_is_error(tmp_path):
    (tmp_path / "f.csv").mkdir()
    assert "NOT_A_FILE" in codes(inv.inventory_file(tmp_path, "f.csv", SPEC, 2.0))


def test_blocking_error_detection(tmp_path):
    write(tmp_path / "ok.csv", "a,b\n1,x\n")
    cfg = {"files": {"ok.csv": SPEC, "missing.csv": SPEC}}
    results = inv.build_inventory(tmp_path, cfg)
    assert inv.has_blocking_errors(results)
    assert not inv.has_blocking_errors(results[:1])


def test_project_data_config_is_well_formed():
    cfg = load_yaml(DEFAULT_DATA_CONFIG)
    assert set(cfg["files"]) == {"transactions_train.csv", "articles.csv", "customers.csv"}
    for spec in cfg["files"].values():
        assert spec["columns"] and isinstance(spec["required"], bool)
    assert len(cfg["files"]["articles.csv"]["columns"]) == 25


def test_resolve_raw_data_dir(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    local = write(tmp_path / "local.yaml", f"raw_data_dir: '{data_dir.as_posix()}'\n")
    assert resolve_raw_data_dir(local) == data_dir


def test_resolve_raw_data_dir_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_raw_data_dir(tmp_path / "absent.yaml")
    bad = write(tmp_path / "bad.yaml", "raw_data_dir: ''\n")
    with pytest.raises(ValueError):
        resolve_raw_data_dir(bad)
    gone = write(tmp_path / "gone.yaml", f"raw_data_dir: '{(tmp_path / 'nodir').as_posix()}'\n")
    with pytest.raises(FileNotFoundError):
        resolve_raw_data_dir(gone)


def test_cli_exit_codes_and_report(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    cfg_path = write(
        tmp_path / "data.yaml",
        "files:\n  f.csv:\n    required: true\n    columns: {a: int, b: string}\n",
    )
    out = tmp_path / "out" / "report.json"
    args = ["--raw-dir", str(raw), "--config", str(cfg_path), "--output", str(out)]

    assert inv.main(args) == 1  # required file missing -> blocking
    assert json.loads(out.read_text())["blocking_errors"] is True

    write(raw / "f.csv", "a,b\n1,x\n")
    assert inv.main(args) == 0
    assert json.loads(out.read_text())["blocking_errors"] is False

    assert inv.main(["--raw-dir", str(tmp_path / "nodir"), "--config", str(cfg_path)]) == 2
