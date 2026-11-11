"""Single validation gate for the raw H&M data.

Runs the three data-quality checks in order and writes one combined summary:

1. inventory           (files, headers, sizes)
2. articles profile    (key integrity, text coverage, code/name consistency)
3. transactions profile (formats, dates, key coverage; heaviest, ~25 s on the real file)

Policy
------
* If the inventory has blocking errors, the remaining stages are **skipped**: there
  is no point profiling files that are missing or malformed.
* Otherwise every stage runs, so one invocation shows all problems at once.
* Trial runs (``--n-rows``) are deliberately not available here: a gate must
  validate the whole file.

Exit codes: 0 = all stages passed, 1 = blocking data errors, 2 = a stage could not
run (configuration problem, unreadable file).

Usage:  python -m hm_forecast.data.validate [--skip-transactions] [--sha256]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from hm_forecast.config import DEFAULT_DATA_CONFIG, REPO_ROOT, load_yaml, resolve_raw_data_dir
from hm_forecast.data import inventory, profile_articles, profile_transactions

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "validation"
SUMMARY_NAME = "validation_summary.json"
HASH_CHUNK_BYTES = 8 * 1024 * 1024

# (stage name, stage main(), report file name)
STAGES = [
    ("inventory", inventory.main, "data_inventory.json"),
    ("articles_profile", profile_articles.main, "articles_profile.json"),
    ("transactions_profile", profile_transactions.main, "transactions_profile.json"),
]
STATUS_BY_EXIT_CODE = {0: "passed", 1: "failed", 2: "error"}


def file_fingerprint(path: Path, sha256: bool = False) -> dict:
    """Size, modification time and (optionally) SHA-256 of a data file."""
    stat = path.stat()
    info = {
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
    }
    if sha256:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
        info["sha256"] = digest.hexdigest()
    return info


def code_revision() -> dict | None:
    """Current Git commit and whether the working tree has uncommitted changes (None if unavailable)."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        if head.returncode != 0:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        return {"commit": head.stdout.strip(), "uncommitted_changes": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return None


def count_issues(report_path: Path) -> dict | None:
    """Count issues by severity in a stage report (inventory nests them per file)."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    issues = list(report.get("issues", []))
    for f in report.get("files", []):
        issues.extend(f.get("issues", []))
    counts = {"errors": 0, "warnings": 0, "info": 0}
    for issue in issues:
        key = {"error": "errors", "warning": "warnings", "info": "info"}.get(issue.get("severity"))
        if key:
            counts[key] += 1
    return counts


def run_gate(raw_dir: Path, config_path: Path, output_dir: Path, skip_transactions: bool = False) -> list[dict]:
    """Run the stages in order and return one result record per stage."""
    results: list[dict] = []
    stop_reason: str | None = None
    for name, stage_main, report_name in STAGES:
        report_path = output_dir / report_name
        record = {"name": name, "report": str(report_path)}
        if stop_reason:
            record.update(status="skipped", exit_code=None, seconds=0.0, note=stop_reason)
            results.append(record)
            continue
        if skip_transactions and name == "transactions_profile":
            record.update(status="skipped", exit_code=None, seconds=0.0, note="skipped by --skip-transactions")
            results.append(record)
            continue

        print(f"\n===== {name} =====")
        start = time.perf_counter()
        exit_code = stage_main(["--raw-dir", str(raw_dir), "--config", str(config_path), "--output", str(report_path)])
        record.update(
            status=STATUS_BY_EXIT_CODE.get(exit_code, "error"),
            exit_code=exit_code,
            seconds=round(time.perf_counter() - start, 2),
        )
        # Only trust a report written by this run.
        record.update(counts=count_issues(report_path) if exit_code in (0, 1) else None)
        results.append(record)
        if name == "inventory" and exit_code != 0:
            stop_reason = "inventory did not pass; later stages were not run"
    return results


def overall_exit_code(results: list[dict]) -> int:
    codes = [r["exit_code"] for r in results if r["exit_code"] is not None]
    if 2 in codes:
        return 2
    return 1 if 1 in codes else 0


def format_summary(results: list[dict], exit_code: int) -> str:
    lines = ["", "===== validation summary ====="]
    for r in results:
        counts = r.get("counts")
        detail = f"errors={counts['errors']} warnings={counts['warnings']} info={counts['info']}" if counts else r.get("note", "")
        lines.append(f"{r['name']:22} {r['status'].upper():8} {r['seconds']:>7}s  {detail}")
    lines.append({0: "OVERALL: PASSED", 1: "OVERALL: FAILED (blocking data errors)", 2: "OVERALL: ERROR (a stage could not run)"}[exit_code])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run all raw-data validation stages and write one summary.")
    parser.add_argument("--raw-dir", type=Path, help="Override raw_data_dir from configs/local.yaml.")
    parser.add_argument("--config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-transactions", action="store_true", help="Skip the heavy transactions profile (not a full validation).")
    parser.add_argument("--sha256", action="store_true", help="Also record SHA-256 hashes of the data files (slow for the 3.5 GB file).")
    args = parser.parse_args(argv)

    try:
        raw_dir = args.raw_dir if args.raw_dir else resolve_raw_data_dir()
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"raw data directory does not exist: {raw_dir}")
        config = load_yaml(args.config)
        file_names = list(config["files"])
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Configuration problem: {exc}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = run_gate(raw_dir, args.config, args.output_dir, args.skip_transactions)
    exit_code = overall_exit_code(results)

    data_files = {}
    for name in file_names:
        path = raw_dir / name
        if path.is_file():
            data_files[name] = file_fingerprint(path, sha256=args.sha256)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_data_dir": str(raw_dir),
        "data_config": str(args.config),
        "code_revision": code_revision(),
        "full_validation": not args.skip_transactions,
        "overall": {0: "passed", 1: "failed", 2: "error"}[exit_code],
        "blocking_errors": exit_code != 0,
        "exit_code": exit_code,
        "data_files": data_files,
        "stages": results,
    }
    summary_path = args.output_dir / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(format_summary(results, exit_code))
    print(f"Summary written to: {summary_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
