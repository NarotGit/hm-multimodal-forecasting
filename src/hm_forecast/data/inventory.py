"""File-level inventory of the raw H&M CSV files.

This is the first data-quality boundary. It reads only file metadata and the CSV
header line, so it is fast and never loads the large transactions table.

Policy
------
ERROR   (blocks later stages):  required file missing, not a file, empty,
        unreadable header, duplicate header names, required column missing.
WARNING (reported, not blocking): optional file missing, unexpected extra column,
        file size far from the documented approximate size.

Nothing is corrected silently: every finding is reported.

Usage:  python -m hm_forecast.data.inventory
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from hm_forecast.config import (
    DEFAULT_DATA_CONFIG,
    REPO_ROOT,
    load_yaml,
    resolve_raw_data_dir,
)

ERROR = "error"
WARNING = "warning"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "validation" / "data_inventory.json"


@dataclass
class Issue:
    severity: str
    code: str
    message: str


@dataclass
class FileInventory:
    name: str
    required: bool
    path: str
    exists: bool = False
    size_bytes: int | None = None
    columns: list[str] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == ERROR for i in self.issues)


def read_csv_header(path: Path) -> list[str]:
    """Return the header row of a CSV file (tolerates a UTF-8 byte-order mark)."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        try:
            return next(csv.reader(f))
        except StopIteration:
            raise ValueError("file has no header line") from None


def inventory_file(raw_dir: Path, name: str, spec: dict, size_tolerance_ratio: float) -> FileInventory:
    """Check one expected file against its specification."""
    path = raw_dir / name
    required = bool(spec.get("required", False))
    inv = FileInventory(name=name, required=required, path=str(path))

    if not path.exists():
        severity = ERROR if required else WARNING
        kind = "required" if required else "optional"
        inv.issues.append(Issue(severity, "FILE_MISSING", f"{kind} file not found: {path}"))
        return inv
    if not path.is_file():
        inv.issues.append(Issue(ERROR, "NOT_A_FILE", f"path exists but is not a file: {path}"))
        return inv

    inv.exists = True
    inv.size_bytes = path.stat().st_size
    if inv.size_bytes == 0:
        inv.issues.append(Issue(ERROR, "EMPTY_FILE", f"file is empty: {path}"))
        return inv

    approx_mb = spec.get("approx_size_mb")
    if approx_mb:
        observed_mb = inv.size_bytes / 1_000_000
        low, high = approx_mb / size_tolerance_ratio, approx_mb * size_tolerance_ratio
        if not (low <= observed_mb <= high):
            inv.issues.append(
                Issue(
                    WARNING,
                    "SIZE_UNEXPECTED",
                    f"{observed_mb:,.1f} MB is outside the plausible range "
                    f"[{low:,.1f}, {high:,.1f}] MB around the expected ~{approx_mb} MB",
                )
            )

    try:
        inv.columns = read_csv_header(path)
    except (OSError, UnicodeDecodeError, ValueError, csv.Error) as exc:
        inv.issues.append(Issue(ERROR, "HEADER_UNREADABLE", f"cannot read CSV header of {path}: {exc}"))
        return inv

    duplicates = sorted({c for c in inv.columns if inv.columns.count(c) > 1})
    if duplicates:
        inv.issues.append(Issue(ERROR, "DUPLICATE_COLUMNS", f"duplicate header names: {duplicates}"))

    expected = list(spec.get("columns", {}))
    inv.missing_columns = [c for c in expected if c not in inv.columns]
    inv.unexpected_columns = [c for c in inv.columns if c not in expected]
    if inv.missing_columns:
        inv.issues.append(Issue(ERROR, "COLUMNS_MISSING", f"missing expected columns: {inv.missing_columns}"))
    if inv.unexpected_columns:
        inv.issues.append(Issue(WARNING, "COLUMNS_UNEXPECTED", f"unexpected columns: {inv.unexpected_columns}"))
    return inv


def build_inventory(raw_dir: Path, data_config: dict) -> list[FileInventory]:
    """Inventory every file listed in the data configuration."""
    ratio = float(data_config.get("size_tolerance_ratio", 2.0))
    return [
        inventory_file(raw_dir, name, spec, ratio)
        for name, spec in data_config["files"].items()
    ]


def has_blocking_errors(inventories: list[FileInventory]) -> bool:
    return any(inv.has_errors for inv in inventories)


def format_report(inventories: list[FileInventory]) -> str:
    lines = []
    for inv in inventories:
        size = f"{inv.size_bytes / 1_000_000:,.1f} MB" if inv.size_bytes is not None else "n/a"
        status = "ERROR" if inv.has_errors else ("WARN" if inv.issues else "OK")
        req = "required" if inv.required else "optional"
        lines.append(f"[{status:5}] {inv.name} ({req}) size={size} columns={len(inv.columns)}")
        for issue in inv.issues:
            lines.append(f"          {issue.severity.upper()} {issue.code}: {issue.message}")
    verdict = "BLOCKING ERRORS FOUND" if has_blocking_errors(inventories) else "no blocking errors"
    lines.append(f"Result: {verdict}")
    return "\n".join(lines)


def write_report(inventories: list[FileInventory], raw_dir: Path, config_path: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_data_dir": str(raw_dir),
        "data_config": str(config_path),
        "blocking_errors": has_blocking_errors(inventories),
        "files": [asdict(inv) for inv in inventories],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory the raw H&M CSV files (header and size only).")
    parser.add_argument("--raw-dir", type=Path, help="Override raw_data_dir from configs/local.yaml.")
    parser.add_argument("--config", type=Path, default=DEFAULT_DATA_CONFIG, help="Data specification YAML.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to write the JSON report.")
    args = parser.parse_args(argv)

    try:
        raw_dir = args.raw_dir if args.raw_dir else resolve_raw_data_dir()
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"raw data directory does not exist: {raw_dir}")
        data_config = load_yaml(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Configuration problem: {exc}", file=sys.stderr)
        return 2

    inventories = build_inventory(raw_dir, data_config)
    print(f"Raw data directory: {raw_dir}")
    print(format_report(inventories))
    write_report(inventories, raw_dir, args.config, args.output)
    print(f"Report written to: {args.output}")
    return 1 if has_blocking_errors(inventories) else 0


if __name__ == "__main__":
    sys.exit(main())
