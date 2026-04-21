#!/usr/bin/env python3
"""
Write a cleaned protocol file containing only rows whose audio paths exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def normalize_delimiter(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if value == " " or value.strip() == "":
        return None
    return value


def split_line(line: str, delimiter: Optional[str]) -> list[str]:
    return line.split(delimiter) if delimiter is not None else line.split()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-path", required=True, help="Dataset root directory")
    parser.add_argument("--protocol-path", required=True, help="Source protocol file")
    parser.add_argument("--output-path", required=True, help="Output cleaned protocol file")
    parser.add_argument("--delimiter", default=" ", help='Protocol delimiter, e.g. " ", "," or "\\t"')
    parser.add_argument("--key-col", type=int, default=0, help="Column index containing the relative audio path")
    parser.add_argument(
        "--keep-comments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve blank lines and comments in the cleaned output",
    )
    return parser.parse_args()


def filter_protocol(
    database_path: Path,
    protocol_path: Path,
    output_path: Path,
    delimiter: Optional[str],
    key_col: int,
    keep_comments: bool,
):
    checked_entries = 0
    kept_entries = 0
    removed_entries = 0
    bad_lines = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with protocol_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, raw in enumerate(src, start=1):
            stripped = raw.strip()

            if not stripped:
                if keep_comments:
                    dst.write(raw)
                continue

            if stripped.startswith("#"):
                if keep_comments:
                    dst.write(raw)
                continue

            parts = split_line(stripped, delimiter)
            if key_col >= len(parts):
                bad_lines += 1
                continue

            checked_entries += 1
            rel_path = parts[key_col]
            full_path = database_path / rel_path
            if full_path.exists():
                dst.write(raw)
                kept_entries += 1
            else:
                removed_entries += 1

    return {
        "protocol_path": str(protocol_path),
        "output_path": str(output_path),
        "database_path": str(database_path),
        "checked_entries": checked_entries,
        "kept_entries": kept_entries,
        "removed_entries": removed_entries,
        "bad_lines": bad_lines,
    }


def main():
    args = parse_args()
    summary = filter_protocol(
        database_path=Path(args.database_path),
        protocol_path=Path(args.protocol_path),
        output_path=Path(args.output_path),
        delimiter=normalize_delimiter(args.delimiter),
        key_col=args.key_col,
        keep_comments=args.keep_comments,
    )

    print(f"Protocol: {summary['protocol_path']}")
    print(f"Output: {summary['output_path']}")
    print(f"Dataset root: {summary['database_path']}")
    print(f"Checked entries: {summary['checked_entries']}")
    print(f"Kept entries: {summary['kept_entries']}")
    print(f"Removed entries: {summary['removed_entries']}")
    print(f"Bad lines: {summary['bad_lines']}")


if __name__ == "__main__":
    main()
