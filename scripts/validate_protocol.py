#!/usr/bin/env python3
"""Remove protocol rows whose referenced audio files do not exist."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MissingEntry:
    line_number: int
    protocol_value: str
    resolved_path: str
    label: str | None


@dataclass
class ProtocolSummary:
    protocol_path: str
    cleaned_protocol_path: str
    total_rows: int
    kept_rows: int
    removed_rows: int
    missing_entries: list[MissingEntry]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan protocol files, report rows whose audio path is missing, "
            "and write cleaned protocol copies with those rows removed."
        )
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        required=True,
        help="Base directory containing the referenced audio files.",
    )
    parser.add_argument(
        "--protocol-path",
        type=Path,
        nargs="+",
        required=True,
        help="One or more protocol files to validate.",
    )
    parser.add_argument(
        "--delimiter",
        default=None,
        help="Column delimiter. Defaults to any whitespace.",
    )
    parser.add_argument(
        "--src-col",
        type=int,
        default=None,
        help="Column index containing a relative or absolute audio path.",
    )
    parser.add_argument(
        "--key-col",
        type=int,
        default=None,
        help="Column index containing an utterance key to map to <database>/<key><extension>.",
    )
    parser.add_argument(
        "--label-col",
        type=int,
        default=None,
        help="Optional label column to include in the missing-entry report.",
    )
    parser.add_argument(
        "--extension",
        default=".wav",
        help="Extension used with --key-col mode. Default: .wav",
    )
    parser.add_argument(
        "--cleaned-dir",
        type=Path,
        default=Path("cleaned_protocols"),
        help="Directory where cleaned protocol files are written.",
    )
    parser.add_argument(
        "--max-reported-missing",
        type=int,
        default=200,
        help="Limit the number of missing entries printed per protocol. Default: 200",
    )
    parser.add_argument(
        "--keep-comments",
        action="store_true",
        help="Preserve blank lines and comment lines starting with '#'.",
    )
    args = parser.parse_args()

    if (args.src_col is None) == (args.key_col is None):
        parser.error("Provide exactly one of --src-col or --key-col.")

    return args


def split_fields(line: str, delimiter: str | None) -> list[str]:
    stripped = line.rstrip("\n")
    if delimiter is None:
        return stripped.split()
    return stripped.split(delimiter)


def join_fields(fields: list[str], delimiter: str | None) -> str:
    if delimiter is None:
        return " ".join(fields)
    return delimiter.join(fields)


def resolve_audio_path(
    fields: list[str],
    database_path: Path,
    src_col: int | None,
    key_col: int | None,
    extension: str,
) -> tuple[str, Path]:
    if src_col is not None:
        protocol_value = fields[src_col]
        candidate = Path(protocol_value)
        if candidate.is_absolute():
            return protocol_value, candidate
        return protocol_value, database_path / candidate

    assert key_col is not None
    protocol_value = fields[key_col]
    return protocol_value, database_path / f"{protocol_value}{extension}"


def cleaned_output_path(cleaned_dir: Path, protocol_path: Path) -> Path:
    suffix = "".join(protocol_path.suffixes)
    stem = protocol_path.name[: -len(suffix)] if suffix else protocol_path.name
    cleaned_name = f"{stem}.cleaned{suffix}" if suffix else f"{stem}.cleaned"
    return cleaned_dir / cleaned_name


def validate_protocol(
    protocol_path: Path,
    database_path: Path,
    delimiter: str | None,
    src_col: int | None,
    key_col: int | None,
    label_col: int | None,
    extension: str,
    keep_comments: bool,
    cleaned_dir: Path,
) -> ProtocolSummary:
    missing_entries: list[MissingEntry] = []
    kept_lines: list[str] = []
    total_rows = 0

    with protocol_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()

            if not stripped:
                if keep_comments:
                    kept_lines.append(raw_line.rstrip("\n"))
                continue

            if stripped.startswith("#"):
                if keep_comments:
                    kept_lines.append(raw_line.rstrip("\n"))
                continue

            fields = split_fields(raw_line, delimiter)
            total_rows += 1

            try:
                protocol_value, resolved_path = resolve_audio_path(
                    fields=fields,
                    database_path=database_path,
                    src_col=src_col,
                    key_col=key_col,
                    extension=extension,
                )
            except IndexError as exc:
                raise ValueError(
                    f"{protocol_path}: line {line_number} does not contain the requested column."
                ) from exc

            if resolved_path.exists():
                kept_lines.append(join_fields(fields, delimiter))
                continue

            label = None
            if label_col is not None and label_col < len(fields):
                label = fields[label_col]

            missing_entries.append(
                MissingEntry(
                    line_number=line_number,
                    protocol_value=protocol_value,
                    resolved_path=str(resolved_path),
                    label=label,
                )
            )

    cleaned_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = cleaned_output_path(cleaned_dir, protocol_path)
    with cleaned_path.open("w", encoding="utf-8") as handle:
        for line in kept_lines:
            handle.write(f"{line}\n")

    return ProtocolSummary(
        protocol_path=str(protocol_path),
        cleaned_protocol_path=str(cleaned_path),
        total_rows=total_rows,
        kept_rows=len(kept_lines),
        removed_rows=len(missing_entries),
        missing_entries=missing_entries,
    )


def print_summary(summary: ProtocolSummary, max_reported_missing: int) -> None:
    print(f"Protocol: {summary.protocol_path}")
    print(f"Cleaned output: {summary.cleaned_protocol_path}")
    print(f"Rows checked: {summary.total_rows}")
    print(f"Rows kept: {summary.kept_rows}")
    print(f"Rows removed: {summary.removed_rows}")

    if not summary.missing_entries:
        print("Missing paths: none")
        print()
        return

    print("Missing paths:")
    for entry in summary.missing_entries[:max_reported_missing]:
        label_suffix = f", label={entry.label}" if entry.label is not None else ""
        print(
            f"  line {entry.line_number}: {entry.resolved_path} "
            f"(value={entry.protocol_value}{label_suffix})"
        )

    hidden_count = len(summary.missing_entries) - max_reported_missing
    if hidden_count > 0:
        print(f"  ... {hidden_count} more missing entries omitted from console output")
    print()


def main() -> None:
    args = parse_args()
    database_path = args.database_path.expanduser().resolve()

    if not database_path.exists():
        raise FileNotFoundError(f"Database path does not exist: {database_path}")

    for protocol_path in args.protocol_path:
        summary = validate_protocol(
            protocol_path=protocol_path.expanduser().resolve(),
            database_path=database_path,
            delimiter=args.delimiter,
            src_col=args.src_col,
            key_col=args.key_col,
            label_col=args.label_col,
            extension=args.extension,
            keep_comments=args.keep_comments,
            cleaned_dir=args.cleaned_dir.expanduser().resolve(),
        )
        print_summary(summary, args.max_reported_missing)


if __name__ == "__main__":
    main()
