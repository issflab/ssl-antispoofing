#!/usr/bin/env python3
"""
Create train/dev/eval protocol files for Codecfake_plus and vocoder data.

Design notes and assumptions:
- The script inspects the source metadata files at runtime and prints a few sample
  rows plus the inferred delimiter/path/speaker/label columns before writing
  outputs.
- Source column structure is preserved as much as possible. When a source file
  already contains an utterance/file column, that column is replaced with the
  required relative path and all other columns are kept in their original order.
- `CoSG_labels.txt` is always treated as evaluation data.
- For vocoder matching, `train.lst` / `dev.lst` entries and `protocol.txt`
  entries are normalized using basename-plus-stem matching so IDs such as
  `foo`, `foo.wav`, and `path/to/foo.wav` resolve consistently.
- Final output protocol files are intentionally standardized to a
  simple 3-column layout: `<relative_path> <label> <source>`. This keeps mixed
  datasets compatible with the current training config.
- Codecfake_plus and vocoder dataset-specific protocols can be regenerated or
  reused from existing files, depending on CLI flags.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


CODECFAKE_ROOT = Path("/data/Data/Codecfake_plus")
VOCODER_ROOT = Path("/data/Data/vocoder_data")
MLAAD_META = Path("/data/Data/MLAAD/fake/en/combined_meta.txt")
MAILABS_US_ROOT = Path("/data/Data/MAILabs/en_US/by_book")
MAILABS_UK_ROOT = Path("/data/Data/MAILabs/en_UK/by_book")
DEFAULT_OUTPUT_DIR = Path("/data/Data/protocols")
CODECFAKE_OUTPUT_DIR = CODECFAKE_ROOT
VOCODER_OUTPUT_DIR = VOCODER_ROOT / "voc_v4"

VALIDATION_SPEAKERS = {"p226", "p229"}
EVAL_CORS_SPEAKERS = {"p227", "p228"}
EXCLUDED_TRAIN_SPEAKERS = VALIDATION_SPEAKERS | EVAL_CORS_SPEAKERS
LABEL_VALUES = {"bonafide", "spoof"}
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3")
SAMPLE_ROWS_TO_PRINT = 5
MLAAD_DEV_MODELS = {
    "parler_tts_large_v1",
    "parler_tts_mini_v0.1",
    "parler_tts_mini_v1",
    "bark",
    "bark-small",
    "llasa-1b",
    "llasa-1b-multilingual",
    "llasa-3b",
    "llasa-8b",
    "mars5",
    "e2-tts",
}
MAILABS_DEV_MODULO = 5


class ProtocolError(RuntimeError):
    """Raised when a protocol file cannot be parsed safely."""


@dataclass(frozen=True)
class SourceInfo:
    name: str
    path: Path
    root_dir: Path
    dataset_prefix: str
    audio_subdir: str


@dataclass
class ParsedRow:
    source_name: str
    source_path: Path
    raw_line: str
    columns: list[str]
    path_idx: int
    label_idx: int | None
    speaker_idx: int | None
    utterance_id: str
    speaker: str | None
    label: str | None
    relative_path: str
    source: str

    def rendered(self) -> str:
        if self.label is None:
            raise ProtocolError(f"Missing label while rendering row from {self.source_name}: {self.raw_line}")
        return f"{self.relative_path} {self.label} {self.source}"

    def rendered_with_original_columns(self) -> str:
        columns = list(self.columns)
        columns[self.path_idx] = self.relative_path
        return " ".join(columns)


@dataclass(frozen=True)
class CombinedRow:
    relative_path: str
    label: str
    source: str

    def rendered(self) -> str:
        return f"{self.relative_path} {self.label} {self.source}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codecfake-root", type=Path, default=CODECFAKE_ROOT)
    parser.add_argument("--vocoder-root", type=Path, default=VOCODER_ROOT)
    parser.add_argument("--mlaad-meta", type=Path, default=MLAAD_META)
    parser.add_argument("--mailabs-us-root", type=Path, default=MAILABS_US_ROOT)
    parser.add_argument("--mailabs-uk-root", type=Path, default=MAILABS_UK_ROOT)
    parser.add_argument("--codecfake-output-dir", type=Path, default=CODECFAKE_OUTPUT_DIR)
    parser.add_argument("--vocoder-output-dir", type=Path, default=VOCODER_OUTPUT_DIR)
    parser.add_argument("--combined-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--recreate-codecfake-protocols",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate codecfake_plus_train/dev/eval.txt instead of reusing existing files.",
    )
    parser.add_argument(
        "--recreate-vocoder-protocols",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate vocoder_train/dev.txt instead of reusing existing files.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise ProtocolError(f"Missing {description}: {path}")


def detect_delimiter(lines: Sequence[str]) -> str | None:
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return None
    if all("\t" in line for line in non_empty[: min(10, len(non_empty))]):
        return "\t"
    if all("," in line for line in non_empty[: min(10, len(non_empty))]):
        return ","
    return None


def split_line(line: str, delimiter: str | None) -> list[str]:
    return line.split(delimiter) if delimiter else line.split()


def read_nonempty_lines(path: Path) -> list[str]:
    ensure_exists(path, "source file")
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def normalize_label(token: str) -> str | None:
    lowered = token.strip().lower()
    return lowered if lowered in LABEL_VALUES else None


def normalize_speaker(token: str) -> str | None:
    cleaned = token.strip()
    return cleaned if re.fullmatch(r"p\d{3}", cleaned) else None


def looks_like_audio_token(token: str) -> bool:
    candidate = token.strip()
    lowered = candidate.lower()
    if "/" in candidate:
        return True
    if any(lowered.endswith(ext) for ext in AUDIO_EXTENSIONS):
        return True
    return bool(re.search(r"[A-Za-z]", candidate) and re.search(r"\d", candidate) and "_" in candidate)


def token_to_filename(token: str) -> str:
    candidate = Path(token.strip()).name
    if Path(candidate).suffix.lower() in AUDIO_EXTENSIONS:
        return candidate
    return f"{candidate}.wav"


def build_relative_path(dataset_prefix: str, audio_subdir: str, token: str) -> str:
    filename = token_to_filename(token)
    return f"{dataset_prefix}/{audio_subdir}/{filename}"


def normalize_relative_path(path: str) -> str:
    return path.strip().lstrip("./")


def normalize_source_value(label: str | None, source: str | None) -> str:
    normalized_label = normalize_label(label or "")
    if normalized_label == "bonafide":
        return "bonafide"
    if source is None:
        raise ProtocolError("Spoof row is missing a source/system name.")
    source_value = source.strip()
    if not source_value:
        raise ProtocolError("Spoof row has an empty source/system name.")
    return source_value


def extract_codecfake_source_from_token(token: str, label: str | None) -> str:
    if normalize_label(label or "") == "bonafide":
        return "bonafide"
    stem = Path(token.strip()).stem
    match = re.match(r"^p\d{3}_\d+_(.+)$", stem)
    if match:
        return match.group(1)
    return stem


def infer_row_source(source_name: str, columns: Sequence[str], path_idx: int, label_idx: int | None, label: str | None) -> str:
    normalized = normalize_label(label or "")
    if normalized == "bonafide":
        return "bonafide"

    lowered_name = source_name.lower()
    if "codecfake_plus cogs" in lowered_name or "cosg" in lowered_name:
        return normalize_source_value(label, columns[0] if columns else None)
    if "codecfake" in lowered_name:
        return normalize_source_value(label, extract_codecfake_source_from_token(columns[path_idx], label))
    if "vocoder" in lowered_name:
        if len(columns) > 3:
            return normalize_source_value(label, columns[3])
        return normalize_source_value(label, columns[0] if columns else None)

    if len(columns) == 3 and path_idx == 0 and label_idx == 1:
        return normalize_source_value(label, columns[2])

    candidates = [
        token
        for idx, token in enumerate(columns)
        if idx not in {path_idx, label_idx} and token.strip() and token.strip() != "-"
    ]
    return normalize_source_value(label, candidates[-1] if candidates else None)


def resolve_audio_path(root_dir: Path, audio_subdir: str, token: str) -> Path:
    return root_dir / audio_subdir / token_to_filename(token)


def infer_label_idx(rows: Sequence[list[str]]) -> int | None:
    if not rows:
        return None
    width = max(len(row) for row in rows)
    scores: list[tuple[int, int]] = []
    for idx in range(width):
        labels = 0
        total = 0
        for row in rows:
            if idx >= len(row):
                continue
            total += 1
            if normalize_label(row[idx]) is not None:
                labels += 1
        if total and labels == total:
            scores.append((idx, labels))
    if scores:
        return scores[-1][0]
    return None


def infer_speaker_idx(rows: Sequence[list[str]]) -> int | None:
    if not rows:
        return None
    width = max(len(row) for row in rows)
    best_idx = None
    best_hits = -1
    for idx in range(width):
        hits = 0
        total = 0
        for row in rows:
            if idx >= len(row):
                continue
            total += 1
            if normalize_speaker(row[idx]) is not None:
                hits += 1
        if total and hits > best_hits:
            best_hits = hits
            best_idx = idx
    if best_hits <= 0:
        return None
    return best_idx


def infer_path_idx(
    rows: Sequence[list[str]],
    *,
    label_idx: int | None,
    speaker_idx: int | None,
    root_dir: Path,
    audio_subdir: str,
) -> int:
    if not rows:
        raise ProtocolError("Cannot infer path column from an empty file.")
    width = max(len(row) for row in rows)
    best_idx = None
    best_score = -1
    for idx in range(width):
        if idx == label_idx:
            continue
        score = 0
        for row in rows:
            if idx >= len(row):
                continue
            token = row[idx]
            if "/" in token:
                score += 4
            if looks_like_audio_token(token):
                score += 2
            if resolve_audio_path(root_dir, audio_subdir, token).is_file():
                score += 5
            if speaker_idx is not None and idx == speaker_idx:
                score -= 2
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx is None:
        raise ProtocolError("Failed to infer the file path / utterance column.")
    return best_idx


def inspect_source(source: SourceInfo) -> tuple[str | None, int, int | None, int | None, list[list[str]]]:
    lines = read_nonempty_lines(source.path)
    delimiter = detect_delimiter(lines[: min(50, len(lines))])
    sample_rows = [split_line(line, delimiter) for line in lines[: min(50, len(lines))]]
    label_idx = infer_label_idx(sample_rows)
    speaker_idx = infer_speaker_idx(sample_rows)
    path_idx = infer_path_idx(
        sample_rows,
        label_idx=label_idx,
        speaker_idx=speaker_idx,
        root_dir=source.root_dir,
        audio_subdir=source.audio_subdir,
    )

    print(f"\nInspecting {source.name}: {source.path}")
    print(f"  delimiter: {'whitespace' if delimiter is None else repr(delimiter)}")
    print(f"  inferred path column: {path_idx}")
    print(f"  inferred speaker column: {speaker_idx if speaker_idx is not None else 'not found'}")
    print(f"  inferred label column: {label_idx if label_idx is not None else 'not found'}")
    print("  sample rows:")
    for row in sample_rows[:SAMPLE_ROWS_TO_PRINT]:
        print(f"    {row}")

    return delimiter, path_idx, speaker_idx, label_idx, sample_rows


def parse_rows(source: SourceInfo) -> list[ParsedRow]:
    delimiter, path_idx, speaker_idx, label_idx, _ = inspect_source(source)
    rows: list[ParsedRow] = []
    for raw_line in read_nonempty_lines(source.path):
        columns = split_line(raw_line, delimiter)
        if path_idx >= len(columns):
            raise ProtocolError(
                f"{source.path} has a row with too few columns for inferred path index {path_idx}: {raw_line}"
            )
        speaker = None
        if speaker_idx is not None and speaker_idx < len(columns):
            speaker = normalize_speaker(columns[speaker_idx])
        if speaker is None:
            speaker = infer_speaker_from_row(columns)
        label = None
        if label_idx is not None and label_idx < len(columns):
            label = normalize_label(columns[label_idx])
        utterance_id = normalize_match_key(columns[path_idx])
        relative_path = normalize_relative_path(build_relative_path(source.dataset_prefix, source.audio_subdir, columns[path_idx]))
        rows.append(
            ParsedRow(
                source_name=source.name,
                source_path=source.path,
                raw_line=raw_line,
                columns=columns,
                path_idx=path_idx,
                label_idx=label_idx,
                speaker_idx=speaker_idx,
                utterance_id=utterance_id,
                speaker=speaker,
                label=label,
                relative_path=relative_path,
                source=infer_row_source(source.name, columns, path_idx, label_idx, label),
            )
        )
    return rows


def infer_speaker_from_row(columns: Sequence[str]) -> str | None:
    for token in columns:
        speaker = normalize_speaker(token)
        if speaker is not None:
            return speaker
        match = re.search(r"\b(p\d{3})\b", token)
        if match:
            return match.group(1)
    return None


def normalize_match_key(value: str) -> str:
    text = Path(value.strip()).name
    stem = Path(text).stem
    return stem.lower()


def require_labels(rows: Iterable[ParsedRow], source_name: str) -> None:
    missing = [row.raw_line for row in rows if row.label is None]
    if missing:
        preview = missing[:3]
        raise ProtocolError(f"{source_name} is missing bonafide/spoof labels for rows such as: {preview}")


def write_protocol(path: Path, rows: Sequence[ParsedRow], *, preserve_original_columns: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            rendered = row.rendered_with_original_columns() if preserve_original_columns else row.rendered()
            handle.write(rendered)
            handle.write("\n")
    print(f"Wrote {len(rows):>7} rows -> {path}")


def unique_rows(rows: Sequence[ParsedRow], *, context: str) -> list[ParsedRow]:
    seen_paths: dict[str, ParsedRow] = {}
    deduped: list[ParsedRow] = []
    duplicates = 0
    for row in rows:
        key = row.relative_path
        existing = seen_paths.get(key)
        if existing is None:
            seen_paths[key] = row
            deduped.append(row)
            continue
        duplicates += 1
        if existing.rendered() != row.rendered():
            print(
                f"Warning: duplicate path with differing metadata in {context}: "
                f"{row.relative_path}. Keeping the first occurrence.",
                file=sys.stderr,
            )
    if duplicates:
        print(f"Removed {duplicates} duplicate entries while building {context}.")
    return deduped


def unique_combined_rows(rows: Sequence[CombinedRow], *, context: str) -> list[CombinedRow]:
    seen: dict[str, CombinedRow] = {}
    deduped: list[CombinedRow] = []
    duplicates = 0
    for row in rows:
        existing = seen.get(row.relative_path)
        if existing is None:
            seen[row.relative_path] = row
            deduped.append(row)
            continue
        duplicates += 1
        if existing != row:
            print(
                f"Warning: duplicate combined path with differing metadata in {context}: "
                f"{row.relative_path}. Keeping the first occurrence.",
                file=sys.stderr,
            )
    if duplicates:
        print(f"Removed {duplicates} duplicate entries while building {context}.")
    return deduped


def write_combined_protocol(path: Path, rows: Sequence[CombinedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.rendered())
            handle.write("\n")
    print(f"Wrote {len(rows):>7} rows -> {path}")


def make_combined_row(src: str, relative_path: str, label: str) -> CombinedRow:
    normalized_label = normalize_label(label)
    if normalized_label is None:
        raise ProtocolError(f"Unsupported label for combined protocol row: {label}")
    return CombinedRow(
        relative_path=normalize_relative_path(relative_path),
        label=normalized_label,
        source=normalize_source_value(normalized_label, src),
    )


def convert_protocol_rows_for_combined(rows: Sequence[ParsedRow], default_src: str) -> list[CombinedRow]:
    combined: list[CombinedRow] = []
    for row in rows:
        if row.label is None:
            raise ProtocolError(f"Missing label while converting row from {row.source_name}: {row.raw_line}")
        source = row.source if row.source else default_src
        combined.append(
            make_combined_row(
                src=source,
                relative_path=normalize_relative_path(row.relative_path),
                label=row.label,
            )
        )
    return combined


def load_existing_protocol_rows(path: Path, source_name: str) -> list[ParsedRow]:
    source = SourceInfo(
        name=source_name,
        path=path,
        root_dir=path.parent,
        dataset_prefix="",
        audio_subdir="",
    )
    delimiter, path_idx, speaker_idx, label_idx, sample_rows = inspect_source(source)
    rows: list[ParsedRow] = []
    for raw_line in read_nonempty_lines(path):
        columns = split_line(raw_line, delimiter)
        if path_idx >= len(columns):
            raise ProtocolError(f"{path} has a row with too few columns for inferred path index {path_idx}: {raw_line}")
        speaker = None
        if speaker_idx is not None and speaker_idx < len(columns):
            speaker = normalize_speaker(columns[speaker_idx])
        if speaker is None:
            speaker = infer_speaker_from_row(columns)
        label = None
        if label_idx is not None and label_idx < len(columns):
            label = normalize_label(columns[label_idx])
        relative_path = normalize_relative_path(columns[path_idx])
        rows.append(
            ParsedRow(
                source_name=source_name,
                source_path=path,
                raw_line=raw_line,
                columns=columns,
                path_idx=path_idx,
                label_idx=label_idx,
                speaker_idx=speaker_idx,
                utterance_id=normalize_match_key(columns[path_idx]),
                speaker=speaker,
                label=label,
                relative_path=relative_path,
                source=infer_row_source(source_name, columns, path_idx, label_idx, label),
            )
        )
    require_labels(rows, source_name)
    return rows


def create_codecfake_protocols(codecfake_root: Path, output_dir: Path) -> dict[str, list[ParsedRow]]:
    cors_source = SourceInfo(
        name="Codecfake_plus CoRS",
        path=codecfake_root / "CoRS_labels.txt",
        root_dir=codecfake_root,
        dataset_prefix="Codecfake_plus",
        audio_subdir="CoRS",
    )
    cosg_source = SourceInfo(
        name="Codecfake_plus CoSG",
        path=codecfake_root / "CoSG_labels.txt",
        root_dir=codecfake_root,
        dataset_prefix="Codecfake_plus",
        audio_subdir="CoSG",
    )

    cors_rows = parse_rows(cors_source)
    cosg_rows = parse_rows(cosg_source)
    require_labels(cors_rows, cors_source.name)
    require_labels(cosg_rows, cosg_source.name)

    train_rows = [row for row in cors_rows if row.speaker not in EXCLUDED_TRAIN_SPEAKERS]
    dev_rows = [row for row in cors_rows if row.speaker in VALIDATION_SPEAKERS]
    eval_cors_rows = [row for row in cors_rows if row.speaker in EVAL_CORS_SPEAKERS]

    # CoSG is always evaluation data.
    eval_rows = eval_cors_rows + list(cosg_rows)

    train_rows = unique_rows(train_rows, context="codecfake_plus_train.txt")
    dev_rows = unique_rows(dev_rows, context="codecfake_plus_dev.txt")
    eval_rows = unique_rows(eval_rows, context="codecfake_plus_eval.txt")

    outputs = {
        "codecfake_plus_train.txt": train_rows,
        "codecfake_plus_dev.txt": dev_rows,
        "codecfake_plus_eval.txt": eval_rows,
    }
    for filename, rows in outputs.items():
        write_protocol(output_dir / filename, rows, preserve_original_columns=True)
    return outputs


def read_list_ids(path: Path) -> list[str]:
    lines = read_nonempty_lines(path)
    print(f"\nInspecting list file: {path}")
    for line in lines[:SAMPLE_ROWS_TO_PRINT]:
        print(f"  sample: {line}")
    return [normalize_match_key(line) for line in lines]


def build_index(rows: Sequence[ParsedRow]) -> dict[str, ParsedRow]:
    index: dict[str, ParsedRow] = {}
    for row in rows:
        key = normalize_match_key(row.columns[row.path_idx])
        if key not in index:
            index[key] = row
    return index


def select_rows_from_list(index: dict[str, ParsedRow], ids: Sequence[str], list_name: str) -> list[ParsedRow]:
    selected: list[ParsedRow] = []
    missing: list[str] = []
    for item_id in ids:
        row = index.get(item_id)
        if row is None:
            missing.append(item_id)
            continue
        selected.append(row)
    if missing:
        preview = ", ".join(missing[:10])
        raise ProtocolError(f"{list_name} contains {len(missing)} IDs missing from protocol.txt. Examples: {preview}")
    return unique_rows(selected, context=list_name)


def create_vocoder_protocols(vocoder_root: Path, output_dir: Path) -> dict[str, list[ParsedRow]]:
    protocol_path = vocoder_root / "voc_v4" / "protocol.txt"
    train_list_path = vocoder_root / "voc_v4" / "scp" / "train.lst"
    dev_list_path = vocoder_root / "voc_v4" / "scp" / "dev.lst"

    source = SourceInfo(
        name="vocoder_data protocol",
        path=protocol_path,
        root_dir=vocoder_root,
        dataset_prefix="vocoder_data",
        audio_subdir="voc_v4/wav",
    )
    protocol_rows = parse_rows(source)
    require_labels(protocol_rows, source.name)
    index = build_index(protocol_rows)

    train_ids = read_list_ids(train_list_path)
    dev_ids = read_list_ids(dev_list_path)
    train_rows = select_rows_from_list(index, train_ids, "vocoder_train.txt")
    dev_rows = select_rows_from_list(index, dev_ids, "vocoder_dev.txt")

    outputs = {
        "vocoder_train.txt": train_rows,
        "vocoder_dev.txt": dev_rows,
    }
    for filename, rows in outputs.items():
        write_protocol(output_dir / filename, rows)
    return outputs


def mlaad_model_key(model_name: str) -> str:
    return model_name.strip().lower().replace(" ", "_")


def mlaad_model_in_dev(model_name: str) -> bool:
    return mlaad_model_key(model_name) in MLAAD_DEV_MODELS


def path_under_data_root(path: Path) -> str:
    data_root = Path("/data/Data")
    try:
        rel = path.resolve().relative_to(data_root.resolve())
    except ValueError as exc:
        raise ProtocolError(f"Path is not under /data/Data and cannot be made relative: {path}") from exc
    return rel.as_posix()


def inspect_csv_source(path: Path, *, delimiter: str, name: str) -> list[dict[str, str]]:
    ensure_exists(path, name)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = list(reader)
    print(f"\nInspecting {name}: {path}")
    print(f"  delimiter: {repr(delimiter)}")
    print(f"  columns: {reader.fieldnames}")
    print("  sample rows:")
    for row in rows[:SAMPLE_ROWS_TO_PRINT]:
        print(f"    {row}")
    return rows


def create_mlaad_combined_rows(mlaad_meta: Path) -> tuple[list[CombinedRow], list[CombinedRow]]:
    rows = inspect_csv_source(mlaad_meta, delimiter="|", name="MLAAD combined metadata")
    required_fields = {"filename", "absolute_path", "model_name"}
    if not rows:
        raise ProtocolError(f"MLAAD metadata is empty: {mlaad_meta}")
    first_row = rows[0]
    missing_fields = required_fields - set(first_row)
    if missing_fields:
        raise ProtocolError(f"MLAAD metadata missing required columns: {sorted(missing_fields)}")

    train_rows: list[CombinedRow] = []
    dev_rows: list[CombinedRow] = []
    for row in rows:
        audio_path = Path(row["absolute_path"])
        relative_path = normalize_relative_path(path_under_data_root(audio_path))
        combined = make_combined_row(src=row["model_name"].strip() or "MLAAD", relative_path=relative_path, label="spoof")
        if mlaad_model_in_dev(row["model_name"]):
            dev_rows.append(combined)
        else:
            train_rows.append(combined)

    print(f"MLAAD split: train={len(train_rows)} dev={len(dev_rows)}")
    return train_rows, dev_rows


def inspect_mailabs_roots(mailabs_roots: Sequence[Path]) -> list[Path]:
    wav_paths: list[Path] = []
    for root in mailabs_roots:
        ensure_exists(root, "MAILabs root")
        root_wavs = sorted(path for path in root.rglob("*.wav") if path.is_file())
        print(f"\nInspecting MAILabs root: {root}")
        print(f"  wav count: {len(root_wavs)}")
        print("  sample wavs:")
        for wav_path in root_wavs[:SAMPLE_ROWS_TO_PRINT]:
            print(f"    {wav_path}")
        wav_paths.extend(root_wavs)
    return sorted(wav_paths)


def create_mailabs_combined_rows(mailabs_roots: Sequence[Path]) -> tuple[list[CombinedRow], list[CombinedRow]]:
    wav_paths = inspect_mailabs_roots(mailabs_roots)
    train_rows: list[CombinedRow] = []
    dev_rows: list[CombinedRow] = []
    for index, wav_path in enumerate(wav_paths):
        relative_path = path_under_data_root(wav_path)
        relative_path = normalize_relative_path(relative_path)
        combined = make_combined_row(src="bonafide", relative_path=relative_path, label="bonafide")
        if index % MAILABS_DEV_MODULO == 0:
            dev_rows.append(combined)
        else:
            train_rows.append(combined)
    print(f"MAILabs split: train={len(train_rows)} dev={len(dev_rows)}")
    return train_rows, dev_rows


def combine_outputs(
    output_dir: Path,
    codecfake_outputs: dict[str, list[ParsedRow]],
    vocoder_outputs: dict[str, list[ParsedRow]],
    mlaad_splits: tuple[list[CombinedRow], list[CombinedRow]],
    mailabs_splits: tuple[list[CombinedRow], list[CombinedRow]],
) -> None:
    mlaad_train_rows, mlaad_dev_rows = mlaad_splits
    mailabs_train_rows, mailabs_dev_rows = mailabs_splits

    train_rows = unique_combined_rows(
        convert_protocol_rows_for_combined(codecfake_outputs["codecfake_plus_train.txt"], "Codecfake_plus")
        + convert_protocol_rows_for_combined(vocoder_outputs["vocoder_train.txt"], "vocoder_data")
        + mlaad_train_rows
        + mailabs_train_rows,
        context="codecfake_plus_vocoder_train.txt",
    )
    dev_rows = unique_combined_rows(
        convert_protocol_rows_for_combined(codecfake_outputs["codecfake_plus_dev.txt"], "Codecfake_plus")
        + convert_protocol_rows_for_combined(vocoder_outputs["vocoder_dev.txt"], "vocoder_data")
        + mlaad_dev_rows
        + mailabs_dev_rows,
        context="codecfake_plus_vocoder_dev.txt",
    )
    write_combined_protocol(output_dir / "codecfake_plus_vocoder_train.txt", train_rows)
    write_combined_protocol(output_dir / "codecfake_plus_vocoder_dev.txt", dev_rows)


def get_codecfake_protocol_rows(codecfake_root: Path, output_dir: Path, recreate: bool) -> dict[str, list[ParsedRow]]:
    if recreate:
        return create_codecfake_protocols(codecfake_root, output_dir)

    paths = {
        "codecfake_plus_train.txt": output_dir / "codecfake_plus_train.txt",
        "codecfake_plus_dev.txt": output_dir / "codecfake_plus_dev.txt",
    }
    for filename, path in paths.items():
        ensure_exists(path, f"existing {filename}")
    print("\nReusing existing Codecfake_plus protocol files.")
    outputs = {filename: load_existing_protocol_rows(path, f"Existing {filename}") for filename, path in paths.items()}
    outputs["codecfake_plus_eval.txt"] = []
    return outputs


def get_vocoder_protocol_rows(vocoder_root: Path, output_dir: Path, recreate: bool) -> dict[str, list[ParsedRow]]:
    if recreate:
        return create_vocoder_protocols(vocoder_root, output_dir)

    paths = {
        "vocoder_train.txt": output_dir / "vocoder_train.txt",
        "vocoder_dev.txt": output_dir / "vocoder_dev.txt",
    }
    for filename, path in paths.items():
        ensure_exists(path, f"existing {filename}")
    print("\nReusing existing vocoder protocol files.")
    return {filename: load_existing_protocol_rows(path, f"Existing {filename}") for filename, path in paths.items()}


def main() -> int:
    args = parse_args()
    try:
        ensure_exists(args.codecfake_root, "Codecfake_plus root")
        ensure_exists(args.vocoder_root, "vocoder_data root")
        ensure_exists(args.mlaad_meta, "MLAAD metadata file")
        ensure_exists(args.mailabs_us_root, "MAILabs en_US root")
        ensure_exists(args.mailabs_uk_root, "MAILabs en_UK root")
        codecfake_output_dir = args.codecfake_output_dir
        vocoder_output_dir = args.vocoder_output_dir
        combined_output_dir = args.combined_output_dir

        codecfake_output_dir.mkdir(parents=True, exist_ok=True)
        vocoder_output_dir.mkdir(parents=True, exist_ok=True)
        combined_output_dir.mkdir(parents=True, exist_ok=True)

        codecfake_outputs = get_codecfake_protocol_rows(
            args.codecfake_root,
            codecfake_output_dir,
            recreate=args.recreate_codecfake_protocols,
        )
        vocoder_outputs = get_vocoder_protocol_rows(
            args.vocoder_root,
            vocoder_output_dir,
            recreate=args.recreate_vocoder_protocols,
        )
        mlaad_splits = create_mlaad_combined_rows(args.mlaad_meta)
        mailabs_splits = create_mailabs_combined_rows([args.mailabs_us_root, args.mailabs_uk_root])
        combine_outputs(combined_output_dir, codecfake_outputs, vocoder_outputs, mlaad_splits, mailabs_splits)
    except ProtocolError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
