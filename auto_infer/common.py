"""Shared JSON/JSONL helpers for the automated inference pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def detect_file_format(path: str | Path) -> str:
    """Return ``json`` for a JSON document, otherwise ``jsonl``."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        first_line = next((line.strip() for line in handle if line.strip()), "")
        if not first_line:
            return "jsonl" if path.suffix.lower() == ".jsonl" else "json"
        if first_line.startswith("["):
            return "json"
        try:
            first_record = json.loads(first_line)
        except json.JSONDecodeError:
            # A pretty-printed JSON object normally starts with a lone "{".
            return "json"
        if not isinstance(first_record, dict):
            return "json"
        if path.suffix.lower() == ".jsonl":
            return "jsonl"
        has_another_record = any(line.strip() for line in handle)
        return "jsonl" if has_another_record else "json"


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON array/object or a JSONL file as a flat record list."""
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path}, line {line_number}: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(
                    f"Expected an object in {path}, line {line_number}, "
                    f"got {type(item).__name__}"
                )
            records.append(item)
        return records

    if isinstance(loaded, dict):
        return [loaded]
    if not isinstance(loaded, list):
        raise ValueError(f"Expected a JSON object/array in {path}")

    records = []
    for item in loaded:
        if isinstance(item, dict):
            records.append(item)
        elif isinstance(item, list):
            records.extend(subitem for subitem in item if isinstance(subitem, dict))
    return records


def save_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Write records in JSONL format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_records(
    path: str | Path,
    records: Iterable[dict[str, Any]],
    file_format: str,
) -> None:
    """Save records as a JSON array or JSONL according to ``file_format``."""
    if file_format == "jsonl":
        save_jsonl(path, records)
        return
    if file_format != "json":
        raise ValueError(f"Unsupported output format: {file_format}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(records), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def record_count(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def mapping_from_prediction(prediction: Any) -> dict[str, str]:
    """Parse ``old:new, old2:new2`` prediction text."""
    if not isinstance(prediction, str):
        return {}
    mapping: dict[str, str] = {}
    for item in prediction.split(","):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        if key.strip():
            mapping[key.strip()] = value.strip()
    return mapping


def identity_value(record: dict[str, Any]) -> Any:
    """Return the binary identity used by either dataset schema."""
    return record.get("binname", record.get("bin"))
