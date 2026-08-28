"""Stage 6: propagate callsites/callees using a configurable prediction field."""

from __future__ import annotations

import argparse
import re
from typing import Any

from common import identity_value, load_records, mapping_from_prediction, save_jsonl


def prediction_mapping(record: dict[str, Any], output_field: str) -> dict[str, str]:
    """Return a usable prediction mapping, or an empty mapping if unavailable."""
    prediction = record.get(output_field)
    if not isinstance(prediction, str) or not prediction.strip():
        return {}
    return mapping_from_prediction(prediction)


def gen_predict_callsites(
    record: dict[str, Any], output_field: str
) -> dict[str, str]:
    code = record.get("code", "")
    mapping = prediction_mapping(record, output_field)
    if not mapping:
        return {}

    result: dict[str, str] = {}
    for callee in record.get("callees", []):
        original_name = callee.split(":", 1)[0].strip()
        matching_line = next(
            (
                line.strip()
                for line in code.splitlines()
                if re.search(rf"\b{re.escape(original_name)}\s*\(", line)
            ),
            None,
        )
        if matching_line is None:
            continue
        for old_name, new_name in mapping.items():
            matching_line = re.sub(
                rf"\b{re.escape(old_name)}\b", new_name, matching_line
            )
        mapped_callee = mapping.get(original_name, original_name)
        call_match = re.search(
            rf"(\b{re.escape(mapped_callee)}\s*\([^)]*\))", matching_line
        )
        result[original_name] = (
            call_match.group(1).strip() if call_match else matching_line
        )
    return result


def propagate_predict_callsites(
    records: list[dict[str, Any]], output_field: str
) -> None:
    index = {(identity_value(item), item.get("funname")): item for item in records}
    for record in records:
        for callee_name, call_expr in gen_predict_callsites(
            record, output_field
        ).items():
            target = index.get((identity_value(record), callee_name))
            if target is None:
                continue
            target.setdefault("predict_callsites", [])
            if call_expr not in target["predict_callsites"]:
                target["predict_callsites"].append(call_expr)


def generate_predict_callees(
    records: list[dict[str, Any]], output_field: str
) -> None:
    index = {(identity_value(item), item.get("funname")): item for item in records}
    for record in records:
        record.setdefault("predict_callees", [])
        for callee in record.get("callees", []):
            callee_name = callee.split(":", 1)[0].strip()
            target = index.get((identity_value(record), callee_name))
            if target is None:
                continue
            mapping = prediction_mapping(target, output_field)
            if not mapping:
                continue
            mapped_name = mapping.get(callee_name, callee_name)
            mapped_params = [
                mapping.get(param, param) for param in target.get("params_aligned", {})
            ]
            item = f"{callee_name}:{mapped_name}({', '.join(mapped_params)})"
            if item not in record["predict_callees"]:
                record["predict_callees"].append(item)


def update_json_file(
    input_path: str,
    output_path: str,
    output_field: str = "predict",
) -> None:
    records = load_records(input_path)
    for record in records:
        # Rebuild context from this round's predictions; do not retain
        # contradictory entries from earlier rounds.
        record["predict_callsites"] = []
        record["predict_callees"] = []
    propagate_predict_callsites(records, output_field)
    generate_predict_callees(records, output_field)
    save_jsonl(output_path, records)
    print(f"Stage 6: wrote {len(records)} records to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--output-field", default="predict")
    args = parser.parse_args()
    update_json_file(**vars(args))


if __name__ == "__main__":
    main()
