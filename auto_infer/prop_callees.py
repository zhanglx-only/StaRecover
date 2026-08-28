"""Stage 2: propagate anchor predictions to the complete input dataset."""

from __future__ import annotations

import argparse
from typing import Any

from common import identity_value, load_records, mapping_from_prediction, save_jsonl


def generate_predict_callees(
    base_records: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
    output_field: str,
) -> list[dict[str, Any]]:
    prediction_index = {
        (identity_value(item), item.get("funname")): item for item in prediction_records
    }
    for record in base_records:
        record["predict_callsites"] = []
        record["predict_callees"] = []
        for callee in record.get("callees", []):
            callee_name = callee.split(":", 1)[0].strip()
            target = prediction_index.get((identity_value(record), callee_name))
            if target is None:
                continue
            prediction = target.get(output_field)
            if not isinstance(prediction, str) or not prediction.strip():
                continue
            mapping = mapping_from_prediction(prediction)
            if not mapping:
                continue
            mapped_name = mapping.get(callee_name, callee_name)
            mapped_params = [
                mapping.get(param, param) for param in target.get("params_aligned", {})
            ]
            item = f"{callee_name}:{mapped_name}({', '.join(mapped_params)})"
            if item not in record["predict_callees"]:
                record["predict_callees"].append(item)
    return base_records


def update_json_file(
    prediction_path: str,
    base_path: str,
    output_path: str,
    output_field: str = "predict",
) -> None:
    records = generate_predict_callees(
        load_records(base_path), load_records(prediction_path), output_field
    )
    save_jsonl(output_path, records)
    print(f"Stage 2: wrote {len(records)} records to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_path")
    parser.add_argument("base_path")
    parser.add_argument("output_path")
    parser.add_argument("--output-field", default="predict")
    args = parser.parse_args()
    update_json_file(**vars(args))


if __name__ == "__main__":
    main()
