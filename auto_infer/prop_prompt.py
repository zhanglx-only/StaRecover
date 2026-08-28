"""Stage 3: rebuild prompts from the latest propagated context."""

from __future__ import annotations

import argparse
from typing import Any

from common import load_records, save_jsonl


def prediction_targets(record: dict[str, Any]) -> str:
    funname = [record["funname"]]
    callees = [item.split(":", 1)[0] for item in record.get("callees", [])]
    params = list(record.get("params_aligned", {}).keys())
    variables = list(record.get("vars_aligned", {}).keys())
    return ", ".join(params + variables + funname + callees)


def generate_context_prompt(
    record: dict[str, Any],
    summary_field: str = "summary",
) -> str:
    teacher_hints = record.get(summary_field)
    if not isinstance(teacher_hints, str):
        raise KeyError(
            f"Record {record.get('funname', '<unknown>')!r} has no string "
            f"summary field {summary_field!r}; set --summary-field correctly"
        )
    teacher_hints = teacher_hints.replace("<SUMMARY>", "").replace(
        "</SUMMARY>", ""
    )
    summary = f"<SUMMARY>\n{teacher_hints}\n</SUMMARY>\n"

    callsites = ", ".join(record.get("predict_callsites", []))
    callees = ", ".join(record.get("predict_callees", []))
    return f"""
{summary}<CallSites>
{callsites}
</CallSites>
<Callees>
{callees}
</Callees>
<code>
{record["code"]}
</code>
predict the original names of {prediction_targets(record)}.
    """


def generate_rag_prompt(record: dict[str, Any]) -> str:
    return f"""
<unstripped_code>
{record["rag"]["unstripped_code"]}
</unstripped_code>
<code>
{record["code"]}
</code>
predict the original names of {prediction_targets(record)}.
    """.strip()


def add_prompt(
    input_path: str,
    output_path: str,
    input_field: str = "input",
    summary_field: str = "summary",
) -> None:
    records = load_records(input_path)
    for index, record in enumerate(records, 1):
        use_context_prompt = "rag" not in record
        if use_context_prompt:
            record[input_field] = generate_context_prompt(record, summary_field)
        else:
            record[input_field] = generate_rag_prompt(record)
        if index % 1000 == 0:
            print(f"Stage 3: prepared {index}/{len(records)} prompts")
    save_jsonl(output_path, records)
    print(f"Stage 3: wrote {len(records)} records to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--input-field", default="input")
    parser.add_argument("--summary-field", default="summary")
    args = parser.parse_args()
    add_prompt(**vars(args))


if __name__ == "__main__":
    main()
