#!/usr/bin/env python3
"""Generate identifier-summary prompts from binary domain summaries."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_FIELD = "qwen"


PROMPT_TEMPLATE = """You are a binary reverse-engineering expert.

Analyze the IDA-decompiled function below and produce a concise semantic summary for every target identifier. Function names and identifiers are placeholders. The summaries will be used by another model to recover meaningful identifier names.

Use the two evidence sources jointly:
- The binary domain summary provides the likely application domain and whole-binary role.
- The function code provides the identifier's local behavior, data flow, inputs, outputs, and relationships.

For every identifier:
- FUNC: describe its main action, processed object, result, and likely domain role.
- PARAM: describe the represented object or value, input/output role, and effect on the function.
- VAR: describe its source, semantic value, and subsequent use.
- CALL: infer the called function's purpose from its arguments, return value, and surrounding code.

Rules:
- Treat the domain summary as a high-level prior, not as proof of local behavior.
- Prefer function-code evidence when the domain summary is broad, uncertain, or inconsistent with the code.
- Provide concrete semantics useful for name recovery, such as buffer, length, key, state, index, offset, flag, result, error code, handle, path, packet, or context when supported.
- Avoid vague explanations such as "input parameter", "temporary variable", or "processes data".
- Do not merely repeat C types, restate the identifier, or invent unsupported behavior.
- Describe semantic meaning and functional role only; do not propose or list candidate identifier names.
- Explain every listed identifier exactly once and preserve the listed order.

Format example:

<EXAMPLE_TARGET_IDENTIFIERS>
FUNC: sub_401000
PARAM: a1
VAR: v3
CALL: sub_402000
</EXAMPLE_TARGET_IDENTIFIERS>

<EXAMPLE_SUMMARY>
FUNC: sub_401000: validates an input buffer, passes its contents to a processing routine, and returns the resulting status
PARAM: a1: points to the input buffer consumed by the function and supplied to the processing routine
VAR: v3: stores the buffer length calculated before validation and controls the permitted processing range
CALL: sub_402000: processes the validated buffer using its address and length and returns an operation status
</EXAMPLE_SUMMARY>

Follow the demonstrated structure only. Infer all semantics exclusively from the domain summary and function code below; do not copy semantic content from the example.

<DOMAIN_SUMMARY>
{DOMAIN_SUMMARY}
</DOMAIN_SUMMARY>

<FUNCTION_CODE>
{FUNCTION_CODE}
</FUNCTION_CODE>

<TARGET_IDENTIFIERS>
{ENTITY_LINES}
</TARGET_IDENTIFIERS>

Output only:

<SUMMARY>
TYPE: identifier: concise semantic meaning and functional role
</SUMMARY>
"""


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


def normalize_domain_summary(value: Any) -> str:
    """Extract the content of an optional outer DOMAIN_SUMMARY block."""
    text = normalize_text(value)
    opening_tag = "<DOMAIN_SUMMARY>"
    closing_tag = "</DOMAIN_SUMMARY>"
    opening_index = text.find(opening_tag)
    closing_index = text.rfind(closing_tag)
    if opening_index >= 0 and closing_index > opening_index:
        text = text[opening_index + len(opening_tag) : closing_index].strip()
    return text


def extract_entity_names(item: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """Extract placeholder parameter, variable, and callee names in order."""
    params = item.get("params_aligned", {})
    variables = item.get("vars_aligned", {})
    callees = item.get("callees", [])

    param_names = list(params) if isinstance(params, dict) else []
    variable_names = list(variables) if isinstance(variables, dict) else []

    call_names: list[str] = []
    seen_calls: set[str] = set()
    if isinstance(callees, list):
        for callee in callees:
            call_name = str(callee).split(":", 1)[0].strip()
            if call_name and call_name not in seen_calls:
                seen_calls.add(call_name)
                call_names.append(call_name)

    return param_names, variable_names, call_names


def build_identifier_summary_prompt(
    item: dict[str, Any], domain_summary: str
) -> str | None:
    code = normalize_text(item.get("code"))
    if not code:
        return None
    if code.startswith("code:"):
        code = "\n".join(code.splitlines()[1:]).strip()

    param_names, variable_names, call_names = extract_entity_names(item)
    entity_lines = [f"PARAM: {name}" for name in param_names]
    entity_lines.extend(f"VAR: {name}" for name in variable_names)

    function_name = normalize_text(item.get("funname"))
    if function_name:
        entity_lines.append(f"FUNC: {function_name}")
    entity_lines.extend(f"CALL: {name}" for name in call_names)

    return PROMPT_TEMPLATE.format(
        DOMAIN_SUMMARY=normalize_domain_summary(domain_summary),
        FUNCTION_CODE=code,
        ENTITY_LINES="\n".join(entity_lines),
    )


def load_json_list(path: Path) -> list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"top-level JSON value must be a list: {path}")
    return data


def load_domain_summary_index(path: Path) -> dict[tuple[str, str], str]:
    """Index non-empty domain summaries by (proj, bin)."""
    data = load_json_list(path)
    summary_index: dict[tuple[str, str], str] = {}

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        proj = normalize_text(item.get("proj"))
        binary = normalize_text(item.get("bin"))
        domain_summary = normalize_domain_summary(item.get("domain_summary"))
        if not proj or not binary or not domain_summary:
            continue

        key = (proj, binary)
        if key in summary_index and summary_index[key] != domain_summary:
            raise ValueError(
                f"conflicting domain summaries for {key} at item {index}"
            )
        summary_index[key] = domain_summary

    if not summary_index:
        raise ValueError(f"no non-empty domain_summary fields found in {path}")
    return summary_index


def write_json_atomically(path: Path, data: list[Any]) -> None:
    original_mode = path.stat().st_mode & 0o7777
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        os.fchmod(file_descriptor, original_mode)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def process_function_file(
    function_file: Path,
    domain_summary_file: Path,
    *,
    output_field: str,
    dry_run: bool,
) -> dict[str, int]:
    summary_index = load_domain_summary_index(domain_summary_file)
    data = load_json_list(function_file)
    stats = {
        "items": len(data),
        "updated": 0,
        "non_dictionary": 0,
        "missing_proj_or_bin": 0,
        "missing_domain_summary": 0,
        "missing_code": 0,
    }

    for item in data:
        if not isinstance(item, dict):
            stats["non_dictionary"] += 1
            continue

        proj = normalize_text(item.get("proj"))
        binary = normalize_text(item.get("bin"))
        if not proj or not binary:
            stats["missing_proj_or_bin"] += 1
            continue

        domain_summary = summary_index.get((proj, binary))
        if domain_summary is None:
            stats["missing_domain_summary"] += 1
            continue

        prompt = build_identifier_summary_prompt(item, domain_summary)
        if prompt is None:
            stats["missing_code"] += 1
            continue

        item[output_field] = prompt
        stats["updated"] += 1

    if not dry_run:
        write_json_atomically(function_file, data)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join function records with binary domain summaries by (proj, bin) "
            "and generate identifier-summary prompts."
        )
    )
    parser.add_argument(
        "function_file",
        type=Path,
        help="Function-level JSON list (for example ./data/functions.json)",
    )
    parser.add_argument(
        "domain_summary_file",
        type=Path,
        help=(
            "Binary-level JSON list containing domain_summary "
            "(for example ./data/domain_summaries.json)"
        ),
    )
    parser.add_argument(
        "--output-field",
        default=DEFAULT_OUTPUT_FIELD,
        help=f"field receiving the generated prompt (default: {DEFAULT_OUTPUT_FIELD})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and count without modifying the function file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        stats = process_function_file(
            args.function_file,
            args.domain_summary_file,
            output_field=args.output_field,
            dry_run=args.dry_run,
        )
    except Exception as error:
        raise SystemExit(f"ERROR: {error}") from error

    action = "Would update" if args.dry_run else "Updated"
    print(f"{action}: {args.function_file}")
    print(f"Domain summaries: {args.domain_summary_file}")
    print(f"Output field: {args.output_field}")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
