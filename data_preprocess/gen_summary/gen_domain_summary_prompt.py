#!/usr/bin/env python3
"""Generate binary-domain-summary prompts from LDD and strings only.

The input JSON must be a list of dictionaries.  Each dictionary is expected to
contain ``analysis.ldd_output`` and ``analysis.strings_truncated``.  The
generated prompt is stored in the dictionary's ``input`` field.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


PROMPT_TEMPLATE = """You are a binary reverse-engineering expert specialized in extracting semantic evidence for variable and function name recovery.

Infer a concise binary-level semantic summary using ONLY the LDD dependencies and readable strings provided below. The summary will be used as external semantic evidence for recovering meaningful identifiers in stripped binaries.

Your goal is to identify the binary's application domain and extract high-level semantic information, including important entities and operations, that can guide downstream identifier recovery.

Evidence guidance:
- Treat distinctive libraries, frameworks, protocols, file formats, devices, commands, paths, ioctl interfaces, and error messages as strong domain evidence.
- Treat common runtime libraries such as libc, libm, libpthread, and the dynamic loader as weak evidence.
- Strings may contain compiler artifacts, symbol fragments, random bytes, or unrelated bundled data; ignore such noise.
- Focus on evidence that reveals:
  * controlled hardware or software components
  * input/output data objects
  * resource handles and identifiers
  * memory buffers and storage objects
  * configuration parameters
  * command-line arguments
  * file/device/network interactions
  * important runtime behaviors

Produce:

- primary_domain:
  The most likely specific application or system domain.

- binary_role:
  The likely responsibility or purpose of the binary.

- key_entities:
  Important semantic entities involved in the program that may correspond to variables, structures, or function parameters.
  Include resources, handles, buffers, files, IDs, dimensions, counters, and configuration objects.

- operation_semantics:
  Important operations inferred from the evidence.
  Describe what the program does with the identified entities, such as device access, initialization, parsing, memory mapping, data transfer, or communication.

Rules:
- Use ONLY evidence from the LDD dependencies and readable strings.
- Prefer specific supported concepts over broad labels such as "utility", "application", or "data processing".
- Do not invent unsupported behavior, algorithms, or implementation details.
- If evidence is insufficient, use unknown.
- Do not generate exact variable names.
- Focus on semantic information useful for downstream identifier summary generation.
- Keep the complete summary under 200 words.

<LDD_OUTPUT>
{LDD_OUTPUT}
</LDD_OUTPUT>

<STRINGS_OUTPUT>
{STRINGS_OUTPUT}
</STRINGS_OUTPUT>

Output only:

<DOMAIN_SUMMARY>

primary_domain:
...

binary_role:
...

key_entities:
- ...

operation_semantics:
- ...

</DOMAIN_SUMMARY>
"""


def normalize_evidence(value: Any) -> str:
    """Convert evidence to deduplicated, non-empty lines without adding data."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)

    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if line and line not in seen:
            seen.add(line)
            lines.append(line)
    return "\n".join(lines)


def build_prompt(item: dict[str, Any]) -> tuple[str, bool, bool]:
    """Build one prompt and report whether LDD or strings are missing."""
    analysis = item.get("analysis", {})
    if not isinstance(analysis, dict):
        analysis = {}

    ldd_output = normalize_evidence(analysis.get("ldd_output"))
    strings_output = normalize_evidence(analysis.get("strings_truncated"))

    prompt = PROMPT_TEMPLATE.format(
        LDD_OUTPUT=ldd_output or "(no LDD evidence available)",
        STRINGS_OUTPUT=strings_output or "(no readable string evidence available)",
    )
    return prompt, not bool(ldd_output), not bool(strings_output)


def write_json_atomically(path: Path, data: Any) -> None:
    """Write JSON beside the destination and atomically replace it."""
    original_mode = path.stat().st_mode & 0o7777
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        os.fchmod(file_descriptor, original_mode)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=4)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def process_file(path: Path, dry_run: bool = False) -> dict[str, int]:
    """Add an input prompt to every dictionary in one JSON list."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("top-level JSON value must be a list")

    stats = {
        "items": 0,
        "updated": 0,
        "non_dictionary": 0,
        "missing_ldd": 0,
        "missing_strings": 0,
        "missing_both": 0,
    }

    for item in data:
        stats["items"] += 1
        if not isinstance(item, dict):
            stats["non_dictionary"] += 1
            continue

        prompt, missing_ldd, missing_strings = build_prompt(item)
        item["input"] = prompt
        stats["updated"] += 1
        stats["missing_ldd"] += int(missing_ldd)
        stats["missing_strings"] += int(missing_strings)
        stats["missing_both"] += int(missing_ldd and missing_strings)

    if not dry_run:
        write_json_atomically(path, data)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add binary-domain-summary prompts to the input field using only "
            "analysis.ldd_output and analysis.strings_truncated."
        )
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="JSON list files to update (for example ./data/test_ldd_string.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print statistics without writing files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    has_error = False

    for path in args.files:
        try:
            stats = process_file(path, dry_run=args.dry_run)
        except Exception as error:
            has_error = True
            print(f"ERROR: {path}: {error}")
            continue

        action = "Would update" if args.dry_run else "Updated"
        print(f"{action}: {path}")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    if has_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
