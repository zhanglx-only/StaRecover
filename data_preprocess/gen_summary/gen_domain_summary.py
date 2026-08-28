#!/usr/bin/env python3
"""Generate one domain summary for every binary in the train/test lists."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")


DEFAULT_TENSOR_PARALLEL_SIZE = 8
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_PROMPT_CHARS = 262_144
DEFAULT_MAX_MODEL_LEN = 262_144
DEFAULT_MAX_OUTPUT_TOKENS = 512


def load_data(file_path: Path) -> list[dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("top-level JSON value must be a list")
    return data


def save_data_atomically(file_path: Path, data: list[dict[str, Any]]) -> None:
    """Save a list without leaving a partially written dataset."""
    original_mode = file_path.stat().st_mode & 0o7777
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{file_path.name}.",
        suffix=".tmp",
        dir=file_path.parent,
        text=True,
    )
    try:
        os.fchmod(file_descriptor, original_mode)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, file_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def truncate_prompt(prompt: str, max_chars: int) -> tuple[str, bool]:
    """Truncate only string evidence while preserving instructions and output."""
    if len(prompt) <= max_chars:
        return prompt, False

    marker = "\n...[STRINGS TRUNCATED TO FIT MODEL CONTEXT]...\n"
    opening_tag = "<STRINGS_OUTPUT>"
    closing_tag = "</STRINGS_OUTPUT>"
    opening_index = prompt.find(opening_tag)
    closing_index = prompt.find(closing_tag)

    if opening_index >= 0 and closing_index > opening_index:
        evidence_start = opening_index + len(opening_tag)
        prefix = prompt[:evidence_start]
        suffix = prompt[closing_index:]
        available = max_chars - len(prefix) - len(suffix) - len(marker)
        if available > 0:
            evidence = prompt[evidence_start:closing_index]
            retained = evidence[:available]
            if "\n" in retained:
                retained = retained.rsplit("\n", 1)[0]
            truncated = prefix + retained + marker + suffix
            return truncated[:max_chars], True

    # Fallback for a malformed or unexpected prompt: preserve both the task
    # prefix and the final output instructions.
    tail_chars = min(2_000, max_chars // 5)
    head_chars = max_chars - tail_chars - len(marker)
    truncated = prompt[:head_chars] + marker + prompt[-tail_chars:]
    return truncated[:max_chars], True


def collect_pending_prompts(
    data: list[Any],
    *,
    overwrite: bool,
    max_prompt_chars: int,
) -> tuple[list[str], list[int], dict[str, int]]:
    prompts: list[str] = []
    indices: list[int] = []
    stats = {
        "items": len(data),
        "pending": 0,
        "already_completed": 0,
        "missing_input": 0,
        "non_dictionary": 0,
        "truncated": 0,
    }

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            stats["non_dictionary"] += 1
            continue

        existing_summary = item.get("domain_summary")
        if not overwrite and isinstance(existing_summary, str) and existing_summary.strip():
            stats["already_completed"] += 1
            continue

        prompt = item.get("input")
        if not isinstance(prompt, str) or not prompt.strip():
            stats["missing_input"] += 1
            continue

        prompt, was_truncated = truncate_prompt(prompt.strip(), max_prompt_chars)
        stats["truncated"] += int(was_truncated)
        prompts.append(prompt)
        indices.append(index)

    stats["pending"] = len(prompts)
    return prompts, indices, stats


def process_file(
    llm: Any,
    sampling_params: Any,
    file_path: Path,
    *,
    batch_size: int,
    max_prompt_chars: int,
    overwrite: bool,
) -> dict[str, int]:
    data = load_data(file_path)
    prompts, indices, stats = collect_pending_prompts(
        data,
        overwrite=overwrite,
        max_prompt_chars=max_prompt_chars,
    )

    if not prompts:
        return stats

    generated = 0
    empty_outputs = 0
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_indices = indices[start : start + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)

        if len(outputs) != len(batch_indices):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for "
                f"{len(batch_indices)} prompts"
            )

        for item_index, output in zip(batch_indices, outputs):
            text = output.outputs[0].text.strip() if output.outputs else ""
            data[item_index]["domain_summary"] = text
            generated += 1
            empty_outputs += int(not text)

        # Save every completed batch so an interrupted run can resume.
        save_data_atomically(file_path, data)
        print(
            f"  batch progress: {min(start + batch_size, len(prompts))}/"
            f"{len(prompts)}"
        )

    stats["generated"] = generated
    stats["empty_outputs"] = empty_outputs
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read each item's input prompt and write the generated result to "
            "its domain_summary field."
        )
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="JSON list files to process (for example ./data/test_ldd_string.json)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Local model directory (for example ./models/codebert)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=DEFAULT_TENSOR_PARALLEL_SIZE,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=DEFAULT_MAX_PROMPT_CHARS,
    )
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate non-empty domain_summary fields",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="count pending/truncated prompts without loading the model",
    )
    args = parser.parse_args()

    for name in (
        "tensor_parallel_size",
        "batch_size",
        "max_prompt_chars",
        "max_model_len",
        "max_output_tokens",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    return args


def print_stats(file_path: Path, stats: dict[str, int]) -> None:
    print(f"File: {file_path}")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def main() -> None:
    args = parse_args()

    if args.dry_run:
        for file_path in args.files:
            data = load_data(file_path)
            _, _, stats = collect_pending_prompts(
                data,
                overwrite=args.overwrite,
                max_prompt_chars=args.max_prompt_chars,
            )
            print_stats(file_path, stats)
        return

    if not args.model.exists():
        raise SystemExit(
            f"Model path does not exist: {args.model}. Pass the correct path with --model."
        )

    from vllm import LLM, SamplingParams

    print(f"Initializing model: {args.model}")
    llm = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=args.max_output_tokens,
    )

    has_error = False
    for file_path in args.files:
        try:
            stats = process_file(
                llm,
                sampling_params,
                file_path,
                batch_size=args.batch_size,
                max_prompt_chars=args.max_prompt_chars,
                overwrite=args.overwrite,
            )
            print_stats(file_path, stats)
        except Exception as error:
            has_error = True
            print(f"ERROR: {file_path}: {error}")

    if has_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
