#!/usr/bin/env python3
"""Generate identifier summaries from prompts stored in JSON-list records."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")


DEFAULT_INPUT_FIELD = "qwen"
DEFAULT_OUTPUT_FIELD = "identifier_summary"
DEFAULT_TENSOR_PARALLEL_SIZE = 8
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_PROMPT_CHARS = 262_144
DEFAULT_MAX_MODEL_LEN = 262_144
DEFAULT_MAX_OUTPUT_TOKENS = 2_048
DEFAULT_CHECKPOINT_EVERY_BATCHES = 10


def load_data(file_path: Path) -> list[Any]:
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"top-level JSON value must be a list: {file_path}")
    return data


def save_data_atomically(file_path: Path, data: list[Any]) -> None:
    """Save without leaving a partially written dataset."""
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
    """Shorten function code while retaining instructions and target identifiers."""
    if len(prompt) <= max_chars:
        return prompt, False

    opening_tag = "<FUNCTION_CODE>"
    closing_tag = "</FUNCTION_CODE>"
    opening_index = prompt.find(opening_tag)
    closing_index = prompt.find(closing_tag)
    marker = "\n...[FUNCTION CODE TRUNCATED TO FIT MODEL CONTEXT]...\n"

    if opening_index >= 0 and closing_index > opening_index:
        code_start = opening_index + len(opening_tag)
        prefix = prompt[:code_start]
        suffix = prompt[closing_index:]
        available = max_chars - len(prefix) - len(suffix) - len(marker)
        if available > 0:
            code = prompt[code_start:closing_index]
            head_chars = available * 3 // 4
            tail_chars = available - head_chars
            retained_head = code[:head_chars]
            retained_tail = code[-tail_chars:] if tail_chars else ""
            truncated = prefix + retained_head + marker + retained_tail + suffix
            return truncated[:max_chars], True

    # Preserve the beginning instructions and final target/output constraints if
    # the prompt does not contain the expected function-code tags.
    tail_chars = min(8_000, max_chars // 4)
    head_chars = max_chars - tail_chars - len(marker)
    truncated = prompt[:head_chars] + marker + prompt[-tail_chars:]
    return truncated[:max_chars], True


def collect_pending_prompts(
    data: list[Any],
    *,
    input_field: str,
    output_field: str,
    overwrite: bool,
    max_prompt_chars: int,
) -> tuple[list[str], list[int], dict[str, int]]:
    prompts: list[str] = []
    indices: list[int] = []
    stats = {
        "items": len(data),
        "pending": 0,
        "already_completed": 0,
        "missing_prompt": 0,
        "non_dictionary": 0,
        "truncated": 0,
    }

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            stats["non_dictionary"] += 1
            continue

        existing_summary = item.get(output_field)
        if (
            not overwrite
            and isinstance(existing_summary, str)
            and existing_summary.strip()
        ):
            stats["already_completed"] += 1
            continue

        prompt = item.get(input_field)
        if not isinstance(prompt, str) or not prompt.strip():
            stats["missing_prompt"] += 1
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
    input_field: str,
    output_field: str,
    batch_size: int,
    max_prompt_chars: int,
    checkpoint_every_batches: int,
    overwrite: bool,
) -> dict[str, int]:
    data = load_data(file_path)
    prompts, indices, stats = collect_pending_prompts(
        data,
        input_field=input_field,
        output_field=output_field,
        overwrite=overwrite,
        max_prompt_chars=max_prompt_chars,
    )

    if not prompts:
        return stats

    generated = 0
    empty_outputs = 0
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    for batch_number, start in enumerate(
        range(0, len(prompts), batch_size), start=1
    ):
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
            data[item_index][output_field] = text
            generated += 1
            empty_outputs += int(not text)

        should_checkpoint = (
            batch_number % checkpoint_every_batches == 0
            or batch_number == total_batches
        )
        if should_checkpoint:
            save_data_atomically(file_path, data)

        print(
            f"  batch progress: {min(start + batch_size, len(prompts))}/"
            f"{len(prompts)}; checkpoint={'yes' if should_checkpoint else 'no'}"
        )

    stats["generated"] = generated
    stats["empty_outputs"] = empty_outputs
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read identifier-summary prompts from each JSON item and write "
            "the generated results to identifier_summary fields."
        )
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="JSON list files to process (for example ./data/test.json)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Local model directory (for example ./models/codebert)",
    )
    parser.add_argument(
        "--input-field",
        default=DEFAULT_INPUT_FIELD,
        help=f"field containing the prompt (default: {DEFAULT_INPUT_FIELD})",
    )
    parser.add_argument(
        "--output-field",
        default=DEFAULT_OUTPUT_FIELD,
        help=f"field receiving the result (default: {DEFAULT_OUTPUT_FIELD})",
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
        "--checkpoint-every-batches",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY_BATCHES,
        help=(
            "atomically save after this many batches; smaller values improve "
            "resume granularity but rewrite the JSON more often"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate non-empty output fields",
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
        "checkpoint_every_batches",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if not args.input_field.strip():
        parser.error("--input-field cannot be empty")
    if not args.output_field.strip():
        parser.error("--output-field cannot be empty")
    if args.input_field == args.output_field:
        parser.error("--input-field and --output-field must be different")
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
                input_field=args.input_field,
                output_field=args.output_field,
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
        temperature=0.2,
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
                input_field=args.input_field,
                output_field=args.output_field,
                batch_size=args.batch_size,
                max_prompt_chars=args.max_prompt_chars,
                checkpoint_every_batches=args.checkpoint_every_batches,
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
