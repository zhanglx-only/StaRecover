"""Shared vLLM inference used by both stage 1 and stage 5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import load_records, record_count


def load_model(
    model_path: str,
    tensor_parallel_size: int,
    stage_name: str,
    gpu_memory_utilization: float = 0.9,
):
    from vllm import LLM

    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be greater than 0 and at most 1")
    print(f"========== {stage_name}: loading vLLM model ==========")
    print(f"GPU memory utilization: {gpu_memory_utilization}")
    return LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def prepare_tasks(
    input_path: str,
    input_field: str,
    label_field: str,
    rag_only: bool = False,
) -> list[tuple[dict[str, Any], str | None, str | None]]:
    tasks = []
    for index, record in enumerate(load_records(input_path), 1):
        if rag_only and "rag" not in record:
            tasks.append((record, None, None))
            continue
        prompt = record.get(input_field)
        label = record.get(label_field)
        if not isinstance(prompt, str):
            raise KeyError(f"Record {index} has no string field {input_field!r}")
        if not isinstance(label, str) or ":" not in label:
            raise ValueError(
                f"Record {index} field {label_field!r} cannot provide a name prefix"
            )
        first_token = label.split(":", 1)[0]
        tasks.append((record, first_token, prompt + first_token + ":"))
    return tasks


def inference_vllm_batch(
    input_path: str,
    output_path: str,
    model_path: str,
    input_field: str = "input",
    output_field: str = "predict",
    label_field: str = "output",
    batch_size: int = 8192,
    max_output_tokens: int = 16384,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    overwrite: bool = False,
    stage_name: str = "Inference",
    llm=None,
    rag_only: bool = False,
) -> None:
    output = Path(output_path)
    if overwrite and output.exists():
        output.unlink()

    tasks = prepare_tasks(input_path, input_field, label_field, rag_only)
    processed = record_count(output)
    if processed > len(tasks):
        raise ValueError(
            f"Output {output} contains {processed} records but only {len(tasks)} tasks exist"
        )
    if processed == len(tasks):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch(exist_ok=True)
        print(f"{stage_name} already complete: {processed}/{len(tasks)}")
        return

    from vllm import SamplingParams

    own_llm = llm is None
    if own_llm:
        llm = load_model(
            model_path,
            tensor_parallel_size,
            stage_name,
            gpu_memory_utilization,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        params = SamplingParams(
            max_tokens=max_output_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            n=1,
        )
        pending = tasks[processed:]
        if rag_only:
            results = [dict(record) for record, _, _ in pending]
            inference_tasks = [
                (index, first_token, prompt)
                for index, (_, first_token, prompt) in enumerate(pending)
                if first_token is not None and prompt is not None
            ]
            for offset in range(0, len(inference_tasks), batch_size):
                batch = inference_tasks[offset : offset + batch_size]
                responses = llm.generate([item[2] for item in batch], params)
                for response, (index, first_token, _) in zip(responses, batch):
                    results[index][output_field] = (
                        first_token + ":" + response.outputs[0].text
                    )
                done = offset + len(batch)
                print(f"{stage_name} inference: {done}/{len(inference_tasks)}")
            with output.open("a", encoding="utf-8") as handle:
                for result in results:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
            print(
                f"{stage_name}: wrote {len(tasks)} records; "
                f"generated {output_field!r} for "
                f"{sum(first_token is not None for _, first_token, _ in tasks)} "
                "records containing 'rag'"
            )
            return

        with output.open("a", encoding="utf-8") as handle:
            for offset in range(0, len(pending), batch_size):
                batch = pending[offset : offset + batch_size]
                responses = llm.generate(
                    [item[2] for item in batch if item[2] is not None], params
                )
                for response, (record, first_token, _) in zip(responses, batch):
                    assert first_token is not None
                    result = dict(record)
                    result[output_field] = first_token + ":" + response.outputs[0].text
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                done = processed + offset + len(batch)
                print(f"{stage_name}: {done}/{len(tasks)}")
    finally:
        if own_llm:
            del llm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("model_path")
    parser.add_argument("--input-field", default="input")
    parser.add_argument("--output-field", default="predict")
    parser.add_argument("--label-field", default="output")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--gpu-memory-utilization",
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stage-name", default="Inference")
    args = parser.parse_args()
    inference_vllm_batch(**vars(args))


if __name__ == "__main__":
    main()
