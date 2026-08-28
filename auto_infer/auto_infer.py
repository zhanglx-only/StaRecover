"""Run the six-stage StaRecover inference pipeline automatically.

Stage 0 filters the initial dataset. Stages 1-2 run once. Stages 3-6 run in the
first N-1 loops, and the final loop stops after stage 5. All generated files
are placed beside the input file.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from common import detect_file_format
from vllm_infer import inference_vllm_batch, load_model


SCRIPT_DIR = Path(__file__).parent
DEFAULT_BATCH_SIZE = 8192
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def gpu_memory_utilization(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


def run_command(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print(f"\n$ {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, check=True, env=env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", help="Initial JSON or JSONL dataset")
    parser.add_argument("--loops", type=nonnegative_int, default=1)
    parser.add_argument(
        "--input-field",
        default="input",
        help="Arbitrary record field name used for prompts",
    )
    parser.add_argument(
        "--output-field",
        default="predict",
        help="Arbitrary record field name used for model predictions",
    )
    parser.add_argument(
        "--label-field",
        default="output",
        help="Field used to obtain the function-name prefix and filter labels",
    )
    parser.add_argument(
        "--summary-field",
        default="summary",
        help="Record field containing summary/teacher hints for stage 3",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        metavar="PATH",
        help="Model path used by both vLLM inference stages (for example ./models/StaRecover)",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
        help="Inference batch size used by both vLLM stages (default: 8192)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        "--gpu_memory_utilization",
        type=gpu_memory_utilization,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help="Fraction of GPU memory available to vLLM (default: 0.9)",
    )
    parser.add_argument("--max-input-tokens", type=positive_int, default=16384)
    parser.add_argument("--max-output-tokens", type=positive_int, default=16384)
    parser.add_argument(
        "--num-gpus",
        type=positive_int,
        help="Number of GPUs used by vLLM; inferred from --gpus when omitted",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=positive_int,
        help="Advanced alias/override for the vLLM GPU count",
    )
    parser.add_argument(
        "--gpus",
        "--cuda-visible-devices",
        dest="cuda_visible_devices",
        help="GPU IDs to use, for example 0,2,3",
    )
    parser.add_argument(
        "--delete-intermediate",
        action="store_true",
        help="Delete numbered intermediate files after successful completion",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard this pipeline's existing outputs instead of resuming",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    input_format = detect_file_format(input_path)
    output_path = input_path.with_name(f"{input_path.stem}.final.jsonl")
    if output_path == input_path:
        raise ValueError("Final output cannot overwrite the original input file")

    gpu_devices = None
    if args.cuda_visible_devices is not None:
        gpu_devices = [
            device.strip()
            for device in args.cuda_visible_devices.split(",")
            if device.strip()
        ]
        if not gpu_devices:
            raise ValueError("--gpus must contain at least one GPU ID")
    if args.num_gpus and gpu_devices and args.num_gpus != len(gpu_devices):
        raise ValueError(
            f"--num-gpus is {args.num_gpus}, but --gpus contains "
            f"{len(gpu_devices)} IDs"
        )
    if (
        args.tensor_parallel_size
        and args.num_gpus
        and args.tensor_parallel_size != args.num_gpus
    ):
        raise ValueError("--tensor-parallel-size must equal --num-gpus")

    inferred_gpu_count = len(gpu_devices) if gpu_devices else None
    tensor_parallel_size = (
        args.tensor_parallel_size or args.num_gpus or inferred_gpu_count or 1
    )
    if gpu_devices and tensor_parallel_size != len(gpu_devices):
        raise ValueError(
            f"vLLM GPU count is {tensor_parallel_size}, but --gpus contains "
            f"{len(gpu_devices)} IDs"
        )

    selected_gpus = (
        ",".join(gpu_devices)
        if gpu_devices is not None
        else os.environ.get("CUDA_VISIBLE_DEVICES", "inherited/default")
    )
    print("========== Inference configuration ==========")
    print(f"Input format: {input_format.upper()}")
    print("Final output format: JSONL")
    print(f"GPUs: {selected_gpus}")
    print(f"GPU count / tensor parallel size: {tensor_parallel_size}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"Inference batch size: {args.batch_size}")
    print("Start point: stage 0")
    print("Model loading: once, shared by stage 1 and all stage-5 runs")

    prefix = input_path.parent / input_path.stem
    initial_filtered = Path(f"{prefix}.00_filtered.jsonl")
    anchor_prediction = Path(f"{prefix}.01_anchor_prediction.jsonl")
    anchor_context = Path(f"{prefix}.02_anchor_context.jsonl")
    intermediates = [initial_filtered, anchor_prediction, anchor_context]
    loop_paths: list[tuple[Path, Path, Path, Path | None]] = []
    for loop_number in range(1, args.loops + 1):
        loop_prefix = f"{prefix}.loop{loop_number:02d}"
        context_path = (
            Path(f"{loop_prefix}.06_context.jsonl")
            if loop_number < args.loops
            else None
        )
        paths = (
            Path(f"{loop_prefix}.03_prompt.jsonl"),
            Path(f"{loop_prefix}.04_filtered.jsonl"),
            Path(f"{loop_prefix}.05_prediction.jsonl"),
            context_path,
        )
        loop_paths.append(paths)
        intermediates.extend(paths[:3])
        if context_path is not None:
            intermediates.append(context_path)

    if args.overwrite and not args.dry_run:
        for path in [*intermediates, output_path]:
            if path.exists():
                path.unlink()

    environment = os.environ.copy()
    if gpu_devices is not None:
        visible_devices = ",".join(gpu_devices)
        environment["CUDA_VISIBLE_DEVICES"] = visible_devices
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    python = sys.executable
    shared_llm = None
    print("\n========== Stage 0: initial token filtering ==========", flush=True)
    run_command(
        [
            python,
            str(SCRIPT_DIR / "filter_token.py"),
            str(input_path),
            str(initial_filtered),
            args.model_path,
            "--input-field",
            args.input_field,
            "--label-field",
            args.label_field,
            "--max-len",
            str(args.max_input_tokens),
        ],
        environment,
        args.dry_run,
    )

    if args.dry_run:
        print(
            f"\n[shared vLLM dry-run] Stage 1 (only records containing 'rag'): "
            f"{initial_filtered} -> "
            f"{anchor_prediction}",
            flush=True,
        )
    else:
        shared_llm = load_model(
            args.model_path,
            tensor_parallel_size,
            "Shared stage 1/5 model",
            args.gpu_memory_utilization,
        )
        print("Using the same loaded model for every inference stage.")
        inference_vllm_batch(
            input_path=str(initial_filtered),
            output_path=str(anchor_prediction),
            model_path=args.model_path,
            input_field=args.input_field,
            output_field=args.output_field,
            label_field=args.label_field,
            batch_size=args.batch_size,
            max_output_tokens=args.max_output_tokens,
            tensor_parallel_size=tensor_parallel_size,
            overwrite=args.overwrite,
            stage_name="Stage 1",
            llm=shared_llm,
            rag_only=True,
        )

    run_command(
        [
            python,
            str(SCRIPT_DIR / "prop_callees.py"),
            str(anchor_prediction),
            str(initial_filtered),
            str(anchor_context),
            "--output-field",
            args.output_field,
        ],
        environment,
        args.dry_run,
    )
    current = anchor_context

    for loop_number, (prompt_path, filtered_path, prediction_path, context_path) in enumerate(
        loop_paths, 1
    ):
        print(f"\n========== Loop {loop_number}/{args.loops} ==========", flush=True)
        prompt_command = [
            python,
            str(SCRIPT_DIR / "prop_prompt.py"),
            str(current),
            str(prompt_path),
            "--input-field",
            args.input_field,
            "--summary-field",
            args.summary_field,
        ]
        run_command(prompt_command, environment, args.dry_run)
        filter_input = prompt_path
        run_command(
            [
                python,
                str(SCRIPT_DIR / "filter_token.py"),
                str(filter_input),
                str(filtered_path),
                args.model_path,
                "--input-field",
                args.input_field,
                "--label-field",
                args.label_field,
                "--max-len",
                str(args.max_input_tokens),
            ],
            environment,
            args.dry_run,
        )
        if args.dry_run:
            print(
                f"\n[shared vLLM dry-run] Stage 5 (loop {loop_number}): "
                f"{filtered_path} -> {prediction_path}",
                flush=True,
            )
        else:
            if shared_llm is None:
                shared_llm = load_model(
                    args.model_path,
                    tensor_parallel_size,
                    "Shared stage-5 model",
                    args.gpu_memory_utilization,
                )
                print("Using the same loaded model for every stage-5 run.")
            inference_vllm_batch(
                input_path=str(filtered_path),
                output_path=str(prediction_path),
                model_path=args.model_path,
                input_field=args.input_field,
                output_field=args.output_field,
                label_field=args.label_field,
                batch_size=args.batch_size,
                max_output_tokens=args.max_output_tokens,
                tensor_parallel_size=tensor_parallel_size,
                overwrite=args.overwrite,
                stage_name=f"Stage 5 (loop {loop_number})",
                llm=shared_llm,
            )
        if context_path is not None:
            run_command(
                [
                    python,
                    str(SCRIPT_DIR / "prop_context.py"),
                    str(prediction_path),
                    str(context_path),
                    "--output-field",
                    args.output_field,
                ],
                environment,
                args.dry_run,
            )
            current = context_path
        else:
            print(
                "Final loop: skip stage 6 context propagation; "
                "use stage 5 prediction as the final result.",
                flush=True,
            )
            current = prediction_path

    if args.dry_run:
        print(f"\nWould write final JSONL result: {current} -> {output_path}")
        if args.delete_intermediate:
            print(f"Would delete {len(intermediates)} intermediate files")
        return

    shutil.copyfile(current, output_path)
    if args.delete_intermediate:
        for path in intermediates:
            if path.exists() and path != output_path:
                path.unlink()

    if shared_llm is not None:
        del shared_llm

    print("\n========== Automated inference complete ==========")
    print(f"Final output: {output_path}")
    print(
        "Intermediate files: deleted"
        if args.delete_intermediate
        else f"Intermediate files: retained beside {input_path}"
    )


if __name__ == "__main__":
    main()
