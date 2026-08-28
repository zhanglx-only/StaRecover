import argparse
import concurrent.futures
import json
import os
import re
import signal
import subprocess
from pathlib import Path

from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


COMMAND_TIMEOUT_SECONDS = 60 * 60
DEFAULT_WORKERS = min(32, max(1, os.cpu_count() or 1))


# ------------------------------ #
# 1. Command execution helper
# ------------------------------ #
def run_cmd(command, timeout=COMMAND_TIMEOUT_SECONDS):
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return stdout
        except subprocess.TimeoutExpired:
            # Terminate the complete process group to avoid orphaned children.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()
            command_name = command[0] if command else "command"
            return (
                f"[Timeout] {command_name} exceeded "
                f"{timeout} seconds and was terminated."
            )
    except Exception as e:
        return str(e)


# ------------------------------ #
# 2. Binary analysis
# ------------------------------ #
def analyze_binary(binary_path):

    # Clean ``file`` output.
    def clean_file_output(text):
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r'^[^:]+:\s*', '', line)
            lines.append(line)
        return "\n".join(lines)

    # Deduplicate lines while preserving order.
    def deduplicate_text(text):
        seen = set()
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                lines.append(line)
        return "\n".join(lines)

    # Keep only the dependency name on each LDD line:
    # linux-vdso.so.1 (0x...) -> linux-vdso.so.1
    # libc.so.6 => /path/libc.so.6 (0x...) -> libc.so.6
    def clean_ldd_output(text):
        cleaned_lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            if "=>" in line:
                line = line.split("=>", 1)[0].strip()
            else:
                line = re.sub(r"\s+\(0x[0-9a-fA-F]+\)\s*$", "", line)

            line = line.strip()
            if line:
                cleaned_lines.append(line)
        return deduplicate_text("\n".join(cleaned_lines))

    # --- file output ---
    file_output = deduplicate_text(
        clean_file_output(run_cmd(["file", binary_path]))
    )

    # --- ldd output ---
    ldd_output = clean_ldd_output(run_cmd(["ldd", binary_path]))

    # --- strings output ---
    strings_output = run_cmd(["strings", binary_path])
    placeholder_patterns = [r"%[sdlu]", r"0x[0-9a-fA-F]+", r"^[\W\d_]+$"]

    filtered_set = set()
    filtered_lines = []

    for line in strings_output.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(re.search(p, line) for p in placeholder_patterns):
            continue
        if line in filtered_set:
            continue
        filtered_set.add(line)
        filtered_lines.append(line)

    # Keep the strings_truncated field name for downstream compatibility;
    # the value is no longer truncated here.
    strings_truncated = "\n".join(filtered_lines)

    return {
        "file_output": file_output,
        "ldd_output": ldd_output,
        "strings_truncated": strings_truncated
    }


# ------------------------------ #
# 3. Process one .result file
# ------------------------------ #
def process_binary_with_result(result_path):
    try:
        result_path_obj = Path(result_path)
        binary_path = result_path_obj.with_suffix("")  # Remove .result.
        proj_name = result_path_obj.parent.name
        bin_name = result_path_obj.stem
        project_name = (
            proj_name.split("_", 1)[1] if "_" in proj_name else proj_name
        )

        if not binary_path.exists():
            return False, f"Binary file not found: {binary_path}"

        analysis = analyze_binary(str(binary_path))
        analysis["project_name"] = project_name

        # Write the analysis JSON with project and binary identifiers.
        analysis_file_path = result_path_obj.with_suffix(".json")
        with open(analysis_file_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "analysis": analysis,
                    "proj": proj_name,
                    "bin": bin_name,
                },
                f,
                indent=4,
                ensure_ascii=False,
            )

        return True, f"Generated {analysis_file_path}"

    except Exception as e:
        return False, f"Processing failed: {result_path}; error: {e}"


# ------------------------------ #
# 4. Process every .result file in a directory
# ------------------------------ #
def process_all_results_in_directory(result_dir, workers=DEFAULT_WORKERS):
    result_files = sorted(Path(result_dir).rglob("*.result"))
    if not result_files:
        raise SystemExit(f"No .result files found under {result_dir}")

    succeeded = 0
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_path = {
            executor.submit(process_binary_with_result, result_path): result_path
            for result_path in result_files
        }

        completed = concurrent.futures.as_completed(future_to_path)
        for future in tqdm(
            completed,
            total=len(future_to_path),
            desc="Processing .result files",
            unit="file",
        ):
            result_path = future_to_path[future]
            try:
                success, message = future.result()
            except Exception as error:
                success = False
                message = f"Processing failed: {result_path}; error: {error}"

            if success:
                succeeded += 1
            else:
                failed += 1
                tqdm.write(f"ERROR: {message}")

    print(f"Completed: success={succeeded}, failed={failed}, workers={workers}")
    return succeeded, failed


# ------------------------------ #
# 5. Command-line entry point
# ------------------------------ #
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Concurrently extract binary file, LDD, and strings information; "
            "each external command runs for at most one hour."
        )
    )
    parser.add_argument(
        "result_dir",
        type=Path,
        help="Directory to scan recursively (for example ./sample_results)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of workers (default: {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def main():
    args = parse_args()
    _, failed = process_all_results_in_directory(args.result_dir, args.workers)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
