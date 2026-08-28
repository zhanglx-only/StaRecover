import os
import subprocess
import json
import time
from pathlib import Path
import argparse

TIMEOUT_PER_FILE = 7200  

def is_binary_file(filepath):
   
    binary_extensions = ['', '.exe', '.dll', '.so', '.bin', '.elf', '.out']
    return Path(filepath).suffix.lower() in binary_extensions


def find_all_binary_files(root_dir):
    
    binary_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath) and is_binary_file(filepath):
                binary_files.append(filepath)
    return binary_files


def run_ida_safe(binary_path, IDA_PATH, SCRIPT_PATH):
    
    cmd = [
        IDA_PATH,
        "-B",
        "-A",
        "-S" + SCRIPT_PATH,
        str(binary_path)
    ]

    print(f"Processing: {binary_path}")

    try:
        start_time = time.time()
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=TIMEOUT_PER_FILE,
            encoding='utf-8'
        )
        processing_time = time.time() - start_time

        
        binary_path = Path(binary_path)

        
        output_file = binary_path.with_name(f"{binary_path.name}.decompiled")

        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Successfully generated {len(data)} functions ({processing_time:.1f}s)")
                return True
        else:
            print(f"Failed: .decompiled file was not generated")
            return False

    except subprocess.TimeoutExpired:
        print(f"Timeout: exceeded {TIMEOUT_PER_FILE} seconds")
        return False
    except Exception as e:
        print(f"Exception: {str(e)}")
        return False



def delete_temp_files(directory):
    temp_exts = ['.i64', '.id0', '.id1', '.id2', '.nam', '.til']
    deleted_count = 0
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in temp_exts):
                try:
                    os.remove(os.path.join(dirpath, filename))
                    deleted_count += 1
                except:
                    pass
    if deleted_count > 0:
        print(f"Cleaned up {deleted_count} temporary files")


def process(STRIPPED_ROOT, UNSTRIPPED_ROOT, IDA_PATH, SCRIPT_PATH):

    root_dirs = [STRIPPED_ROOT, UNSTRIPPED_ROOT]
    #root_dirs = [STRIPPED_ROOT]
    all_binary_files = []

    for root_dir in root_dirs:
        files = find_all_binary_files(root_dir)
        all_binary_files.extend(files)

    if not all_binary_files:
        print("No binary files found")
        return

    print(f"Found {len(all_binary_files)} binary files")

    success_count = 0
    failed_count = 0
    skipped_count = 0  

    for i, binary_path in enumerate(all_binary_files, 1):
        binary_path = Path(binary_path)
        output_file = binary_path.with_name(f"{binary_path.name}.decompiled")
        if output_file.exists():
            print(f"\n[{i}/{len(all_binary_files)}] Skipped: {output_file.name} already exists")
            skipped_count += 1
            continue  
        print(f"\n[{i}/{len(all_binary_files)}] ", end="")
        if run_ida_safe(binary_path, IDA_PATH, SCRIPT_PATH):
            success_count += 1
        else:
            failed_count += 1

    print("\nProcessing completed:")
    print(f"Success: {success_count} files")
    print(f"Skipped: {skipped_count} files")
    print(f"Failed: {failed_count} files")
    print("\nCleaning up temporary files...")
    for root_dir in root_dirs:
        delete_temp_files(root_dir)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--STRIPPED_ROOT",
        required=True,
        help="Path to the stripped binaries root directory"
    )

    parser.add_argument(
        "--UNSTRIPPED_ROOT",
        required=True,
        help="Path to the unstripped binaries root directory"
    )

    parser.add_argument(
        "--IDA_PATH",
        required=True,
        help="Path to IDA Pro executable or installation directory"
    )

    parser.add_argument(
        "--IDA_SCRIPT_PATH",
        required=True,
        help="Path to analysis or processing script"
    )

    args = parser.parse_args()

    
    process(args.STRIPPED_ROOT, args.UNSTRIPPED_ROOT, args.IDA_PATH, args.IDA_SCRIPT_PATH)
