# -*- coding: utf-8 -*-
import json
import os
import re
from pathlib import Path
import argparse


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_name(name: str):
    match = re.search(r'([A-Za-z_]\w*)\s*\((.*?)\)', name, re.S)
    if not match:
        return name.strip()
    func_name = match.group(1)
    params = match.group(2).strip()
    if params:
        param_names = []
        for p in params.split(','):
            tokens = p.strip().split()
            if tokens:
                name_part = tokens[-1].replace('*', '').replace('&', '').strip()
                param_names.append(name_part)
        return f"{func_name}({', '.join(param_names)})"
    else:
        return f"{func_name}()"


def extract_callsites(func_name, callers_data):
    callsites = []

    for caller in callers_data:
        code = caller.get("code", "")
        start = 0
        while True:
            idx = code.find(func_name, start)
            if idx == -1:
                break

            paren_idx = code.find('(', idx + len(func_name))
            if paren_idx == -1:
                start = idx + len(func_name)
                continue

            if idx > 0 and (code[idx - 1].isalnum() or code[idx - 1] in ['_', '.','>']):
                start = idx + len(func_name)
                continue

            stack = 1
            i = paren_idx + 1
            while i < len(code) and stack > 0:
                if code[i] == '(':
                    stack += 1
                elif code[i] == ')':
                    stack -= 1
                i += 1

            callsites.append(code[idx:i].strip())
            start = i

    return list(set(callsites))


def extract_callee_signature(callee):
    if isinstance(callee, dict):
        code = callee.get("code", "")
        first_line = code.split("{", 1)[0].strip()
        return normalize_name(first_line)
    else:
        return normalize_name(callee)


def process_pair(unstripped_path, stripped_path, root_dir):
    print(f"\n=== Processing matched files ===")
    print(f"unstripped: {unstripped_path}")
    print(f"stripped:   {stripped_path}")

    try:
        unstripped_data = load_json(unstripped_path)
        stripped_data = load_json(stripped_path)
    except Exception as e:
        print(f"[Load failed] {e}")
        return

    unstripped_map = {item["addr"]: item for item in unstripped_data if "addr" in item}
    result = []

    for func in stripped_data:
        addr = func.get("addr")
        if addr not in unstripped_map:
            continue

        un_func = unstripped_map[addr]
        func_name = un_func.get("funname")

        callsites = extract_callsites(func_name, un_func.get("callers", []))

        callees = []
        stripped_callees = func.get("callees", [])
        unstripped_callees = un_func.get("callees", [])
        for sc, uc in zip(stripped_callees, unstripped_callees):
            sc_name = sc.get("funname") if isinstance(sc, dict) else str(sc)
            uc_signature = extract_callee_signature(uc)
            callees.append(f"{sc_name}:{uc_signature}")
        callees = list(set(callees))

        result.append({
            "addr": addr,
            "funname": func.get("funname", ""),
            "code": func.get("code", ""),
            "callsites": callsites,
            "callees": callees
        })

    out_path = (
        root_dir
        / Path(stripped_path).parts[-2]
        / (Path(stripped_path).stem.replace(".decompiled", "") + ".result")
    )
    save_json(result, str(out_path))
    print(f"Output file: {out_path}")
    print(f"Total matched functions: {len(result)}")


def gen_context(root_dir):
    unstripped_root = root_dir / "unstripped"
    stripped_root = root_dir / "stripped"


    for project_dir in unstripped_root.iterdir():
        if not project_dir.is_dir():
            continue

        project_name = project_dir.name
        stripped_project_dir = stripped_root / project_name
        if not stripped_project_dir.exists():
            print(f"[Skipped] Project not found in stripped dataset: {project_name}")
            continue

        for un_file in project_dir.rglob("*.decompiled"):
            rel_path = un_file.relative_to(project_dir)
            stripped_file = stripped_project_dir / rel_path
            if stripped_file.exists():
                process_pair(un_file, stripped_file, root_dir)
            else:
                print(f"[No matching file found] {stripped_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        required=True,
        help="Path to the binaries root directory"
    )
    args = parser.parse_args()
    gen_context(args.root_dir)
