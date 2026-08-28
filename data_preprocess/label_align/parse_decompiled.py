import os
import json
import re
import argparse

TYPE_SIZE = {
    "int": 4,
    "char": 1,
    "unsigned int": 4,
    "__int64": 8,
    "__int64 *": 8,
    "__int64 **": 8,
    "unsigned __int64": 8,
    "void*": 8,
    "void *": 8,
    "char*": 8,
    "char *": 8,
    "const char *": 8,
    "pthread_t": 8,
    "struct timeval": 16,
    "double": 8,
    "float": 4,
    "_QWORD *": 8,
    "const char[]": None,
}


def get_type_size(type_str):

    type_str = type_str.strip()

    if '*' in type_str:
        return 8


    return TYPE_SIZE.get(type_str, None)

def extract_function_params(code: str) -> list:
    params = []
    code = code.split("{", 1)[0]
    start = code.find("(")
    if start == -1:
        return params

    depth = 0
    end = -1
    for i in range(start, len(code)):
        if code[i] == "(":
            depth += 1
        elif code[i] == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return params

    param_str = code[start+1:end].strip()
    if not param_str or param_str == "void":
        return params

    raw_params = []
    temp = ""
    depth = 0
    for c in param_str:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if c == "," and depth == 0:
            raw_params.append(temp.strip())
            temp = ""
        else:
            temp += c
    if temp:
        raw_params.append(temp.strip())

    for p in raw_params:
        p = p.strip()
        if not p or p == "...":
            continue
        if "(" in p:
            m_fp = re.search(r"\(\s*[^()]*\*\s*([A-Za-z_]\w*)\s*\)", p)
            if m_fp:
                params.append(m_fp.group(1))
            else:
                m = re.search(r"([A-Za-z_]\w*)\s*(?:\[\s*\])?\s*$", p)
                if m:
                    params.append(m.group(1))
                else:
                    params.append(None)
        else:
            m = re.search(r"[*\s]([A-Za-z_]\w*)\s*(?:\[\s*\])?$", p)
            if m:
                params.append(m.group(1))
            else:
                params.append(None)

    return params

def extract_vars(code: str):
    results = []
    lines = code.splitlines()

    for ln in lines:
        m = re.search(r'\[rbp([+-]?[0-9A-Fa-f]+)h\]', ln)
        if not m:
            continue

        rbp_hex = m.group(1)

        decl_part = ln.split(';')[0]

        decl_part = decl_part.strip()

        vm = re.search(r'[*\s]([A-Za-z_]\w*)(?:\s*\[.*\])?$', decl_part)
        if not vm:
            continue

        var_name = vm.group(1)

        idx = decl_part.rfind(var_name)
        var_type = decl_part[:idx].rstrip()

        sign = 1
        s = rbp_hex
        if s.startswith('+'):
            s = s[1:]
        elif s.startswith('-'):
            sign = -1
            s = s[1:]
        try:
            rbp_offset = int(s, 16) * sign
        except ValueError:
            rbp_offset = None
        size = get_type_size(var_type.strip())
        stack_start = rbp_offset
        stack_end = rbp_offset + size if size else None

        results.append({
            "type": var_type,
            "variable": var_name,
            "rbp_offset_hex": rbp_hex,
            "rbp_offset_int": rbp_offset,
            "size": size,
            "stack_start": stack_start,
            "stack_end": stack_end
        })

    return results

def process_decompiled_file(input_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        funcs = json.load(f)

    for func in funcs:
        code = func.get("code", "")
        func["extracted_params"] = extract_function_params(code)
        func["extracted_vars"] = extract_vars(code)

    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(funcs, f, indent=4, ensure_ascii=False)
    print(f"Extended decompiled info saved to {input_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--STRIPPED_ROOT",
        required=True,
        help="Path to the stripped binaries root directory"
    )
    
    args = parser.parse_args()

    for dirpath, dirnames, filenames in os.walk(args.STRIPPED_ROOT):
        for filename in filenames:
            if filename.endswith(".decompiled"):
                decompiled_path = os.path.join(dirpath, filename)
                process_decompiled_file(decompiled_path)
