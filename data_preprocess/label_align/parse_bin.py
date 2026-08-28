# -*- coding: utf-8 -*-
import os
import json
from elftools.elf.elffile import ELFFile
from elftools.dwarf.descriptions import describe_attr_value
import re
import argparse

DEFAULT_TYPE_SIZES = {
    "char": 1, "signed char": 1, "unsigned char": 1,
    "short": 2, "short int": 2, "unsigned short": 2, "unsigned short int": 2,
    "int": 4, "signed int": 4, "unsigned int": 4, "uint32_t": 4,
    "long": 8, "long int": 8, "unsigned long": 8, "unsigned long int": 8,
    "long long": 8, "long long int": 8, "unsigned long long": 8, "unsigned long long int": 8,
    "size_t": 8, "ssize_t": 8, "float": 4, "double": 8, "bool": 1,
    "pthread_t": 8, "pthread_mutex_t": 40, "pthread_cond_t": 48,
}

def get_type_name(type_die, dwarfinfo):
    if type_die is None:
        return "unknown"
    tag = type_die.tag
    if tag == "DW_TAG_base_type":
        name_attr = type_die.attributes.get("DW_AT_name")
        return name_attr.value.decode() if name_attr else "unknown"
    if tag == "DW_TAG_typedef":
        name_attr = type_die.attributes.get("DW_AT_name")
        if name_attr:
            return name_attr.value.decode()
        base_type_attr = type_die.attributes.get("DW_AT_type")
        if base_type_attr:
            base_die = type_die.get_DIE_from_attribute("DW_AT_type")
            return get_type_name(base_die, dwarfinfo)
        return "unknown"
    if tag == "DW_TAG_pointer_type":
        base_type_attr = type_die.attributes.get("DW_AT_type")
        if base_type_attr:
            base_die = type_die.get_DIE_from_attribute("DW_AT_type")
            return get_type_name(base_die, dwarfinfo) + "*"
        return "void*"
    if tag == "DW_TAG_const_type":
        base_type_attr = type_die.attributes.get("DW_AT_type")
        if base_type_attr:
            base_die = type_die.get_DIE_from_attribute("DW_AT_type")
            base_name = get_type_name(base_die, dwarfinfo)
            return base_name if base_name.startswith("const ") else "const " + base_name
        return "const unknown"
    if tag == "DW_TAG_structure_type":
        name_attr = type_die.attributes.get("DW_AT_name")
        return "struct " + (name_attr.value.decode() if name_attr else "anon_struct")
    if tag == "DW_TAG_union_type":
        name_attr = type_die.attributes.get("DW_AT_name")
        return "union " + (name_attr.value.decode() if name_attr else "anon_union")
    if tag == "DW_TAG_array_type":
        base_type_attr = type_die.attributes.get("DW_AT_type")
        if base_type_attr:
            base_die = type_die.get_DIE_from_attribute("DW_AT_type")
            return get_type_name(base_die, dwarfinfo) + "[]"
        return "unknown[]"
    return "unknown"

def resolve_type_name_and_size(type_attr, die, dwarfinfo):

    if not type_attr:
        return "unknown", None
    try:
        type_die = die.get_DIE_from_attribute("DW_AT_type")
        if type_die is None:
            return "unknown", None

        tag = type_die.tag

        if tag == "DW_TAG_base_type":
            name_attr = type_die.attributes.get("DW_AT_name")
            type_name = name_attr.value.decode() if name_attr else "unknown"
            size_attr = type_die.attributes.get("DW_AT_byte_size")
            size = size_attr.value if size_attr else DEFAULT_TYPE_SIZES.get(type_name)
            return type_name, size

        if tag == "DW_TAG_typedef":
            name_attr = type_die.attributes.get("DW_AT_name")
            type_name = name_attr.value.decode() if name_attr else "unknown"
            base_type_attr = type_die.attributes.get("DW_AT_type")
            _, size = resolve_type_name_and_size(base_type_attr, type_die, dwarfinfo)
            return type_name, size

        if tag == "DW_TAG_const_type":
            base_type_attr = type_die.attributes.get("DW_AT_type")
            base_name, size = resolve_type_name_and_size(base_type_attr, type_die, dwarfinfo)
            type_name = base_name if base_name.startswith("const ") else "const " + base_name
            return type_name, size

        if tag == "DW_TAG_pointer_type":
            base_type_attr = type_die.attributes.get("DW_AT_type")
            base_name, _ = resolve_type_name_and_size(base_type_attr, type_die, dwarfinfo) if base_type_attr else ("void", None)
            type_name = base_name + "*"
            size = 8  
            return type_name, size

        if tag == "DW_TAG_array_type":
            base_type_attr = type_die.attributes.get("DW_AT_type")
            base_name, elem_size = resolve_type_name_and_size(base_type_attr, type_die, dwarfinfo) if base_type_attr else ("unknown", None)
            type_name = base_name
            counts = []
            for child in type_die.iter_children():
                if child.tag == "DW_TAG_subrange_type":
                    upper = child.attributes.get("DW_AT_upper_bound")
                    count = int(upper.value) + 1 if upper else 1
                    counts.append(count)
            if not counts:
                counts = [1]
            type_name += "[]" * len(counts)
            total_count = 1
            for c in counts:
                total_count *= c
            size = elem_size * total_count if elem_size else None
            return type_name, size

        if tag == "DW_TAG_structure_type":
            name_attr = type_die.attributes.get("DW_AT_name")
            type_name = "struct " + (name_attr.value.decode() if name_attr else "anon_struct")
            size_attr = type_die.attributes.get("DW_AT_byte_size")
            size = size_attr.value if size_attr else None
            return type_name, size

        if tag == "DW_TAG_union_type":
            name_attr = type_die.attributes.get("DW_AT_name")
            type_name = "union " + (name_attr.value.decode() if name_attr else "anon_union")
            size_attr = type_die.attributes.get("DW_AT_byte_size")
            size = size_attr.value if size_attr else None
            return type_name, size

        type_name = get_type_name(type_die, dwarfinfo)
        byte_size_attr = type_die.attributes.get("DW_AT_byte_size")
        size = byte_size_attr.value if byte_size_attr else DEFAULT_TYPE_SIZES.get(type_name)
        return type_name, size

    except Exception:
        return "unknown", None


def parse_dwarf(binary_path):

    funcs = {}

  

    with open(binary_path, "rb") as f:
        elffile = ELFFile(f)
        if not elffile.has_dwarf_info():
            print("Binary has no DWARF info")
            return funcs

        dwarfinfo = elffile.get_dwarf_info()

        for CU in dwarfinfo.iter_CUs():
            for DIE in CU.iter_DIEs():
                if DIE.tag != "DW_TAG_subprogram":
                    continue

                low_pc_attr = DIE.attributes.get("DW_AT_low_pc")
                if not low_pc_attr:
                    continue
                addr = low_pc_attr.value
                func_name_attr = DIE.attributes.get("DW_AT_name")
                func_name = (
                    func_name_attr.value.decode("utf-8", "ignore")
                    if func_name_attr
                    else None
                )

                funcs[addr] = {
                    "name": func_name,
                    "params": [],
                    "locals": [],
                }

                for child in DIE.iter_children():
                    if child.tag == "DW_TAG_formal_parameter":
                        n = child.attributes.get("DW_AT_name")
                        loc = child.attributes.get("DW_AT_location")
                        type_attr = child.attributes.get("DW_AT_type")
                        if not n:
                            continue

                        param_name = n.value.decode("utf-8", "ignore")
                        param_type, param_size = resolve_type_name_and_size(type_attr, child, dwarfinfo)

                        param_info = {
                            "name": param_name,
                            "type": param_type,
                            "size": param_size,
                            "raw_loc": describe_attr_value(loc, child, dwarfinfo) if loc else None,
                            "location": None,
                            "location_type": "unknown",
                            "stack_start": None,
                            "stack_end": None,
                        }

                        if loc:
                            loc_str = describe_attr_value(loc, child, dwarfinfo)
                            # DW_OP_fbreg
                            if "DW_OP_fbreg" in loc_str and param_size:
                                m_fb = re.search(r"DW_OP_fbreg[: ]+(-?\d+)", loc_str)
                                if m_fb:
                                    offset = int(m_fb.group(1))
                                    param_info["location"] = f"fp{offset:+d}"
                                    param_info["location_type"] = "stack"
                                    param_info["stack_start"] = offset
                                    param_info["stack_end"] = offset + param_size
                            # DW_OP_reg
                            elif "DW_OP_reg" in loc_str:
                                m_reg = re.search(r"(DW_OP_reg\d+)", loc_str)
                                if m_reg:
                                    param_info["location"] = m_reg.group(1)
                                    param_info["location_type"] = "reg"
                            # DW_OP_addr
                            elif "DW_OP_addr" in loc_str:
                                m_addr = re.search(r"DW_OP_addr[: ]*(0x[0-9a-fA-F]+|[0-9a-fA-F]+)", loc_str)
                                if m_addr:
                                    addr_val = m_addr.group(1)
                                    if not addr_val.startswith("0x"):
                                        addr_val = "0x" + addr_val
                                    param_info["location"] = f"addr_{addr_val}"
                                    param_info["location_type"] = "global"

                        funcs[addr]["params"].append(param_info)

                    elif child.tag == "DW_TAG_variable":
                        n = child.attributes.get("DW_AT_name")
                        loc = child.attributes.get("DW_AT_location")
                        type_attr = child.attributes.get("DW_AT_type")
                        if not (n and loc):
                            continue

                        var_name = n.value.decode("utf-8", "ignore")
                        var_type, var_size = resolve_type_name_and_size(type_attr, child, dwarfinfo)
                        loc_str = describe_attr_value(loc, child, dwarfinfo)

                        var_info = {
                            "name": var_name,
                            "type": var_type,
                            "size": var_size,
                            "raw_loc": loc_str,
                            "location": None,
                            "location_type": "unknown",
                            "stack_start": None,
                            "stack_end": None,
                        }

                        if "DW_OP_fbreg" in loc_str and var_size:
                            m_fb = re.search(r"DW_OP_fbreg[: ]+(-?\d+)", loc_str)
                            if m_fb:
                                offset = int(m_fb.group(1))
                                var_info["location"] = f"fp{offset:+d}"
                                var_info["location_type"] = "stack"
                                var_info["stack_start"] = offset
                                var_info["stack_end"] = offset + var_size
                        elif "DW_OP_reg" in loc_str:
                            m_reg = re.search(r"(DW_OP_reg\d+)", loc_str)
                            if m_reg:
                                var_info["location"] = m_reg.group(1)
                                var_info["location_type"] = "reg"
                        elif "DW_OP_addr" in loc_str:
                            m_addr = re.search(r"DW_OP_addr[: ]*(0x[0-9a-fA-F]+|[0-9a-fA-F]+)", loc_str)
                            if m_addr:
                                addr_val = m_addr.group(1)
                                if not addr_val.startswith("0x"):
                                    addr_val = "0x" + addr_val
                                var_info["location"] = f"addr_{addr_val}"
                                var_info["location_type"] = "global"

                        funcs[addr]["locals"].append(var_info)

    return funcs

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

def update_decompiled_with_dwarf(unstripped_root):
    for root, dirs, files in os.walk(unstripped_root):
        for file in files:
            if not file.endswith(".decompiled"):
                continue
            decompiled_path = os.path.join(root, file)
            elf_path = os.path.splitext(decompiled_path)[0]
            if not os.path.exists(elf_path):
                print(f"[!] ELF file not found for {decompiled_path}")
                continue

            dwarf_funcs = parse_dwarf(elf_path)

            dwarf_addr_map = {f"0x{addr:x}": info for addr, info in dwarf_funcs.items()}

            try:
                with open(decompiled_path, "r", encoding="utf-8") as f:
                    funcs_data = json.load(f)

                for func in funcs_data:
                    func_addr = func.get("addr")
                    code = func.get("code", "")
                    if func_addr and func_addr in dwarf_addr_map:
                        func["extracted_params"] = extract_function_params(code)
                        func["params"] = dwarf_addr_map[func_addr]["params"]
                        func["locals"] = dwarf_addr_map[func_addr]["locals"]


                with open(decompiled_path, "w", encoding="utf-8") as f:
                    json.dump(funcs_data, f, indent=4, ensure_ascii=False)

                print(f"[+] Updated {decompiled_path}")

            except Exception as e:
                print(f"[!] Failed to update {decompiled_path}: {e}")



def clean_unstripped_files(unstripped_root):

    for root, dirs, files in os.walk(unstripped_root):
        for file in files:
            if not file.endswith('.decompiled'):
                continue

            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[!] Failed to load {file_path}: {e}")
                continue

            if not isinstance(data, list):
                print(f"[!] Unexpected format in {file_path}, skipping")
                continue

            cleaned_data = [f for f in data if 'params' in f and 'locals' in f]

            if len(cleaned_data) != len(data):

                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(cleaned_data, f, indent=4, ensure_ascii=False)
                    print(f"[+] Cleaned {file_path}, removed {len(data) - len(cleaned_data)} entries")
                except Exception as e:
                    print(f"[!] Failed to write {file_path}: {e}")
            else:

                print(f"[=] {file_path} OK, no entries removed")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--UNSTRIPPED_ROOT",
        required=True,
        help="Path to the unstripped binaries root directory"
    )
    args = parser.parse_args() 
    update_decompiled_with_dwarf(args.UNSTRIPPED_ROOT)
    print("[+] All .decompiled files updated with DWARF params and locals")
    clean_unstripped_files(args.UNSTRIPPED_ROOT)

