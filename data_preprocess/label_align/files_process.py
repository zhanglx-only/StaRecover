# -*- coding: utf-8 -*-
import idaapi
import idautils
import idc
import ida_nalt
import json
import os
import time
from pathlib import Path


def is_external_or_plt_function(func_ea):
    try:
        seg = idaapi.getseg(func_ea)
        if not seg:
            return False
        seg_name = idaapi.get_segm_name(seg).lower()
        if '.plt' in seg_name or '.extern' in seg_name:
            return True
        external_segments = ['.got', '.idata', '.import']
        if any(ext_seg in seg_name for ext_seg in external_segments):
            return True
        return False
    except:
        return False


def is_compiler_generated_function(func_ea):
    if is_external_or_plt_function(func_ea):
        return True
    func_name = idc.get_func_name(func_ea)
    compiler_functions = {
        'deregister_tm_clones',
        'register_tm_clones',
        '__do_global_dtors_aux',
        '__libc_csu_init',
        '__libc_csu_fini',
        '_start',
        '_fini',
        '_init',
        '__do_global_ctors_aux',
        'frame_dummy',
        '__gmon_start__'
    }
    if func_name in compiler_functions:
        return True
    if func_name and any(func_name.startswith(prefix) for prefix in ['__', '.', '_start', '_init', '_fini']):
        return True
    return False


def decompile_function(func_ea):
    try:
        if is_external_or_plt_function(func_ea):
            return None
        if is_compiler_generated_function(func_ea):
            return None
        for attempt in range(3):
            cfunc = idaapi.decompile(func_ea)
            if cfunc:
                code = str(cfunc).strip()
                if code:
                    return code
    except:
        time.sleep(1)
    return None


def get_callers(func_ea):

    callers = []
    for xref in idautils.XrefsTo(func_ea, 0):
        caller_func = idaapi.get_func(xref.frm)
        if caller_func and not is_compiler_generated_function(caller_func.start_ea):
            code = decompile_function(caller_func.start_ea)
            if code:
                callers.append({
                    "addr": hex(caller_func.start_ea),
                    "funname": idc.get_func_name(caller_func.start_ea),
                    "code": code
                })
    return callers  


def get_callees(func_ea):

    callees = []
    for insn_ea in idautils.FuncItems(func_ea):
        if idc.print_insn_mnem(insn_ea).lower() == "call":
            target = idc.get_operand_value(insn_ea, 0)
            if idc.get_func_name(target) and not is_compiler_generated_function(target):
                code = decompile_function(target)
                if code:
                    callees.append({
                        "addr": hex(target),
                        "funname": idc.get_func_name(target),
                        "code": code
                    })
    return callees


def main():
    try:
        print("=== Starting processing (skipping compiler-generated functions) ===")
        idaapi.auto_wait()
        time.sleep(2)

        input_file = ida_nalt.get_input_file_path()
        output_file = Path(input_file).with_name(Path(input_file).name + ".decompiled")
        print(f"Output file: {output_file}")

        results = []
        total_count = 0
        user_code_count = 0
        compiler_func_count = 0

        for func_ea in idautils.Functions():
            total_count += 1
            func_name = idc.get_func_name(func_ea)

            if is_compiler_generated_function(func_ea):
                compiler_func_count += 1
                continue

            code = decompile_function(func_ea)
            if code:
                user_code_count += 1
                results.append({
                    "addr": hex(func_ea),
                    "funname": func_name,
                    "code": code,
                    "callers": get_callers(func_ea),
                    "callees": get_callees(func_ea)
                })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Processing complete:")
        print(f"Total functions: {total_count}")
        print(f"User code: {user_code_count}")
        print(f"Compiler functions skipped: {compiler_func_count}")
        print(f"Output file: {output_file}")
        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    finally:
        idaapi.qexit(0)


if __name__ == "__main__":
    main()
