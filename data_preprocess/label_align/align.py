import os
import json
import re
import argparse

def load_decompiled_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_result_list(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def find_result_file(decompiled_file, result_root):

    base_name = os.path.splitext(os.path.basename(decompiled_file))[0] + ".result"
    for root, dirs, files in os.walk(result_root):
        if base_name in files:
            return os.path.join(root, base_name)
    return None

def align_functions(un_file, stripped_file):
    un_list = load_decompiled_list(un_file)
    stripped_list = load_decompiled_list(stripped_file)

    un_dict = {f['addr']: f for f in un_list}
    stripped_dict = {f['addr']: f for f in stripped_list}

    aligned_list = []

    for addr, s_func in stripped_dict.items():
        aligned_func = s_func.copy()
        u_func = un_dict.get(addr, {})

        funname_aligned = {}
        funname_aligned[s_func.get('funname', 'unknown')] = u_func.get('funname', 'unknown')

        s_params = s_func.get('extracted_params', [])
        u_params = u_func.get('params', [])
        min_len = min(len(s_params), len(u_params))
        params_aligned = {s_params[i]: u_params[i] for i in range(min_len)}

        vars_aligned = {}
        s_vars = s_func.get('extracted_vars', [])
        u_vars = u_func.get('locals', []) + u_func.get('params', [])
        for sv in s_vars:
            sv_name = sv.get('variable', '')
            rbp_offset = sv.get('rbp_offset_int', None)
            stack_start = sv.get('stack_start', None)
            stack_end = sv.get('stack_end', None)
            matched = False
            if rbp_offset is not None:
                for uv in u_vars:
                    loc = str(uv.get('location', ''))
                    m = re.search(r'fp([+-]?\d+)', loc)
                    if m:
                        fp = int(m.group(1))
                        if rbp_offset == fp + 16:
                            vars_aligned[sv_name] = uv.get('name', 'unknown')
                            matched = True
                            break
            if not matched and stack_start is not None and stack_end is not None:
                for uv in u_vars:
                    uv_start = uv.get('stack_start')
                    uv_end = uv.get('stack_end')
                    if uv_start is not None and uv_end is not None:
                        if stack_start >= uv_start + 16 and stack_end <= uv_end + 16:
                            vars_aligned[sv_name] = '-'
                            break

        cluster_var = {}
        for uv in u_vars:
            uv_type = uv.get('type', '')
            uv_start = uv.get('stack_start')
            uv_end = uv.get('stack_end')
            cluster = []
            for sv in s_vars:
                sv_start = sv.get('stack_start')
                sv_end = sv.get('stack_end')
                if sv_start is None or sv_end is None:
                    continue
                if all(isinstance(x, int) for x in (sv_start, sv_end, uv_start, uv_end)):
                    if sv_start >= uv_start + 16 and sv_end <= uv_end + 16 and not (sv_start == uv_start + 16 and sv_end == uv_end + 16):
                        cluster.append(sv.get('variable', 'unknown'))

            if cluster:
                cluster_var[uv_type] = cluster

        aligned_func['funname_aligned'] = funname_aligned
        aligned_func['params_aligned'] = params_aligned
        aligned_func['vars_aligned'] = vars_aligned
        if cluster_var:
            aligned_func['cluster_var'] = cluster_var

        aligned_list.append(aligned_func)

    return aligned_list

def process_all(unstripped_root, stripped_root, result_root):
    for root, dirs, files in os.walk(stripped_root):
        for file in files:
            if file.endswith('.decompiled'):
                stripped_path = os.path.join(root, file)
                rel_path = os.path.relpath(stripped_path, stripped_root)
                unstripped_path = os.path.join(unstripped_root, rel_path)
                if not os.path.exists(unstripped_path):
                    print(f"[!] No matching unstripped file for {stripped_path}")
                    continue

                result_file = find_result_file(stripped_path, result_root)
                if not result_file:
                    print(f"[!] No result file for {stripped_path}")
                    continue

                aligned_data = align_functions(unstripped_path, stripped_path)

                with open(result_file, 'r', encoding='utf-8') as f:
                    result_list = json.load(f)

                result_dict = {f['addr']: f for f in result_list}
                for f in aligned_data:
                    addr = f['addr']
                    if addr in result_dict:
                        result_dict[addr].update({
                            k: f[k] for k in ['funname_aligned','params_aligned','vars_aligned','cluster_var'] if k in f
                        })

                updated_result = list(result_dict.values())
                save_result_list(result_file, updated_result)
                print(f"[+] Updated {result_file}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--UNSTRIPPED_ROOT",
        required=True,
        help="Path to the unstripped binaries root directory"
    )
    
    parser.add_argument(
        "--STRIPPED_ROOT",
        required=True,
        help="Path to the stripped binaries root directory"
    )


    parser.add_argument(
        "--result_root",
        required=True,
        help="Path to the result_root directory"
    )
    args = parser.parse_args()

    process_all(args.UNSTRIPPED_ROOT, args.STRIPPED_ROOT, args.result_root)
