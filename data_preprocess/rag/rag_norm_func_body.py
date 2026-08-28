import argparse
import json
import re
from tqdm import tqdm  # Progress bar.


def load_json(file_path):
    """Load a JSON or JSONL file and return a list of records."""
    try:
        data = []
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif file_path.endswith('.result'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line.strip()) for line in f if line.strip()]
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def save_json(data, file_path):
    """Save records to a JSON or JSONL file."""
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        elif file_path.endswith('.jsonl'):
            with open(file_path, 'w', encoding='utf-8') as f:
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"Updated data saved to {file_path}")
    except Exception as e:
        print(f"Error saving {file_path}: {e}")


def replace_variable_names(code, ori_variable_name, new_variable_name):
    """Replace a variable name with a regex while preserving token boundaries."""
    pattern = re.compile(r'([^a-zA-Z0-9_@]|^)(%s)([^a-zA-Z0-9_@])' % re.escape(ori_variable_name))
    return pattern.sub(r'\g<1>%s\g<3>' % new_variable_name, code)

def norm_func_code(entry):
    code = entry.get("code")
    # 1. Remove comments so addresses do not affect similarity.
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # Block comments.
    code = re.sub(r'//.*', '', code)  # Line comments.
    new_code = code
    # IDA-generated function names (for example sub_401568, j_sub_401568, nullsub_1).
    new_code = re.sub(r'\b(?:j_sub_|sub_|nullsub_)[0-9a-fA-F]+\b', '<fun>', new_code)
    # IDA-generated code labels (for example loc_401000, locret_401020).
    new_code = re.sub(r'\b(?:loc_|locret_)[0-9a-fA-F]+\b', '<label>', new_code)
    # IDA-generated global object names, such as xmmword_403000 or dword_40B2F0.
    new_code = re.sub(
        r'\b(?:zmmword_|ymmword_|xmmword_|oword_|tbyte_|qword_|dword_|word_|byte_|'
        r'unk_|off_|asci_|asc_|stru_|flt_|dbl_)[0-9a-fA-F]+\b',
        '<global_var>',
        new_code,
    )
    # Normalize hexadecimal addresses/constants (for example 0x401568 or 0x1000).
    new_code = re.sub(r'\b0[xX][0-9a-fA-F]+\b', '<hex>', new_code)
    # IDA hexadecimal literals with an h suffix (for example 401000h or 0FFh).
    new_code = re.sub(r'\b[0-9a-fA-F]+[hH]\b', '<hex>', new_code)
    # Remove whitespace and lowercase so formatting differences do not affect matching.
    new_code = re.sub(r'\s+', '', new_code).lower()

    return new_code


def normalize_json_codes(file_path, output_file_path):
    """Normalize every code record in a JSON or JSONL file and save the result."""
    try:
        # 1. Load JSON data.
        print("Loading JSON data")
        data = load_json(file_path)
        print("JSON data loaded")

        # 2. Normalize each code record with a progress bar.
        for entry in tqdm(data, desc="Normalizing code", unit="record"):
            if "code" in entry:  # Require a code field.
                entry["rag_norm_code"] = norm_func_code(entry)

        # 3. Save normalized data.
        save_json(data, output_file_path)
        print(f"Normalization complete; result saved to {output_file_path}")
    except Exception as e:
        print(f"Processing failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize code records for RAG retrieval."
    )
    parser.add_argument("input_file", help="Input JSON or JSONL file (for example ./data/input.json)")
    parser.add_argument("output_file", help="Output JSON or JSONL file (for example ./data/output.json)")
    args = parser.parse_args()
    normalize_json_codes(args.input_file, args.output_file)
