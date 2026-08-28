import argparse
import json
import re
from tqdm import tqdm


def load_json(file_path):
    """Load a JSON or JSONL file and return its records."""
    try:
        data = []
        if file_path.endswith('.json'):
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
    """Save records as JSON or JSONL."""
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
    """Replace variable names with a regular expression using exact boundaries."""
    pattern = re.compile(r'([^a-zA-Z0-9_@]|^)(%s)([^a-zA-Z0-9_@])' % re.escape(ori_variable_name))
    return pattern.sub(r'\g<1>%s\g<3>' % new_variable_name, code)


def parse_output_call(output_call_str):
    """
    Parse a string such as "sub_401302:secc,a1:code".
    Return a mapping such as {'sub_401302': 'secc', 'a1': 'code'}.
    """
    mapping = {}
    if not output_call_str:
        return mapping

    # Split multiple mappings by comma first.
    parts = output_call_str.split(',')
    for part in parts:
        if ':' in part:
            k, v = part.split(':', 1)
            mapping[k.strip()] = v.strip()
    return mapping


def norm_func_code(entry):
    """Normalize the function body using output mappings and remove comments."""
    code = entry.get("code")
    # 1. Remove comments so address offsets do not affect similarity.
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # Remove block comments.
    code = re.sub(r'//.*', '', code)  # Remove line comments.
    # 2. Parse the output_call field.
    output_call_str = entry.get("output", "")
    mapping = parse_output_call(output_call_str)
    # 3. Replace variable and function names.
    new_code = code
    for k, v in mapping.items():
        new_code = replace_variable_names(new_code, k, v)
    # 4. Normalize completely by removing whitespace and lowercasing.
    # This treats logically equivalent code with different indentation equally.
    new_code = re.sub(r'\s+', '', new_code).lower()

    return new_code


def normalize_json_codes(file_path, output_file_path):
    """Normalize code in each JSON or JSONL record and save a new file."""
    try:
        # 1. Load JSON data.
        print("Loading JSON data")
        data = load_json(file_path)
        print("JSON data loaded")

        # 2. Normalize each record's code with a progress bar.
        for entry in tqdm(data, desc="Normalizing code", unit="records"):
            if "code" in entry:  # Process records containing a code field.
                entry["norm_code"] = norm_func_code(entry)

        # 3. Save normalized data to the output file.
        save_json(data, output_file_path)
        print(f"Normalization complete; results saved to {output_file_path}")
    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize function bodies in a dataset.")
    parser.add_argument("input_file_path", help="Input JSON or JSONL file")
    parser.add_argument("output_file_path", help="Output JSON or JSONL file")
    args = parser.parse_args()
    normalize_json_codes(args.input_file_path, args.output_file_path)
