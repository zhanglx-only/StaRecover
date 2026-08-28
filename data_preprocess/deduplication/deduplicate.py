import json
import argparse
from fuzzywuzzy import fuzz
from tqdm import tqdm


def get_index_features(index):
    """Read structural features used by the duplicate pre-filter."""
    index = index if isinstance(index, dict) else {}

    try:
        arg_count = int(index.get("arg_count", 0))
    except (TypeError, ValueError):
        arg_count = 0

    return_type = index.get("return_type", "")
    return_type = return_type.strip().lower() if isinstance(return_type, str) else ""

    try:
        loc = int(index.get("loc", 0))
    except (TypeError, ValueError):
        loc = 0

    return {
        "arg_count": arg_count,
        "return_type": return_type,
        "loc": loc,
    }


def fast_rough_filter(feat_a, feat_b):
    """Apply a fast pre-filter based on index features."""
    if feat_a["arg_count"] != feat_b["arg_count"]:
        return False
    if feat_a["return_type"] != feat_b["return_type"]:
        return False

    max_loc = max(feat_a["loc"], feat_b["loc"], 1)
    if abs(feat_a["loc"] - feat_b["loc"]) / max_loc > 0.3:
        return False

    return True


def similarity(code_a, code_b, index_a=None, index_b=None):
    features_a = get_index_features(index_a)
    features_b = get_index_features(index_b)

    if not fast_rough_filter(features_a, features_b):
        return False

    return fuzz.ratio(code_a, code_b) >= 90

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

def process_duplicates(data):
    """
    Deduplicate records:
    1. Use names in "funname_aligned" to identify candidates.
    2. Remove duplicates when same-name "norm_code" similarity exceeds 90.
    """
    # Keep all retained representatives for each name, not only the first one.
    seen = {}
    unique_data = []

    print(f"Records before deduplication: {len(data)}")

    for entry in tqdm(data):
        funname_values = list(entry.get("funname_aligned", {}).values())  # Get names.
        norm_code = entry.get("norm_code")

        if not funname_values or not norm_code:
            continue
        # For each function name, compare against every retained representative.
        match_found = False
        for funname in funname_values:
            if funname == "main":  # Skip the main function.
                continue
            if funname in seen:
                for existing_entry in seen[funname]:
                    similarity_flag = similarity(
                        norm_code,
                        existing_entry.get("norm_code", ""),
                        index_a=entry.get("index"),
                        index_b=existing_entry.get("index"),
                    )
                    if similarity_flag:
                        match_found = True
                        break
            if match_found:
                break

        if not match_found:
            unique_data.append(entry)
            # Register a record only after it is retained for later comparisons.
            for funname in funname_values:
                if funname != "main":
                    seen.setdefault(funname, []).append(entry)

    print(f"Records after deduplication: {len(unique_data)}")
    return unique_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate normalized code records.")
    parser.add_argument("input_file_path", help="Input JSON or JSONL file")
    parser.add_argument("output_file_path", help="Output JSON or JSONL file")
    args = parser.parse_args()

    print("Loading JSON data")
    data = load_json(args.input_file_path)
    print("JSON data loaded")

    deduplicated_data = process_duplicates(data)
    save_json(deduplicated_data, args.output_file_path)
