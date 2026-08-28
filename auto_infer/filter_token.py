"""Stage 4: filter empty-label and over-length inference records."""

from __future__ import annotations

import argparse

from common import load_records, save_jsonl


def clean_dataset(
    input_path: str,
    output_path: str,
    tokenizer_path: str,
    input_field: str = "input",
    label_field: str = "output",
    max_len: int = 16384,
) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    records = load_records(input_path)
    cleaned = []
    dropped_for_length = 0
    dropped_for_empty_label = 0

    for index, record in enumerate(records, 1):
        prompt = record.get(input_field, "")
        label = record.get(label_field, "")
        if not isinstance(label, str) or not label.strip():
            dropped_for_empty_label += 1
            continue
        if not isinstance(prompt, str):
            raise TypeError(f"Record {index} field {input_field!r} is not a string")
        token_ids = tokenizer.encode(
            prompt + label, add_special_tokens=False, truncation=False
        )
        if len(token_ids) > max_len:
            dropped_for_length += 1
            continue
        cleaned.append(record)

    save_jsonl(output_path, cleaned)
    print("========== Stage 4 complete ==========")
    print(f"Total: {len(records)}")
    print(f"Kept: {len(cleaned)}")
    print(f"Dropped for length: {dropped_for_length}")
    print(f"Dropped for empty {label_field!r}: {dropped_for_empty_label}")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("tokenizer_path")
    parser.add_argument("--input-field", default="input")
    parser.add_argument("--label-field", default="output")
    parser.add_argument("--max-len", type=int, default=16384)
    args = parser.parse_args()
    clean_dataset(**vars(args))


if __name__ == "__main__":
    main()
