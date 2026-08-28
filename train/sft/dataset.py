import json
import torch
import random
from concurrent.futures import ThreadPoolExecutor
 
random.seed(1234)


def _encode_item(item, tokenizer, max_len, truncat):
    """Tokenize and pad/truncate one record.

    Kept at module level so it can be reused by the threaded loader.  Returning
    ``None`` preserves Dataset's behavior for records without ``input``.
    """
    if "input" not in item:
        return None

    input_text = item["input"]
    output_text = item["output"]

    inputs = tokenizer.encode(input_text)
    outputs = tokenizer.encode(output_text + tokenizer.eos_token)
    all_input = inputs + outputs
    cur_len = len(all_input)

    if not truncat and cur_len > max_len:
        return None

    if cur_len < max_len:
        input_id = [int(x) for x in all_input]
        input_id.extend([tokenizer.eos_token_id] * (max_len - cur_len))
        label = [-100] * len(inputs) + outputs + [-100] * (max_len - cur_len)
        attention_mask = [1] * cur_len + [0] * (max_len - cur_len)
    else:
        input_id = all_input[:max_len]
        label = ([-100] * len(inputs) + outputs)[:max_len]
        attention_mask = [1] * max_len

    return {
        "input_ids": torch.LongTensor(input_id),
        "labels": torch.LongTensor(label),
        "attention_mask": torch.LongTensor(attention_mask),
    }


def _batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _iter_json_array(fp, chunk_size=1024 * 1024):
    """Yield objects from a top-level JSON array without loading the file at once."""
    decoder = json.JSONDecoder()
    buffer = ""
    pos = 0
    started = False
    need_value = True
    eof = False

    while True:
        if not eof:
            chunk = fp.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True

        while True:
            while pos < len(buffer) and buffer[pos].isspace():
                pos += 1

            if not started:
                if pos >= len(buffer):
                    break
                if buffer[pos] != "[":
                    raise ValueError("JSON dataset must contain a top-level array")
                started = True
                pos += 1
                continue

            if need_value:
                if pos >= len(buffer):
                    break
                if buffer[pos] == "]":
                    return
                try:
                    item, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    break
                yield item
                pos = end
                need_value = False
            else:
                if pos >= len(buffer):
                    break
                if buffer[pos] == ",":
                    pos += 1
                    need_value = True
                elif buffer[pos] == "]":
                    return
                else:
                    raise ValueError("Invalid separator in JSON array")

        if pos:
            buffer = buffer[pos:]
            pos = 0
        if eof:
            raise ValueError("Unexpected end of JSON array")


def _parallel_encode(items, tokenizer, max_len, truncat, num_workers, batch_size):
    """Encode records concurrently while yielding results in input order."""
    if num_workers <= 1:
        for item in items:
            encoded = _encode_item(item, tokenizer, max_len, truncat)
            if encoded is not None:
                yield encoded
        return

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for batch in _batched(items, batch_size):
            for encoded in executor.map(
                _encode_item,
                batch,
                [tokenizer] * len(batch),
                [max_len] * len(batch),
                [truncat] * len(batch),
            ):
                if encoded is not None:
                    yield encoded
 
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path,
        tokenizer,
        max_len=2048,
        shuffle=False,
        max_cnt=None,
        truncat=True,
        num_workers=8,
        batch_size=1024,
    ):
        self.data = []
        with open(file_path, 'r') as fp:
            self.data.extend(
                _parallel_encode(
                    _iter_json_array(fp),
                    tokenizer,
                    max_len,
                    truncat,
                    num_workers,
                    batch_size,
                )
            )
        if max_cnt is not None:
            self.data = self.data[: max_cnt]
        if shuffle:
            random.shuffle(self.data)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(file_path, 'loaded:', len(self.data))
 
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]


class Dataset1(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path,
        tokenizer,
        max_len=2048,
        shuffle=False,
        max_cnt=None,
        truncat=True,
        num_workers=8,
        batch_size=1024,
    ):
        self.data = []

        with open(file_path, "r", encoding="utf-8") as fp:
            self.data.extend(
                _parallel_encode(
                    self._iter_records(fp, file_path),
                    tokenizer,
                    max_len,
                    truncat,
                    num_workers,
                    batch_size,
                )
            )

        if max_cnt is not None:
            self.data = self.data[:max_cnt]

        if shuffle:
            random.shuffle(self.data)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(file_path, "loaded:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def _iter_records(fp, file_path):
        """Parse JSONL lazily so raw records are not duplicated in memory."""
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{file_path} line {line_no} is not valid JSON: {e}"
                ) from e
