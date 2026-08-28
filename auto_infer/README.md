# StaRecover Automated Inference

The main script is `auto_infer.py`. It accepts JSON arrays, single JSON
objects, and JSONL files, detecting the format automatically. The original
input file is never modified. Stage 0 first applies token filtering to the
input; stages 1-6 process only the filtered data. All intermediate files and
the final result are written as JSONL beside the input file. vLLM is loaded
once and the same model instance is shared by stage 1 and every stage-5 run.

## Usage example

```bash
python ./auto_infer.py \
  ./test/sample.json \
  --model-path ./models/StaRecover \
  --loops 3 \
  --gpus 1 \
  --batch-size 2 \
  --gpu_memory_utilization 0.5
```

## All parameters

| Parameter | Default | Description |
|---|---:|---|
| `input_path` | Required | Path to the initial JSON/JSONL file, for example `./test/sample.jsonl`. |
| `--model-path PATH` | Required | Model path shared by stages 1 and 5 and used as the stage-4 tokenizer path, for example `./models/StaRecover`. The model is loaded only once. |
| `--loops N` | `1` | Number of inference loops. Stages 0-2 run once; the first `N-1` loops run 3 -> 4 -> 5 -> 6, and the final loop runs only 3 -> 4 -> 5. With `0`, only stages 0-2 run. |
| `--input-field NAME` | `input` | Record field used to store and read prompts. Any field name is accepted, for example `input_new6`. |
| `--output-field NAME` | `predict` | Record field used to store model predictions, not an output file path. Any field name is accepted, for example `predict(input_new6)`. |
| `--label-field NAME` | `output` | Reference-label field used to obtain the `function-name:` prefix, check for empty labels, and calculate filtering length. Its content must include a colon. |
| `--summary-field NAME` | `summary` | Record field containing the summary or teacher hints read by stage 3 for context prompts. |
| `--batch-size N` | `8192` | vLLM inference batch size for stages 1 and 5. Reduce it when GPU memory is insufficient. |
| `--gpu-memory-utilization FLOAT` | `0.9` | Fraction of GPU memory available to vLLM, in `(0, 1]`. It can also be written as `--gpu_memory_utilization`. |
| `--max-input-tokens N` | `16384` | Maximum token count allowed by stages 0 and 4. The count includes the prompt from `input-field` and the label from `label-field`; records exceeding it are filtered out. |
| `--max-output-tokens N` | `16384` | Maximum number of tokens generated for each record in stages 1 and 5. |
| `--gpus GPU_IDS` | Inherited environment | Physical GPU IDs, for example `--gpus 0,1,2,3` or `--gpus 1,3,5`. This also sets `CUDA_VISIBLE_DEVICES`; the GPU count is inferred from the list. |
| `--cuda-visible-devices GPU_IDS` | Inherited environment | Compatibility alias for `--gpus`. |
| `--num-gpus N` | Inferred or `1` | Explicit vLLM GPU count. When `--gpus` is supplied, the count must match the number of listed IDs. |
| `--tensor-parallel-size N` | GPU count | Advanced vLLM tensor-parallel size. When combined with `--gpus` or `--num-gpus`, all counts must match. |
| `--delete-intermediate` | Keep files | Delete numbered intermediate files after successful completion, retaining only the original input and the final `.final.jsonl`. Failed runs do not delete files. |
| `--overwrite` | Keep existing files | Delete this pipeline's existing intermediate and final files before starting. The original input is never deleted. |
| `--dry-run` | Disabled | Print the configuration, commands, and output paths without running inference or creating, replacing, or deleting files. |
| `-h` / `--help` | N/A | Show command-line help. |

## Pipeline

| Stage | Script | Frequency | Purpose |
|---:|---|---|---|
| 0 | `filter_token.py` | Once | Before loading the model, filter records with empty labels or prompts exceeding the token limit. |
| 1 | `vllm_infer.py` | Once | Run anchor-function inference for records containing a `rag` field. Records without `rag` are copied without a prediction field. |
| 2 | `prop_callees.py` | Once | Read the stage-1 predictions and stage-0 filtered data, then propagate anchor-function information through the complete filtered dataset. |
| 3 | `prop_prompt.py` | Each loop | Generate an RAG prompt for records with `rag`; generate a context prompt containing the summary for all other records. |
| 4 | `filter_token.py` | Each loop | Filter empty labels and over-length prompts. Records removed here do not enter the current or later inference stages. |
| 5 | `vllm_infer.py` | Each loop | Run inference for user functions, using the same inference implementation as stage 1. |
| 6 | `prop_context.py` | First `N-1` loops only | Rebuild call-site and callee context from the current loop's predictions. The final loop skips this stage. |

### Loop examples

| Option | Actual pipeline |
|---|---|
| `--loops 0` | 0 -> 1 -> 2; stage 2 is the final result. |
| `--loops 1` | 0 -> 1 -> 2 -> 3 -> 4 -> 5; stage 6 is skipped. |
| `--loops 2` | 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 3 -> 4 -> 5. |
| `--loops 3` | 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 3 -> 4 -> 5 -> 6 -> 3 -> 4 -> 5. |

## File names

For `./test/sample.json` with `--loops 2`, the pipeline creates:

| File | Meaning |
|---|---|
| `sample.00_filtered.jsonl` | Stage-0 filtered data; subsequent stages use this as their initial dataset. |
| `sample.01_anchor_prediction.jsonl` | Stage-1 output. All stage-0 records are retained; only records with `rag` receive predictions. |
| `sample.02_anchor_context.jsonl` | Stage-2 output. |
| `sample.loop01.03_prompt.jsonl` | Loop-1 prompts. |
| `sample.loop01.04_filtered.jsonl` | Loop-1 filtered data. |
| `sample.loop01.05_prediction.jsonl` | Loop-1 inference output. |
| `sample.loop01.06_context.jsonl` | Loop-1 propagated context. |
| `sample.loop02.03_prompt.jsonl` | Loop-2 prompts. |
| `sample.loop02.04_filtered.jsonl` | Loop-2 filtered data. |
| `sample.loop02.05_prediction.jsonl` | Final-loop inference output. |
| `sample.final.jsonl` | Final result copied from the last prediction file. |

The final loop does not create `sample.loop02.06_context.jsonl`. Intermediate
files are kept by default; `--delete-intermediate` retains only `sample.json`
and `sample.final.jsonl`.

## Common commands

```bash
# Full inference: four GPUs, three loops, batch size 8192
python ./auto_infer.py \
  ./test/sample.jsonl \
  --model-path ./models/StaRecover \
  --loops 3 \
  --gpus 0,1,2,3 \
  --batch-size 8192

# Start from the original input and delete intermediate files on success
python ./auto_infer.py \
  ./test/sample.jsonl \
  --model-path ./models/StaRecover \
  --gpus 0,1,2,3 \
  --overwrite \
  --delete-intermediate

# Inspect configuration and execution paths without running inference
python ./auto_infer.py \
  ./test/sample.jsonl \
  --model-path ./models/StaRecover \
  --gpus 0,1,2,3 \
  --dry-run
```
