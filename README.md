# Stopping the Snowball: Mitigating Cascade Errors in Iterative Variable Name Recovery for Stripped Binaries

This repository contains the source code, dataset, trained model, configuration file and README file for StaRecover, a cascade-aware framework for iterative variable name recovery in stripped binaries.

## Environment setup

```bash
conda create -n starecover python=3.9
conda activate starecover
python -m pip install -r ./requirements.txt
```

For GRPO training, install the additional packages listed in
`./train/grpo/requirements_grpo.txt`.

## Data preprocessing

Run the following stages with explicit input and output locations.

### 1. Analyze binaries with IDA Pro

```bash
python ./data_preprocess/label_align/batch_process_all_files.py \
  --STRIPPED_ROOT ./data/stripped \
  --UNSTRIPPED_ROOT ./data/unstripped \
  --IDA_PATH ./tools/ida \
  --IDA_SCRIPT_PATH ./tools/ida_script.py
```

### 2. Parse decompiled code and debug information

```bash
python ./data_preprocess/label_align/parse_decompiled.py \
  --STRIPPED_ROOT ./data/stripped

python ./data_preprocess/label_align/parse_bin.py \
  --UNSTRIPPED_ROOT ./data/unstripped
```

### 3. Generate context and align labels

```bash
python ./data_preprocess/label_align/gen_gt_context.py \
  --root_dir ./data

python ./data_preprocess/label_align/align.py \
  --UNSTRIPPED_ROOT ./data/unstripped \
  --STRIPPED_ROOT ./data/stripped \
  --result_root ./data/aligned

python ./data_preprocess/label_align/gen_train_data.py \
  --input_directory ./data/aligned \
  --output_file ./data/functions.json
```

### 4. Normalize, index, and deduplicate

```bash
python ./data_preprocess/deduplication/normalize_function_body.py \
  ./data/functions.json ./data/functions_norm.json

python ./data_preprocess/deduplication/build_index.py \
  ./data/functions_norm.json ./data/functions_index.json

python ./data_preprocess/deduplication/deduplicate.py \
  ./data/functions_index.json ./data/functions_unique.json
```

### 5. Prepare RAG data

```bash
python ./data_preprocess/rag/rag_norm_func_body.py \
  ./data/functions_unique.json ./data/functions_rag_norm.json

python ./data_preprocess/rag/rag_simi.py \
  ./data/rag_candidates.json \
  ./data/functions_rag_norm.json \
  ./data/functions_rag.json
```

### 6. Generate summary prompts and summaries

```bash
python ./data_preprocess/gen_summary/gen_analysis.py ./data/results --workers 8

python ./data_preprocess/gen_summary/gen_domain_summary_prompt.py \
  ./data/domain_records.json

python ./data_preprocess/gen_summary/gen_domain_summary.py \
  ./data/domain_records.json --model ./models/summary-model

python ./data_preprocess/gen_summary/gen_identifier_summary_prompt.py \
  ./data/functions_unique.json ./data/domain_records.json

python ./data_preprocess/gen_summary/gen_identifier_summary.py \
  ./data/functions_unique.json --model ./models/summary-model
```

The summary-generation scripts require explicit files and model paths. Use
`--dry-run` to validate records without loading a model or writing changes.

## Training

### Supervised fine-tuning

```bash
accelerate launch ./train/sft/train.py \
  ./data/train.json \
  ./data/eval.json \
  ./models/output \
  ./models/base
```

### GRPO training

```bash
bash ./train/grpo/launch_train.sh \
  ./data/train.json \
  ./models/output \
  ./models/base
```

Set `CODEBERT_MODEL_PATH` when the reward model uses a local CodeBERT
checkpoint. For example:

```bash
export CODEBERT_MODEL_PATH=./models/codebert
```

## Inference

The automated pipeline is implemented in `./auto_infer/auto_infer.py`.
It accepts JSON or JSONL input, keeps intermediate files beside the input, and
requires an explicit model path.

```bash
python ./auto_infer/auto_infer.py \
  ./data/test.jsonl \
  --model-path ./models/starecover \
  --loops 4 \
  --gpus 0,1
```

Use `--dry-run` to inspect the planned commands. See
`./auto_infer/README.md` for the complete option and pipeline reference.

## Evaluation

```bash
python ./evaluation/eval.py ./data/predictions.jsonl
```

## Dataset and Model Available
- Dataset. The dataset used in this study is derived from the publicly available [ReSym](https://zenodo.org/records/13923982) dataset, which is a state-of-the-art method for the variable name recovery task.
- Model. Due to the large size of the StaRecover model, we have hosted it in an anonymous [Zenodo](https://zenodo.org/records/22135763) repository with a timestamp for review by the paper reviewers. 

