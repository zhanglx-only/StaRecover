#!/bin/bash

# Set environment variables.
export ACCELERATE_LOG_LEVEL=info
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# Set CODEBERT_MODEL_PATH before launching, for example:
# export CODEBERT_MODEL_PATH=./models/codebert
# Usage: ./launch_train.sh TRAIN_DATA SAVE_DIR MODEL_DIR

# Launch training with Accelerate.
accelerate launch \
  --config_file zero3.yaml \
  train_grpo.py "$@"
