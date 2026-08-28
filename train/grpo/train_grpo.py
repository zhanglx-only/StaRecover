import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
from reward_model import grpo_hybrid_reward
from typing import List, Dict, Any


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSON or JSONL file and return a list of records."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":  # JSON array.
            data = json.load(f)
        else:  # JSONL, one JSON object per line.
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    # data = data[int(len(data) * 0.2):]
    return data


def prepare_dataset_for_grpo(data_path):
    # Load a JSON or JSONL dataset.
    data = load_json_or_jsonl(data_path)

    formatted_data = []

    for item in data:
        prompt = item.get('input', '')
        gt = item.get('output', '')

        if prompt and gt:
            formatted_data.append({
                'prompt': prompt,
                'gt_text': gt
            })

    if len(formatted_data) == 0:
        return None
    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.shuffle(seed=42)
    return dataset


# ------------------------------
# Reward function for GRPO.
# ------------------------------
def grpo_reward_function(completions, **kwargs):
    rewards = []
    gt_texts = kwargs.get('gt_text', [])
    for pred, gt in zip(completions, gt_texts):
        reward = grpo_hybrid_reward(pred, gt)
        rewards.append(float(reward))

    return rewards


# ------------------------------
# Training function.
# ------------------------------
def train(train_fpath, save_dir, model_dir):
    # Convert records to GRPO format (the dataset must contain a 'prompt' field).
    grpo_dataset = prepare_dataset_for_grpo(train_fpath)

    if grpo_dataset is None or len(grpo_dataset) == 0:
        print("Error: the dataset is empty or does not contain a 'prompt' field.")
        return

    print(f"Dataset prepared: {len(grpo_dataset)} samples")

    # GRPO configuration.
    grpo_config = GRPOConfig(
        output_dir=save_dir,

        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,

        learning_rate=1e-6,
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        # warmup_steps=100,
        bf16=True,

        num_train_epochs=1,
        #max_steps=2000,
        beta=0.1,


        num_generations=8,  # Generate multiple responses for each prompt.

        gradient_checkpointing=True,  # Enable gradient checkpointing
        use_vllm=True,
        vllm_mode="colocate",
        max_prompt_length=15360,
        max_completion_length=1024,
        # vllm_mode="server",  # Use a separately hosted vLLM server.
        # vllm_server_host="example-host",  # Replace with the server hostname.
        # vllm_device='auto',
        vllm_gpu_memory_utilization=0.45,  # Control GPU memory usage
        generation_kwargs={
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
            "max_tokens": 256,
        },

        logging_steps=1,
        save_strategy='steps',
        save_steps=0.2,
        #save_total_limit=5,


        remove_unused_columns=False,
        report_to="tensorboard",  # Enable TensorBoard logging.
    )

    trainer = GRPOTrainer(
        model=model_dir,
        args=grpo_config,
        train_dataset=grpo_dataset,
        reward_funcs=grpo_reward_function,
    )
    # Start training.
    print(">>> Starting TRL GRPO training...")
    #trainer.train(resume_from_checkpoint=True)
    trainer.train()
    print(f"Model saved to {save_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('train_fpath', help="Training dataset path")
    parser.add_argument('save_dir', help="Directory to save the trained model")
    parser.add_argument('model_dir', help="Directory containing the pretrained model")
    args = parser.parse_args()

    train(args.train_fpath, args.save_dir, args.model_dir)
