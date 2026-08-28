import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from dataset import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
import argparse

def train(train_fpath, eval_fpath, save_dir, model_dir):
    if not os.path.isfile(train_fpath):
        raise FileNotFoundError(f"Training dataset does not exist: {train_fpath}")
    if not os.path.isfile(eval_fpath):
        raise FileNotFoundError(f"Evaluation dataset does not exist: {eval_fpath}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
    )

    model.config.use_cache = False
    print("Model loaded successfully!")


    print(f"Preparing dataset from {train_fpath}...")
    train_dataset = Dataset(
        train_fpath,
        tokenizer,
        max_len=4096,
        truncat=False,
        shuffle=True,
        num_workers=8,
        batch_size=1024,
    )
    print(f"Dataset loaded with {len(train_dataset)} samples.")

    print(f"Preparing evaluation dataset from {eval_fpath}...")
    eval_dataset = Dataset(
        eval_fpath,
        tokenizer,
        max_len=4096,
        truncat=False,
        shuffle=False,
        num_workers=8,
        batch_size=1024,
    )
    print(f"Evaluation dataset loaded with {len(eval_dataset)} samples.")

    if len(train_dataset) == 0:
        raise ValueError("The training dataset contains no usable records.")
    if len(eval_dataset) == 0:
        raise ValueError("The evaluation dataset contains no usable records.")


    per_device_train_batch_size = 1
    gradient_accumulation_steps = 8



    trainer_args = TrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=5e-5,
        lr_scheduler_type='cosine',
        warmup_steps=500,
        num_train_epochs=2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim='adamw_torch',
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='steps',
        logging_steps=5,
        logging_first_step=True,
        prediction_loss_only=True,
        per_device_eval_batch_size=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        bf16=True,
        seed=1234,
        ddp_find_unused_parameters=False,
        report_to="tensorboard",
    )

    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    print("Training complete.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_fpath', help="Training dataset path")
    parser.add_argument('eval_fpath', help="Evaluation dataset path")
    parser.add_argument('save_dir', help="Directory to save model weights")
    parser.add_argument('model_dir', help="Directory containing the base model")
    args = parser.parse_args()

    train(args.train_fpath, args.eval_fpath, args.save_dir, args.model_dir)

