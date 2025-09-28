#!/usr/bin/env python
# coding: utf-8

"""
train_sft.py — Supervised fine-tuning for role-specific adapters (LoRA).
- Supports Llama-2-7B in 4-bit (bitsandbytes).
- Expects a JSONL with fields: {"prompt": "...", "target": "..."} per line.

Run:
  accelerate launch train_sft.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data data/sft_dialogs_solver_train.jsonl \
    --output_dir adapters/agent_solver
"""

import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--data", type=str, required=True, help="JSONL with prompt/target")
    p.add_argument("--output_dir", type=str, default="adapters/agent_solver")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=1024)
    return p.parse_args()


def main():
    args = parse_args()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb,
        device_map="auto",
    )
    base.config.pad_token_id = tok.pad_token_id
    base.config.eos_token_id = tok.eos_token_id

    lora_cfg = LoraConfig(
        r=8, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(base, lora_cfg)

    ds = load_dataset("json", data_files=args.data, split="train")

    def preprocess(ex):
        return tok(
            ex["prompt"],
            text_target=ex["target"],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = ds.map(preprocess, remove_columns=ds.column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
        bf16=torch.cuda.is_available(),
        optim="paged_adamw_32bit",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
