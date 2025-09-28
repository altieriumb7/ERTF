#!/usr/bin/env python
# coding: utf-8

"""
train_sft.py — Supervised fine-tuning for role-specific adapters.
Patched to support Llama-2-7B with Hugging Face + bitsandbytes (4-bit).
"""

import os
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str,
                   default="meta-llama/Llama-2-7b-hf",
                   help="Base model path or HF hub id")
    p.add_argument("--data", type=str, required=True,
                   help="Path to JSONL dataset with prompt/target fields")
    p.add_argument("--output_dir", type=str, default="adapters/agent_solver")
    p.add_argument("--role", type=str, default="solver")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    return p.parse_args()


def main():
    args = parse_args()

    # ---- BitsAndBytes quantization for Llama2 ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # ---- Load tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load base model in 4-bit ----
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # ---- Add LoRA adapter ----
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # ---- Load dataset ----
    dataset = load_dataset("json", data_files=args.data, split="train")

    def preprocess(ex):
        return tokenizer(
            ex["prompt"],
            text_target=ex["target"],
            truncation=True,
            max_length=512,
        )

    tokenized = dataset.map(preprocess, remove_columns=dataset.column_names)

    # ---- Training args ----
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        output_dir=args.output_dir,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        optim="paged_adamw_32bit",
        report_to="none",
    )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
