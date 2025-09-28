#!/usr/bin/env python
# coding: utf-8

"""
train_rl_trl_multiturn_with_val.py
Patched for Llama-2-7B Hugging Face + bitsandbytes (4-bit).
"""

import argparse
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str,
                   default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--train_file", type=str, required=True,
                   help="Path to training JSONL with 'query','response','reward'")
    p.add_argument("--output_dir", type=str, default="ppo_checkpoints")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--mini_batch_size", type=int, default=1)
    p.add_argument("--ppo_epochs", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    # ---- BitsAndBytes quantization ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Value-head model for PPO ----
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # ---- PPO config ----
    config = PPOConfig(
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        learning_rate=args.learning_rate,
        ppo_epochs=args.ppo_epochs,
        log_with=None,
    )

    # ---- PPO Trainer ----
    dataset = load_dataset("json", data_files=args.train_file, split="train")
    trainer = PPOTrainer(config, model, tokenizer, dataset=dataset)

    # ---- PPO Loop (toy example) ----
    for step, batch in enumerate(trainer.dataloader):
        query_tensors = batch["input_ids"]
        response_tensors = []
        for q in query_tensors:
            output = model.generate(q.unsqueeze(0).to(model.device), max_new_tokens=64)
            response_tensors.append(output[0])
        rewards = [torch.tensor([float(r)]) for r in batch["reward"]]
        trainer.step(query_tensors, response_tensors, rewards)
        if step % 10 == 0:
            print(f"[INFO] Step {step} done")
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
