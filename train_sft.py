#!/usr/bin/env python
# coding: utf-8
"""
train_sft.py — Supervised fine-tuning for role-specific adapters (LoRA).
- Supports Llama-2-7B in 4-bit (bitsandbytes) when available; otherwise fp16.
- Accepts JSONL with any of these schema per line:
    {"prompt": "...", "target": "..."}       # original expectation
    {"prompt": "...", "response": "..."}     # your GSM8K files
    {"input":  "...", "target": "..."}       # alt schema

Run:
  accelerate launch --mixed_precision fp16 train_sft.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data data/gsm8k/train_prompt_target.jsonl \
    --output_dir outputs/gsm8k-sft-llama2-7b
"""

import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

# optional imports guarded for no-bnb environments
_BNB_AVAILABLE = False
try:
    from transformers import BitsAndBytesConfig
    import bitsandbytes as bnb  # noqa: F401
    _BNB_AVAILABLE = True
except Exception:
    BitsAndBytesConfig = None  # type: ignore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--data", type=str, required=True, help="JSONL with prompt/target or prompt/response")
    p.add_argument("--output_dir", type=str, default="adapters/agent_solver")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--load_in_4bit", action="store_true", help="Enable 4-bit bnb loading if available")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- Tokenizer (ensure pad token) ----
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # ---- Model (bnb 4-bit if requested & available; else fp16) ----
    quantization_config = None
    if args.load_in_4bit and _BNB_AVAILABLE and BitsAndBytesConfig is not None:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    base.config.pad_token_id = tok.pad_token_id
    if getattr(base.config, "eos_token_id", None) is None and tok.eos_token_id is not None:
        base.config.eos_token_id = tok.eos_token_id

    # ---- LoRA ----
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)

    # ---- Data ----
    ds = load_dataset("json", data_files=args.data, split="train")
def preprocess(ex):
    # accept {"prompt","target"} or {"prompt","response"} or {"input","target"}
    prompt = ex.get("prompt", ex.get("input"))
    target = ex.get("target", ex.get("response"))
    if prompt is None or target is None:
        raise KeyError(f"Example missing text fields. Keys: {list(ex.keys())}")

    # tokenize without adding extra special tokens
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tok(target, add_special_tokens=False)["input_ids"]

    # ensure an EOS at the end of the target (helps training stability)
    if tok.eos_token_id is not None and (len(target_ids) == 0 or target_ids[-1] != tok.eos_token_id):
        target_ids = target_ids + [tok.eos_token_id]

    # truncate so total length <= max_length (keep all of target, trim prompt from the left)
    max_len = args.max_length
    total_len = len(prompt_ids) + len(target_ids)
    if total_len > max_len:
        keep_prompt = max(0, max_len - len(target_ids))
        prompt_ids = prompt_ids[-keep_prompt:]
        total_len = len(prompt_ids) + len(target_ids)
        assert total_len <= max_len

    input_ids = prompt_ids + target_ids
    # labels mask: ignore prompt positions, learn on target
    labels = [-100] * len(prompt_ids) + target_ids

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

if __name__ == "__main__":
    main()
