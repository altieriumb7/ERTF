#!/usr/bin/env python
# coding: utf-8
"""
train_sft.py — Supervised fine-tuning with LoRA for causal LMs (Llama-2-7B, etc.)

- Expects a local JSONL file (no remote download).
- Accepts any of these schemas per line:
    {"prompt": "...", "target": "..."}
    {"prompt": "...", "response": "..."}
    {"input":  "...", "target": "..."}
- Concatenates prompt+target; labels = [-100]*len(prompt) + target_ids (+ EOS),
  so input and labels always have the same length.

Run example:
  accelerate launch --mixed_precision fp16 train_sft.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data data/gsm8k/train_prompt_target.jsonl \
    --output_dir outputs/gsm8k-sft-llama2-7b \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 3 \
    --learning_rate 2e-4 \
    --max_length 4096
"""

import argparse
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# Optional bitsandbytes (guarded import)
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
    p.add_argument("--data", type=str, required=True, help="Path to local JSONL")
    p.add_argument("--output_dir", type=str, default="adapters/agent_solver")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--load_in_4bit", action="store_true", help="Enable bnb 4-bit if available")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Tokenizer ---
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # --- Model (4-bit optional) ---
    quant_cfg = None
    if args.load_in_4bit and _BNB_AVAILABLE and BitsAndBytesConfig is not None:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quant_cfg,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    base.config.pad_token_id = tok.pad_token_id
    if getattr(base.config, "eos_token_id", None) is None and tok.eos_token_id is not None:
        base.config.eos_token_id = tok.eos_token_id

    # --- LoRA ---
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

    # --- Local JSONL dataset ---
    ds = load_dataset("json", data_files=args.data, split="train")

    def preprocess(ex):
        # Auto-detect fields
        prompt = ex.get("prompt", ex.get("input"))
        target = ex.get("target", ex.get("response"))
        if prompt is None or target is None:
            raise KeyError(f"Example missing text fields. Keys present: {list(ex.keys())}")

        # Tokenize without specials; we'll manage EOS and masking manually.
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tok(target, add_special_tokens=False)["input_ids"]

        # Ensure target ends with EOS (useful for causal LMs)
        if tok.eos_token_id is not None and (len(target_ids) == 0 or target_ids[-1] != tok.eos_token_id):
            target_ids = target_ids + [tok.eos_token_id]

        # Truncate from the LEFT on the prompt so that prompt+target <= max_length
        max_len = args.max_length
        total = len(prompt_ids) + len(target_ids)
        if total > max_len:
            keep_prompt = max(0, max_len - len(target_ids))
            prompt_ids = prompt_ids[-keep_prompt:]
            total = len(prompt_ids) + len(target_ids)
            assert total <= max_len

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    tokenized = ds.map(preprocess, remove_columns=ds.column_names)

    # --- Collator: dynamic padding + label masking ---
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tok,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # --- Training ---
    optim_name = "paged_adamw_32bit" if _BNB_AVAILABLE else "adamw_torch"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        bf16=False,
        optim=optim_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tok,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
