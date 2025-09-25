#!/usr/bin/env python3
# Minimal SFT finetune script using QLoRA+LoRA (small scale for smoke testing).
import os, json, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
from tqdm import tqdm

def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(l) for l in f]

def make_prompt(prob):
    return f"[ROLE: Solver]\nProblem: {prob}\nInstruction: Provide numbered steps.\n1) "

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='sft_config.yaml')
    parser.add_argument('--lora_config', default='lora_config.yaml')
    args = parser.parse_args()
    # load small dataset
    train = read_jsonl('data/maths_train.jsonl')
    val = read_jsonl('data/maths_val.jsonl')
    # load tokenizer and model (small test: use a tiny model if you don't have 7b)
    model_name = 'meta-llama/Llama-2-7b'
    print('Loading tokenizer and model (may require bitsandbytes)...')
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map='auto')
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(r=8, lora_alpha=32, target_modules=['q_proj','v_proj'], lora_dropout=0.05, task_type='CAUSAL_LM')
    model = get_peft_model(model, lora_cfg)
    model.train()
    # Very small "training" loop: do 1 epoch over training data, teacher forcing
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    for ex in tqdm(train):
        prompt = make_prompt(ex['problem'])
        target = ex['answer']
        inputs = tokenizer(prompt + target, return_tensors='pt', truncation=True).to(model.device)
        labels = inputs['input_ids'].clone()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()
        optim.zero_grad()
    # save adapters
    save_dir = 'outputs/erft_sft_test'
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print('Saved test adapters to', save_dir)

if __name__ == '__main__':
    main()
