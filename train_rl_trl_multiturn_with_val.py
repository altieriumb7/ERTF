#!/usr/bin/env python3
"""
train_rl_trl_multiturn_with_val.py

Multi-turn PPO training (TRL) with:
 - explicit PEFT adapter loading if `--sft_model_dir` is provided
 - validation loop that computes episode-level metrics and writes ckpt_summary.json

Usage (example):
accelerate launch train_rl_trl_multiturn_with_val.py \
  --rl_config rl_config.yaml \
  --sft_model_dir outputs/erft_sft \
  --dataset data/maths_train.jsonl \
  --val_dataset data/maths_val.jsonl \
  --output_dir outputs/erft_rl_multiturn_val
"""
import os
import argparse
import yaml
import json
import random
import math
from typing import List

import torch
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

# verifier_improved provides robust parsing and checking utilities
# Make sure verifier_improved.py is present in repo root (we provided it earlier)
from verifier_improved import parse_steps_from_text, verifier_pass_fail, check_final_answer_from_text, canonicalize_problem

# ----------------- utilities -----------------

def load_yaml(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def read_jsonl(path: str):
    arr = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                arr.append(json.loads(line))
    return arr

def build_initial_prompt(problem_text: str) -> str:
    return (
        f"[ROLE: Solver]\n"
        f"Problem: {problem_text}\n"
        "Instruction: Provide a short CLAIM and numbered PROOF_SKETCH with explicit intermediate numeric steps when possible.\n"
        "1) "
    )

def build_verifier_hint(failures: List[dict]) -> str:
    msgs = [f.get('msg', '') for f in failures[:4]]
    return "Verifier hint: " + " ; ".join(msgs)

def run_conversation(prompt: str, model, tokenizer, gen_kwargs: dict, max_rounds: int = 3):
    """
    Run a multi-turn Solver <-> Verifier conversation.
    Returns: final_text, all_steps (list of step strings), rounds_used (int)
    """
    conversation = prompt
    all_steps = []
    final_text = ""
    rounds_used = 0

    for r in range(max_rounds):
        rounds_used += 1
        inputs = tokenizer(conversation, return_tensors="pt").to(model.device)
        # generate
        out_ids = model.generate(**inputs, do_sample=True, **gen_kwargs)
        out_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)

        # sometimes generation includes the prompt; strip if present
        if conversation and conversation in out_text:
            out_text = out_text.split(conversation, 1)[1].strip()

        steps = parse_steps_from_text(out_text)
        verif_results = verifier_pass_fail(steps)
        all_steps.extend(steps)
        final_text = out_text

        # if all parsed steps are verified -> accept
        verified = sum(1 for v in verif_results if v['ok'])
        if len(verif_results) > 0 and verified == len(verif_results):
            break

        # otherwise, create a hint from failures and ask Solver to refine
        failures = [v for v in verif_results if not v['ok']]
        hint = build_verifier_hint(failures)
        conversation = conversation + "\n\n[VERIFIER]: " + hint + "\n\n[ROLE: Solver] Refine steps:\n1) "

    return final_text, all_steps, rounds_used

def compute_episode_reward(final_text: str, all_steps: List[str], rounds: int, cfg: dict, gold_answer):
    """
    Reward = w_final * final_correct + w_verif * verif_rate - w_round_penalty * (rounds / max_rounds)
    where verif_rate = fraction of steps that verifier accepted
    """
    verif_results = verifier_pass_fail(all_steps)
    verif_rate = sum(1 for v in verif_results if v['ok']) / max(1, len(verif_results))
    final_correct = check_final_answer_from_text(final_text, gold_answer)
    w_final = cfg['reward_weights'].get('w_final', 1.0)
    w_verif = cfg['reward_weights'].get('w_verif', 0.6)
    w_round = cfg['reward_weights'].get('w_round_penalty', 0.0)
    max_rounds = cfg.get('max_rounds', 3)
    r = w_final * float(final_correct) + w_verif * float(verif_rate) - w_round * (rounds / max(1, max_rounds))
    return float(r), float(verif_rate), bool(final_correct)

def run_validation(model, tokenizer, val_data, gen_kwargs, cfg, num_samples=32):
    """
    Run validation on a sample of val_data. Return a dict summary.
    """
    model.eval()
    samples = val_data if len(val_data) <= num_samples else random.sample(val_data, num_samples)
    total_reward = 0.0
    total_verif_rate = 0.0
    total_correct = 0
    traces = []

    for ex in samples:
        problem = canonicalize_problem(ex['problem'])
        prompt = build_initial_prompt(problem)
        final_text, all_steps, rounds = run_conversation(prompt, model, tokenizer, gen_kwargs, max_rounds=cfg.get('max_rounds', 3))
        reward, verif_rate, final_correct = compute_episode_reward(final_text, all_steps, rounds, cfg, ex.get('answer'))
        total_reward += reward
        total_verif_rate += verif_rate
        total_correct += 1 if final_correct else 0
        traces.append({'problem': ex['problem'], 'final_text': final_text, 'verif_rate': verif_rate, 'final_correct': final_correct})

    n = len(samples)
    if n == 0:
        return {'val_num_samples': 0, 'val_avg_reward': 0.0, 'val_avg_verif_rate': 0.0, 'val_final_correct_pct': 0.0, 'traces': []}
    return {
        'val_num_samples': n,
        'val_avg_reward': total_reward / n,
        'val_avg_verif_rate': total_verif_rate / n,
        'val_final_correct_pct': total_correct / n,
        'traces': traces[:5]  # keep a few traces for quick debugging
    }

# ----------------- main training routine -----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl_config", type=str, default="rl_config.yaml")
    parser.add_argument("--sft_model_dir", type=str, default=None, help="Path to SFT PEFT adapters (optional).")
    parser.add_argument("--dataset", type=str, required=True, help="Training dataset JSONL (problems).")
    parser.add_argument("--val_dataset", type=str, default=None, help="Validation dataset JSONL (optional).")
    parser.add_argument("--output_dir", type=str, default="outputs/erft_rl_multiturn_val")
    args = parser.parse_args()

    cfg = load_yaml(args.rl_config)
    os.makedirs(args.output_dir, exist_ok=True)

    # load datasets
    train_data = read_jsonl(args.dataset)
    val_data = read_jsonl(args.val_dataset) if args.val_dataset and os.path.exists(args.val_dataset) else []

    base_model = cfg.get('model_dir', args.sft_model_dir) or cfg.get('base_model', 'meta-llama/Llama-2-7b')

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading policy model (value head) in 4-bit...")
    # load model with value head
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model,
        load_in_4bit=True,
        device_map="auto",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    model = prepare_model_for_kbit_training(model)

    # apply LoRA config
    lora_cfg = LoraConfig(
        r=cfg.get('lora', {}).get('r', 8),
        lora_alpha=cfg.get('lora', {}).get('alpha', 32),
        target_modules=cfg.get('lora', {}).get('target_modules', ["q_proj", "v_proj", "k_proj", "o_proj"]),
        lora_dropout=cfg.get('lora', {}).get('dropout', 0.05),
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)

    # if SFT adapters exist, try to load them into model
    if args.sft_model_dir and os.path.isdir(args.sft_model_dir):
        try:
            print(f"Loading SFT adapters from {args.sft_model_dir} into model...")
            model = PeftModel.from_pretrained(model, args.sft_model_dir)
            print("Loaded adapters successfully.")
        except Exception as e:
            print("Warning: failed to load adapters via PeftModel.from_pretrained():", e)
            print("Continuing with new LoRA adapters (training from scratch).")

    # freeze base model params; make LoRA/adapter params trainable
    for n, p in model.named_parameters():
        if "lora" in n.lower() or "adapter" in n.lower() or "peft" in n.lower():
            p.requires_grad = True
        else:
            p.requires_grad = False

    # reference model (frozen) for PPO
    print("Loading reference model (frozen) in 4-bit...")
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model,
        load_in_4bit=True,
        device_map="auto",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    ref_model = prepare_model_for_kbit_training(ref_model)
    for p in ref_model.parameters():
        p.requires_grad = False

    # instantiate PPO trainer
    print("Initializing PPO trainer...")
    ppo_config = PPOConfig(
        model_name=base_model,
        batch_size=cfg['ppo'].get('episodes_per_update', 8),
        forward_batch_size=cfg['ppo'].get('minibatch_size', 4),
        ppo_epochs=cfg['ppo'].get('ppo_epochs', 4),
        learning_rate=cfg['ppo'].get('policy_lr', 1e-5)
    )
    ppo_trainer = PPOTrainer(config=ppo_config, model=model, ref_model=ref_model, tokenizer=tokenizer)

    gen_kwargs = {
        "max_new_tokens": cfg.get('generation', {}).get('max_new_tokens', 128),
        "temperature": cfg.get('generation', {}).get('temperature', 0.8),
        "top_p": cfg.get('generation', {}).get('top_p', 0.95),
        "do_sample": True
    }

    total_updates = cfg['training'].get('total_updates', 100)
    episodes_per_update = cfg['ppo'].get('episodes_per_update', 8)
    save_every = cfg['training'].get('save_every', 10)

    print(f"Starting RL: total_updates={total_updates}, episodes_per_update={episodes_per_update}, save_every={save_every}")

    # main training loop
    for update in range(total_updates):
        prompts = []
        responses = []
        rewards = []
        infos = []

        for ep in range(episodes_per_update):
            ex = random.choice(train_data)
            problem_text = canonicalize_problem(ex['problem'])
            prompt = build_initial_prompt(problem_text)
            final_text, all_steps, rounds = run_conversation(prompt, model, tokenizer, gen_kwargs, max_rounds=cfg.get('max_rounds', 3))
            reward, verif_rate, final_correct = compute_episode_reward(final_text, all_steps, rounds, cfg, ex.get('answer'))
            prompts.append(prompt)
            responses.append(final_text)
            rewards.append(reward)
            infos.append({'verif_rate': verif_rate, 'final_correct': final_correct, 'rounds': rounds})

        # PPO step
        try:
            stats = ppo_trainer.step(prompts=prompts, responses=responses, rewards=rewards)
        except Exception as e:
            # If TRL API changes, print error and still continue/stop gracefully
            print("Error during PPOTrainer.step():", e)
            raise

        avg_reward = sum(rewards) / len(rewards)
        print(f"[Update {update+1}/{total_updates}] avg_reward={avg_reward:.4f} stats={stats} sample_info={infos[0]}")

        # Save checkpoint + validation summary
        if (update + 1) % save_every == 0 or (update + 1) == total_updates:
            ckpt_dir = os.path.join(args.output_dir, f"ppo_ckpt_{update+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            print("Saving checkpoint to", ckpt_dir)
            try:
                ppo_trainer.model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
            except Exception as e:
                print("Warning: failed to save model via save_pretrained():", e)

            # run validation if val_data present
            val_summary = {}
            if len(val_data) > 0:
                print("Running validation on val set...")
                val_summary = run_validation(model, tokenizer, val_data, gen_kwargs, cfg, num_samples=min(64, len(val_data)))
                print("Validation summary:", val_summary)
            else:
                print("No validation dataset provided; skipping validation.")

            # write ckpt summary (includes training avg reward + validation metrics)
            ckpt_summary = {
                'update': update + 1,
                'train_avg_reward': avg_reward,
                'train_sample_info': infos[0],
                'val_summary': val_summary
            }
            try:
                with open(os.path.join(ckpt_dir, "ckpt_summary.json"), "w", encoding='utf-8') as f:
                    json.dump(ckpt_summary, f, indent=2)
                with open(os.path.join(args.output_dir, "latest_ckpt_summary.json"), "w", encoding='utf-8') as f:
                    json.dump(ckpt_summary, f, indent=2)
            except Exception as e:
                print("Warning: failed to write ckpt summary:", e)

    print("Training complete. Checkpoints & summaries in", args.output_dir)

if __name__ == "__main__":
    main()
