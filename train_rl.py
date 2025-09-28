#!/usr/bin/env python
# coding: utf-8

"""
train_rl.py — PPO finetuning for one agent role at a time (Solver/Verifier/Strategist).
- Loads Llama-2 7B in 4-bit with value head (TRL).
- Records exact prompts given to the trainable role and uses them for PPO.
- Expects problems JSONL: {"problem": "...", "id": "..."} per line.

Run:
  accelerate launch train_rl.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --train_role solver \
    --adapter_dir_train adapters/agent_solver_train \
    --problems_jsonl data/problems_small.jsonl \
    --output_dir ppo_checkpoints
"""

import os
import json
import argparse
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# ----- simple step checker -----
from sympy.parsing.sympy_parser import parse_expr

def step_ok(s: str) -> bool:
    try:
        if "=" in s:
            a, b = s.split("=", 1)
            return parse_expr(a.strip()).equals(parse_expr(b.strip()))
        parse_expr(s)
        return True
    except Exception:
        return False

def split_steps(text: str):
    lines = [l.strip("-. \t") for l in text.strip().splitlines() if l.strip()]
    if len(lines) <= 1:
        parts = text.replace(";", "\n").splitlines()
        lines = [p.strip() for p in parts if p.strip()]
    return lines if lines else [text.strip()]

def build_prompt(role, problem, history):
    head = f"[ROLE: {role.upper()}]\n"
    conv = "".join(f"[{r.upper()}]: {t}\n" for (r, t) in history)
    return f"{head}Problem: {problem}\nConversation so far:\n{conv}{head}Output:"

def compute_ep_reward(ep, w_final=1.0, w_verif=0.5, w_round=0.05):
    final_ok = 1.0 if ep.get("final_verified", False) else 0.0
    verif_frac = ep.get("verifier_ok_frac", 0.0)
    rounds = sum(1 for (r, _) in ep.get("turns", []) if r == "solver")
    return float(w_final * final_ok + w_verif * verif_frac - w_round * rounds)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--train_role", type=str, default="solver", choices=["solver","verifier","strategist"])
    p.add_argument("--adapter_dir_train", type=str, required=True)
    p.add_argument("--problems_jsonl", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="ppo_checkpoints")
    p.add_argument("--batch_episodes", type=int, default=4)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--total_updates", type=int, default=200)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16
    )

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name_or_path, quantization_config=bnb, device_map="auto"
    )
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id

    # PPO config
    ppo_cfg = PPOConfig(
        model_name=args.model_name_or_path,
        batch_size=args.batch_episodes,
        mini_batch_size=1,
        learning_rate=1e-5,
        ppo_epochs=args.ppo_epochs,
        log_with=None,
    )
    trainer = PPOTrainer(config=ppo_cfg, model=model, tokenizer=tok)

    # load problems
    problems = []
    with open(args.problems_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            problems.append((obj.get("id", None), obj["problem"]))

    # generation settings (stable first; sampling optional later)
    gen_kwargs = dict(max_new_tokens=192, do_sample=False, temperature=None, top_p=None,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    MAX_PROMPT_TOKENS = 3500

    for update in range(args.total_updates):
        queries, responses, rewards = [], [], []

        for ep_i in range(args.batch_episodes):
            pid, prob = problems[(update * args.batch_episodes + ep_i) % len(problems)]
            # run a short 3-role episode and collect exact prompts for train_role
            convo = []
            role_prompts = []
            verifier_ok, total_checked = 0, 0
            final_verified = False

            # one round (extend to multi-round as needed)
            # SOLVER
            ptxt = build_prompt("solver", prob, convo)
            q = tok(ptxt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS).to(model.device)
            out = model.generate(**q, **gen_kwargs)
            stext = tok.decode(out[0], skip_special_tokens=True)
            convo.append(("solver", stext))
            if args.train_role == "solver": role_prompts.append((ptxt, stext))
            for step in split_steps(stext):
                total_checked += 1
                verifier_ok += int(step_ok(step))

            # VERIFIER
            ptxt = build_prompt("verifier", prob, convo)
            q = tok(ptxt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS).to(model.device)
            out = model.generate(**q, **gen_kwargs)
            vtext = tok.decode(out[0], skip_special_tokens=True)
            convo.append(("verifier", vtext))
            if args.train_role == "verifier": role_prompts.append((ptxt, vtext))

            # STRATEGIST
            ptxt = build_prompt("strategist", prob, convo)
            q = tok(ptxt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS).to(model.device)
            out = model.generate(**q, **gen_kwargs)
            ttext = tok.decode(out[0], skip_special_tokens=True)
            convo.append(("strategist", ttext))
            if args.train_role == "strategist": role_prompts.append((ptxt, ttext))

            if total_checked > 0 and verifier_ok == total_checked:
                final_verified = True

            ep = {
                "id": pid, "problem": prob, "turns": convo,
                "verifier_ok_frac": verifier_ok / max(1, total_checked),
                "final_verified": final_verified,
            }
            ep_reward = compute_ep_reward(ep)

            # add all (prompt,response) pairs of train_role
            for prompt_text, response_text in role_prompts:
                # trim to avoid overly long strings
                prompt_trim = prompt_text[-4000:] if len(prompt_text) > 4000 else prompt_text
                response_trim = response_text[:2000] if len(response_text) > 2000 else response_text
                queries.append(prompt_trim)
                responses.append(response_trim)
                rewards.append(ep_reward)

        stats = trainer.step(queries, responses, rewards)
        if (update + 1) % 10 == 0:
            print(f"[update {update+1}] stats: {stats}")
            save_path = os.path.join(args.output_dir, f"{args.train_role}_update_{update+1}")
            model.save_pretrained(save_path)

    final_path = os.path.join(args.output_dir, f"{args.train_role}_final")
    model.save_pretrained(final_path)
    print("[DONE] PPO adapter saved to", final_path)


if __name__ == "__main__":
    main()
