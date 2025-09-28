#!/usr/bin/env python
# coding: utf-8
"""
Minimal 3-agent loop (Solver, Verifier, Strategist) with a single Llama-2 7B backbone.
- Loads base in 4-bit (bitsandbytes) and optionally wraps per-role PEFT adapters.
- Enforces safe prompt truncation well below 4096 ctx to avoid NaNs.
- Uses greedy decoding by default for stability; flip do_sample=True once it's working.

Run:
  python src/sandbox/agent_loop.py --model_name_or_path meta-llama/Llama-2-7b-hf \
    --out_jsonl results/dialogs_llama2_7b.jsonl
"""

import os
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sympy.parsing.sympy_parser import parse_expr

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True,
                   help="HF model id or local path (e.g., meta-llama/Llama-2-7b-hf)")
    p.add_argument("--adapter_solver", type=str, default="adapters/agent_solver")
    p.add_argument("--adapter_verifier", type=str, default="adapters/agent_verifier")
    p.add_argument("--adapter_strategist", type=str, default="adapters/agent_strategist")
    p.add_argument("--out_jsonl", type=str, default="results/dialogs.jsonl")
    p.add_argument("--max_rounds", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--sample", action="store_true",
                   help="Use sampling (temperature/top-p). Default is greedy for stability.")
    return p.parse_args()

# ---------------- Model Loading ----------------
def load_model_and_tokenizer(model_name: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    # pad/eos safety
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
    )
    # ensure model uses the same pad/eos
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.eval()
    return tok, model

def maybe_wrap_adapter(base_model, adapter_dir):
    if adapter_dir and os.path.isdir(adapter_dir):
        return PeftModel.from_pretrained(base_model, adapter_dir).eval()
    return base_model

# ---------------- Helpers ----------------
def build_prompt(role, problem, history):
    head = f"[ROLE: {role.upper()}]\n"
    conv = "".join(f"[{r.upper()}]: {t}\n" for (r, t) in history)
    return f"{head}Problem: {problem}\nConversation so far:\n{conv}{head}Output:"

def split_steps(text: str):
    lines = [l.strip("-. \t") for l in text.strip().splitlines() if l.strip()]
    if len(lines) <= 1:
        parts = text.replace(";", "\n").splitlines()
        lines = [p.strip() for p in parts if p.strip()]
    return lines if lines else [text.strip()]

def check_step(step_text: str):
    try:
        if "=" in step_text:
            a, b = step_text.split("=", 1)
            return parse_expr(a.strip()).equals(parse_expr(b.strip()))
        parse_expr(step_text)  # parses OK
        return True
    except Exception:
        return False

# ---------------- Core Loop ----------------
def run_demo(problems, model_name, adapters, out_jsonl, max_rounds, max_new_tokens, use_sampling=False):
    tok, base = load_model_and_tokenizer(model_name)

    # wrap with adapters if present (all share same backbone weights in memory)
    models = {
        "solver": maybe_wrap_adapter(base, adapters.get("solver")),
        "verifier": maybe_wrap_adapter(base, adapters.get("verifier")),
        "strategist": maybe_wrap_adapter(base, adapters.get("strategist")),
    }

    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

    # stable generation defaults
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    if use_sampling:
        gen_kwargs.update(dict(do_sample=True, temperature=0.7, top_p=0.9))

    # context headroom: 4096 ctx for Llama-2; keep prompt ≤ 3500 to be safe when adding new tokens
    MAX_PROMPT_TOKENS = 3500

    with open(out_jsonl, "a", encoding="utf-8") as fout:
        for prob in problems:
            convo = []
            ok_cnt, chk_cnt = 0, 0
            log = {"problem": prob, "turns": []}

            for _ in range(max_rounds):
                # ---- SOLVER ----
                p = build_prompt("solver", prob, convo)
                inputs = tok(p, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS)
                inputs = {k: v.to(models["solver"].device) for k, v in inputs.items()}

                try:
                    out = models["solver"].generate(**inputs, **gen_kwargs)
                except RuntimeError:
                    # fallback: greedy no-sampling (avoid NaN/inf)
                    safe_kwargs = dict(gen_kwargs, do_sample=False, temperature=None, top_p=None)
                    out = models["solver"].generate(**inputs, **safe_kwargs)

                stext = tok.decode(out[0], skip_special_tokens=True)
                convo.append(("solver", stext))
                log["turns"].append({"role": "solver", "text": stext})

                # quick step checks
                for step in split_steps(stext):
                    chk_cnt += 1
                    ok_cnt += int(check_step(step))

                # ---- VERIFIER ----
                p = build_prompt("verifier", prob, convo)
                inputs = tok(p, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS)
                inputs = {k: v.to(models["verifier"].device) for k, v in inputs.items()}

                try:
                    out = models["verifier"].generate(**inputs, **gen_kwargs)
                except RuntimeError:
                    safe_kwargs = dict(gen_kwargs, do_sample=False, temperature=None, top_p=None)
                    out = models["verifier"].generate(**inputs, **safe_kwargs)

                vtext = tok.decode(out[0], skip_special_tokens=True)
                convo.append(("verifier", vtext))
                log["turns"].append({"role": "verifier", "text": vtext})

                # ---- STRATEGIST ----
                p = build_prompt("strategist", prob, convo)
                inputs = tok(p, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS)
                inputs = {k: v.to(models["strategist"].device) for k, v in inputs.items()}

                try:
                    out = models["strategist"].generate(**inputs, **gen_kwargs)
                except RuntimeError:
                    safe_kwargs = dict(gen_kwargs, do_sample=False, temperature=None, top_p=None)
                    out = models["strategist"].generate(**inputs, **safe_kwargs)

                ttext = tok.decode(out[0], skip_special_tokens=True)
                convo.append(("strategist", ttext))
                log["turns"].append({"role": "strategist", "text": ttext})

                # stop early if all solver substeps verified
                if chk_cnt > 0 and ok_cnt == chk_cnt:
                    break

            log["verifier_accept_rate"] = ok_cnt / max(1, chk_cnt)
            fout.write(json.dumps(log, ensure_ascii=False) + "\n")

    print(f"[OK] Dialogs saved to {out_jsonl}")


if __name__ == "__main__":
    args = parse_args()
    adapters = {
        "solver": args.adapter_solver,
        "verifier": args.adapter_verifier,
        "strategist": args.adapter_strategist,
    }
    # simple built-in problems; replace with MATH later
    PROBLEMS = [
        "Compute 3 + 4 * 2.",
        "If 2x + 3 = 11, what is x?",
        "A rectangle has sides 3 and 5. What is its area?",
    ]
    run_demo(
        problems=PROBLEMS,
        model_name=args.model_name_or_path,
        adapters=adapters,
        out_jsonl=args.out_jsonl,
        max_rounds=args.max_rounds,
        max_new_tokens=args.max_new_tokens,
        use_sampling=args.sample,
    )
