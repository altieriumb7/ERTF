#!/usr/bin/env python3
"""
Evaluation & decoding for competition_math with Best-of-N and optional CoVe verification.
"""
import os, json, argparse, re
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from utils_math import extract_boxed, canonicalize_answer, seed_everything
from verifier_unified import Verifier, VerifierConfig


def load_model(base_model: str):
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model


def format_prompt(problem: str) -> str:
    return (
        "You are a careful competition mathematician.\n"
        "Solve the problem step by step. Show your reasoning. "
        "Double-check arithmetic. Finally, output only the final answer inside \\\\boxed{...}.\n\n"
        f"Problem: {problem}\n\nSolution:"
    )


def generate_one(model, tok, prompt, max_new_tokens=512, temperature=0.6, top_p=0.95, seed=0):
    if seed is not None:
        seed_everything(seed)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):]


def cove_check(model, tok, problem: str, draft: str, budget=96) -> float:
    """Chain-of-Verification: return penalty in [0,1]."""
    q = (
        "You will check the following solution for mistakes.\n"
        "Mark any incorrect step briefly and provide the corrected line if needed. "
        "Return a score between 0 (incorrect) and 1 (fully correct).\n\n"
        f"Problem:\n{problem}\n\nProposed solution:\n{draft}\n\n"
        "Assessment and score:"
    )
    check = generate_one(model, tok, q, max_new_tokens=budget, temperature=0.2, top_p=0.9)
    m = re.search(r"([01](?:\\.\\d+)?)\\s*$", check)
    if m:
        score = float(m.group(1))
        return max(0.0, min(1.0, 1.0 - score))
    return 0.5


def evaluate_predictions(gold_answers: List[str], pred_answers: List[str]) -> dict:
    total = len(gold_answers)
    correct = 0
    for g, p in zip(gold_answers, pred_answers):
        if canonicalize_answer(g) == canonicalize_answer(p):
            correct += 1
    acc = correct / max(1, total)
    return {"n": total, "correct": correct, "accuracy": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--hf_path", default="qwedsacf/competition_math")
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--samples", type=int, default=1, help="# candidates per problem")
    ap.add_argument("--von", type=int, default=0, help="alias for --samples")
    ap.add_argument("--vgd", action="store_true", help="Verifier-Guided Decoding")
    ap.add_argument("--cove", action="store_true", help="enable Chain-of-Verification penalty")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="limit # problems for debug")
    ap.add_argument("--out_csv", default="outputs/competition_math_eval.csv")
    args = ap.parse_args()

    if args.von and args.samples == 1:
        args.samples = args.von

    seed_everything(args.seed)

    tok, model = load_model(args.base_model)

    ds = load_dataset(args.hf_path)
    if args.split not in ds:
        raise SystemExit(f"Split '{args.split}' not found. Available: {list(ds.keys())}")

    data = ds[args.split]
    if args.limit:
        data = data.select(range(min(args.limit, len(data))))

    verifier = Verifier(VerifierConfig(strict_box=True, length_penalty=0.05))

    Path(os.path.dirname(args.out_csv) or ".").mkdir(parents=True, exist_ok=True)
    import csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["idx", "subject", "level", "gold_boxed", "pred_boxed", "is_correct", "pick_score"])

        gold, preds = [], []
        temps = [0.2, 0.6, 0.8, 1.0]

        for i, ex in enumerate(data):
            problem = ex["problem"]
            gold_box = extract_boxed(ex["solution"]) if "solution" in ex else ""

            prompt = format_prompt(problem)

            candidates, scores = [], []
            for k in range(max(1, args.samples)):
                t = temps[k % len(temps)] if args.samples > 1 else 0.6
                draft = generate_one(model, tok, prompt,
                                     max_new_tokens=args.max_new_tokens,
                                     temperature=t, top_p=0.95,
                                     seed=args.seed + k)
                ans = extract_boxed(draft)

                score = 0.0
                if args.vgd:
                    score = verifier.score(problem, draft, ans, meta={"temp": t})
                if args.cove:
                    score -= cove_check(model, tok, problem, draft)
                candidates.append((draft, ans))
                scores.append(score)

            if args.vgd or args.cove:
                best_idx = max(range(len(candidates)), key=lambda j: scores[j])
            else:
                best_idx = 0

            best_draft, best_ans = candidates[best_idx]

            gold.append(gold_box)
            preds.append(best_ans)
            is_correct = int(canonicalize_answer(gold_box) == canonicalize_answer(best_ans))

            writer.writerow([
                i, ex.get("type", ""), ex.get("level", ""), gold_box,
                best_ans, is_correct,
                f"{scores[best_idx]:.3f}" if scores else ""
            ])

    metrics = evaluate_predictions(gold, preds)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
