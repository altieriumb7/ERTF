# scripts/preprocess_math.py
"""
Convert MATH dataset (or generic JSON/JSONL/CSV of problems) into:
- data/problems_{split}.jsonl  (one problem per line)
- optionally, create initial SFT examples (prompt+target) for roles.

Expected input formats:
- HuggingFace datasets (if available)
- JSONL with fields like: { "problem": "...", "solution": "...", "answer": "..." }
- JSON with nested fields
- CSV with columns "problem", "solution", "answer"

This script is intentionally forgiving / heuristic-based.
"""

import os
import json
import csv
import argparse
from pathlib import Path

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            yield json.loads(line)

def normalize_text(s):
    return " ".join(s.strip().split())

def detect_and_extract_from_record(rec):
    # try common patterns
    if isinstance(rec, dict):
        for key in ("problem","question","prompt"):
            if key in rec and rec[key]:
                problem = rec[key]
                break
        else:
            # fallback: buy first longish text field
            cand = max(rec.items(), key=lambda kv: len(str(kv[1]) or ""))
            problem = cand[1]
        # try to find answer / solution
        answer = rec.get("answer") or rec.get("solution") or rec.get("final_answer") or rec.get("correct")
        # solution may be long; normalize
        return normalize_text(problem), (normalize_text(answer) if answer else None)
    else:
        return str(rec), None

def convert_jsonl_to_problems(in_path, out_problems_path, split_name="train"):
    ensure_dir(os.path.dirname(out_problems_path))
    count = 0
    with open(out_problems_path, "w", encoding="utf-8") as fout:
        for rec in read_jsonl(in_path):
            problem, answer = detect_and_extract_from_record(rec)
            obj = {"id": f"{split_name}_{count}", "problem": problem}
            if answer:
                obj["answer"] = answer
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    print(f"[INFO] Wrote {count} problems to {out_problems_path}")

def convert_csv_to_problems(in_csv, out_problems_path, split_name="train", problem_col=None, answer_col=None):
    ensure_dir(os.path.dirname(out_problems_path))
    count = 0
    with open(in_csv, newline="", encoding="utf-8") as csvfile, open(out_problems_path, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(csvfile)
        # heuristics
        if problem_col is None:
            for c in reader.fieldnames:
                if "problem" in c.lower() or "question" in c.lower() or "prompt" in c.lower():
                    problem_col = c
                    break
            if problem_col is None:
                problem_col = reader.fieldnames[0]
        if answer_col is None:
            for c in reader.fieldnames:
                if "answer" in c.lower() or "solution" in c.lower():
                    answer_col = c
                    break

        csvfile.seek(0)
        next(reader)  # skip header? already used by DictReader

        for row in reader:
            problem = row.get(problem_col, "")
            answer = row.get(answer_col) if answer_col else None
            obj = {"id": f"{split_name}_{count}", "problem": normalize_text(problem)}
            if answer:
                obj["answer"] = normalize_text(answer)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    print(f"[INFO] Wrote {count} problems to {out_problems_path}")

def create_sft_examples_from_problems(problems_jsonl, out_sft_jsonl, role="solver", max_examples=None):
    """
    Create simple SFT pairs for adapter training.
    For solver role: prompt = role header + problem, target = canonical chain-of-thought if available (not available), fallback to empty.
    We'll create placeholder targets if no solution is available — better to use demo-generated traces later.
    """
    ensure_dir(os.path.dirname(out_sft_jsonl))
    count = 0
    with open(problems_jsonl, "r", encoding="utf-8") as fin, open(out_sft_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            if max_examples and count >= max_examples:
                break
            obj = json.loads(line)
            problem = obj.get("problem")
            prompt = f"[ROLE: {role.upper()}]\nProblem: {problem}\nPlease give your step-by-step solution.\n"
            # If ground-truth answer available we put it in target as "Answer: X" but best to use full CoT if you have
            answer = obj.get("answer")
            if answer:
                target = f"Answer: {answer}"
            else:
                target = "Answer: "  # placeholder; replaced later with generated traces
            fout.write(json.dumps({"prompt": prompt, "target": target, "role": role}) + "\n")
            count += 1
    print(f"[INFO] Wrote {count} SFT examples for role {role} to {out_sft_jsonl}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="input file (jsonl/json/csv) or directory")
    p.add_argument("--out_dir", type=str, default="data", help="output data directory")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--make_sft", action="store_true", help="create SFT prompt-targets (placeholder if no solutions available)")
    p.add_argument("--sft_roles", type=str, default="solver,verifier,strategist", help="comma separated roles")
    p.add_argument("--max_sft_examples", type=int, default=1000)
    args = p.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    out_problems = out_dir / f"problems_{args.split}.jsonl"

    # If input is a directory with files, try to find jsonl/csv
    if input_path.is_dir():
        # find any jsonl file
        files = list(input_path.glob("**/*"))
        candidate = None
        for f in files:
            if f.suffix.lower() in [".jsonl", ".json", ".csv"]:
                candidate = f
                break
        if candidate is None:
            raise ValueError("No suitable file found in directory.")
        input_file = candidate
    else:
        input_file = input_path

    # dispatch by suffix
    if input_file.suffix.lower() == ".jsonl":
        convert_jsonl_to_problems(str(input_file), str(out_problems), split_name=args.split)
    elif input_file.suffix.lower() == ".json":
        # try to load json array
        with open(str(input_file), "r", encoding="utf-8") as f:
            data = json.load(f)
        # write each item as line
        with open(out_problems, "w", encoding="utf-8") as fout:
            for i, rec in enumerate(data):
                problem, answer = detect_and_extract_from_record(rec)
                fout.write(json.dumps({"id": f"{args.split}_{i}", "problem": problem, "answer": answer}) + "\n")
        print(f"[INFO] Wrote {len(data)} problems to {out_problems}")
    elif input_file.suffix.lower() == ".csv":
        convert_csv_to_problems(str(input_file), str(out_problems), split_name=args.split)
    else:
        # fallback: assume plain text file with one problem per line
        with open(str(input_file), "r", encoding="utf-8") as fin, open(out_problems, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if not line.strip(): continue
                fout.write(json.dumps({"id": f"{args.split}_{i}", "problem": normalize_text(line)}) + "\n")
        print(f"[INFO] Wrote problems to {out_problems}")

    # optionally create sft placeholders for each role
    if args.make_sft:
        roles = [r.strip() for r in args.sft_roles.split(",")]
        for r in roles:
            out_sft = out_dir / f"sft_dialogs_{r}_{args.split}.jsonl"
            create_sft_examples_from_problems(str(out_problems), str(out_sft), role=r, max_examples=args.max_sft_examples)

if __name__ == "__main__":
    main()
