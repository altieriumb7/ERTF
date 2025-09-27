# scripts/eval_math.py
"""
Evaluate predicted answers vs ground truth for MATH-style problems.
Usage:
python scripts/eval_math.py --preds preds.jsonl --gold data/problems_test.jsonl --out eval_results.json
preds.jsonl each line: {"id": "train_12", "prediction": "7"} or {"id":"...","prediction":"x=3"}
gold file: problems_{split}.jsonl with fields id and answer (optional)
"""

import json
import argparse
import math
from sympy.parsing.sympy_parser import parse_expr
from sympy import simplify, Eq

def normalize_answer(s):
    if s is None:
        return ""
    s = s.strip()
    # remove dollar signs, extra whitespace
    s = s.replace("$","")
    s = s.replace("\\,", "")
    s = " ".join(s.split())
    return s

def try_numeric(s):
    try:
        # remove trailing punctuation
        s2 = s.strip().rstrip(".,")
        if "/" in s2:
            # rational
            if " " in s2:
                # mixed fraction like '1 1/2'
                parts = s2.split()
                whole = int(parts[0])
                frac = parts[1]
                num, den = frac.split("/")
                val = whole + int(num)/int(den)
                return float(val)
            else:
                num, den = s2.split("/")
                return float(int(num)/int(den))
        return float(s2)
    except Exception:
        return None

def sympy_equal(a, b):
    try:
        ea = parse_expr(a)
        eb = parse_expr(b)
        # simplify difference
        diff = simplify(ea - eb)
        return diff == 0
    except Exception:
        return False

def compare(pred, gold, tol=1e-6):
    p = normalize_answer(pred)
    g = normalize_answer(gold) if gold is not None else ""

    # exact match
    if p == g and p != "":
        return True, "exact"

    # numeric match
    pn = try_numeric(p)
    gn = try_numeric(g)
    if pn is not None and gn is not None:
        if math.isclose(pn, gn, rel_tol=1e-6, abs_tol=tol):
            return True, "numeric"
        else:
            return False, "numeric_mismatch"

    # try sympy equivalence
    if p != "" and g != "":
        if sympy_equal(p, g):
            return True, "sympy_eq"

    # fallback: substring match (last ditch)
    if g != "" and g in p:
        return True, "contains_gold"

    return False, "no_match"

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            yield json.loads(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=str, required=True)
    parser.add_argument("--gold", type=str, required=True)
    parser.add_argument("--out", type=str, default="eval_results.json")
    args = parser.parse_args()

    gold_map = {}
    for rec in load_jsonl(args.gold):
        if "id" in rec:
            gold_map[rec["id"]] = rec.get("answer")
        else:
            # generate id
            # not ideal but fallback
            gold_map[len(gold_map)] = rec.get("answer")

    results = []
    correct = 0
    total = 0
    for pred in load_jsonl(args.preds):
        pid = pred.get("id")
        ptxt = pred.get("prediction", "")
        gold = gold_map.get(pid)
        ok, reason = compare(ptxt, gold)
        results.append({"id": pid, "prediction": ptxt, "gold": gold, "correct": ok, "reason": reason})
        if ok:
            correct += 1
        total += 1

    acc = correct / max(1, total)
    out = {"accuracy": acc, "total": total, "correct": correct, "details": results}
    with open(args.out, "w", encoding="utf-8") as fout:
        json.dump(out, fout, indent=2)
    print(f"[INFO] Accuracy: {acc:.4f} ({correct}/{total}). Results saved to {args.out}")

if __name__ == "__main__":
    main()
