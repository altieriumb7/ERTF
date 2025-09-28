# scripts/make_problems.py
# Creates data/problems_small.jsonl with a handful of simple problems.
# Usage:
#   python scripts/make_problems.py
# Output:
#   data/problems_small.jsonl   (each line: {"problem": "...", "id": "small_<n>"})


import json
import os

EXAMPLES = [
    "Compute 3 + 4 * 2.",
    "If 2x + 3 = 11, what is x?",
    "A rectangle has sides 3 and 5. What is its area?",
    "If a + b = 10 and a - b = 2, find a and b.",
    "What is the sum of the first 5 positive integers?",
    "Solve for y: 3y - 7 = 2y + 8.",
    "Compute (12/3) + (15/5).",
    "If 5% of N is 12, what is N?",
]

def main():
    os.makedirs("data", exist_ok=True)
    out_path = "data/problems_small.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, p in enumerate(EXAMPLES):
            f.write(json.dumps({"id": f"small_{i}", "problem": p}) + "\n")
    print(f"Wrote {len(EXAMPLES)} problems to {out_path}")

if __name__ == "__main__":
    main()
