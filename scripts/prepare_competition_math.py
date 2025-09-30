#!/usr/bin/env python3
import os, json, argparse, random
from datasets import load_dataset, DatasetDict

SUBJECT_COL = "type"
LEVEL_COL = "level"

TEMPLATE_INSTRUCTION = (
    "You are a careful competition mathematician. Solve the problem step by step, "
    "check your work, then return only the final answer in \\\\boxed{...}.\\n\\n"
)

def extract_sft(example):
    return {
        "instruction": TEMPLATE_INSTRUCTION + example["problem"].strip(),
        "input": "",
        "output": example["solution"].strip(),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_path", default="qwedsacf/competition_math")
    ap.add_argument("--out_dir", default="data/competition_math")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.1, help="fraction of TRAIN after test split")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset(args.hf_path)
    if "train" not in ds:
        key = list(ds.keys())[0]
        ds = DatasetDict({"train": ds[key]})

    if SUBJECT_COL in ds["train"].column_names:
        tmp = ds["train"].train_test_split(test_size=args.test_size, seed=args.seed,
                                           stratify_by_column=SUBJECT_COL)
    else:
        tmp = ds["train"].train_test_split(test_size=args.test_size, seed=args.seed)

    if LEVEL_COL in tmp["train"].column_names:
        tv = tmp["train"].train_test_split(test_size=args.val_size, seed=args.seed,
                                           stratify_by_column=LEVEL_COL)
    else:
        tv = tmp["train"].train_test_split(test_size=args.val_size, seed=args.seed)

    splits = DatasetDict({"train": tv["train"], "validation": tv["test"], "test": tmp["test"]})

    def dump_jsonl(path, iterator):
        with open(path, "w", encoding="utf-8") as f:
            for ex in iterator:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    dump_jsonl(os.path.join(args.out_dir, "train_raw.jsonl"),
               ({k: ex[k] for k in ("problem", "solution", SUBJECT_COL, LEVEL_COL) if k in ex}
                for ex in splits["train"]))
    dump_jsonl(os.path.join(args.out_dir, "validation_raw.jsonl"),
               ({k: ex[k] for k in ("problem", "solution", SUBJECT_COL, LEVEL_COL) if k in ex}
                for ex in splits["validation"]))
    dump_jsonl(os.path.join(args.out_dir, "test_raw.jsonl"),
               ({k: ex[k] for k in ("problem", "solution", SUBJECT_COL, LEVEL_COL) if k in ex}
                for ex in splits["test"]))

    dump_jsonl(os.path.join(args.out_dir, "train_sft.jsonl"), (extract_sft(ex) for ex in splits["train"]))
    dump_jsonl(os.path.join(args.out_dir, "validation_sft.jsonl"), (extract_sft(ex) for ex in splits["validation"]))

    preview_path = os.path.join(args.out_dir, "PREVIEW.txt")
    with open(preview_path, "w", encoding="utf-8") as f:
        for i in range(5):
            ex = splits["train"][i]
            f.write(f"Problem {i+1}:\n{ex['problem']}\n---\nSolution:\n{ex['solution'][:400]}...\n\n")

    print("Saved splits to:", args.out_dir)

if __name__ == "__main__":
    main()
