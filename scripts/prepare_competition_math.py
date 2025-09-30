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

def dump_jsonl(path, iterator):
    with open(path, "w", encoding="utf-8") as f:
        for ex in iterator:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

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

    # --- First split: Train/Test, stratified by SUBJECT_COL if present ---
    train_ds = ds["train"]
    if SUBJECT_COL in train_ds.column_names:
        # convert to ClassLabel so stratify works
        train_ds = train_ds.class_encode_column(SUBJECT_COL)
        tmp = train_ds.train_test_split(
            test_size=args.test_size,
            seed=args.seed,
            stratify_by_column=SUBJECT_COL,
        )
        # ensure test split keeps same encoded dtype for consistency (optional)
        test_ds = ds["train"].class_encode_column(SUBJECT_COL)
        test_idx = tmp["test"].indices if hasattr(tmp["test"], "indices") else None
        if test_idx is not None:
            test_ds = test_ds.select(test_idx)
        else:
            test_ds = tmp["test"]
    else:
        tmp = train_ds.train_test_split(test_size=args.test_size, seed=args.seed)
        test_ds = tmp["test"]

    # --- Second split: from remaining Train, carve Validation stratified by LEVEL_COL if present ---
    train_rem = tmp["train"]
    if LEVEL_COL in train_rem.column_names:
        train_rem = train_rem.class_encode_column(LEVEL_COL)
        tv = train_rem.train_test_split(
            test_size=args.val_size,
            seed=args.seed,
            stratify_by_column=LEVEL_COL,
        )
        train_final = tv["train"]
        val_final = tv["test"]
    else:
        tv = train_rem.train_test_split(test_size=args.val_size, seed=args.seed)
        train_final = tv["train"]
        val_final = tv["test"]

    splits = DatasetDict({"train": train_final, "validation": val_final, "test": test_ds})

    # --- Export raw JSONL (problem/solution + optional meta) ---
    def keep_fields(ex):
        out = {"problem": ex["problem"], "solution": ex["solution"]}
        if SUBJECT_COL in ex: out[SUBJECT_COL] = ex[SUBJECT_COL]
        if LEVEL_COL in ex: out[LEVEL_COL] = ex[LEVEL_COL]
        return out

    dump_jsonl(os.path.join(args.out_dir, "train_raw.jsonl"), (keep_fields(ex) for ex in splits["train"]))
    dump_jsonl(os.path.join(args.out_dir, "validation_raw.jsonl"), (keep_fields(ex) for ex in splits["validation"]))
    dump_jsonl(os.path.join(args.out_dir, "test_raw.jsonl"), (keep_fields(ex) for ex in splits["test"]))

    # --- Export SFT JSONL ---
    dump_jsonl(os.path.join(args.out_dir, "train_sft.jsonl"), (extract_sft(ex) for ex in splits["train"]))
    dump_jsonl(os.path.join(args.out_dir, "validation_sft.jsonl"), (extract_sft(ex) for ex in splits["validation"]))

    # --- Tiny preview ---
    preview_path = os.path.join(args.out_dir, "PREVIEW.txt")
    with open(preview_path, "w", encoding="utf-8") as f:
        for i in range(min(5, len(splits["train"]))):
            ex = splits["train"][i]
            f.write(f"Problem {i+1}:\n{ex['problem']}\n---\nSolution:\n{ex['solution'][:400]}...\n\n")

    print("Saved splits to:", args.out_dir)

if __name__ == "__main__":
    main()
