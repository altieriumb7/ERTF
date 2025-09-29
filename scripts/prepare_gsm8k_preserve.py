import argparse, json, random, pathlib
from datasets import load_dataset
from tqdm import tqdm

def to_row(question: str, answer: str) -> dict:
    # Preserve the original dataset verbatim.
    # No added instructions, no formatting changes.
    return {"prompt": question, "response": answer}

def write_jsonl(path: pathlib.Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            # Dump exactly; do not trim or alter whitespace.
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="data/gsm8k")
    ap.add_argument("--val_ratio", type=float, default=0.1,
                    help="Fraction of TRAIN to hold out for validation.")
    ap.add_argument("--val_size", type=int, default=None,
                    help="Alternative to --val_ratio: exact number to hold out.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)

    # Load the official GSM8K (configuration: 'main'); splits are 'train' and 'test'.
    ds = load_dataset("gsm8k", "main")

    # Convert TRAIN to our prompt/response rows, preserving text.
    train_rows = [to_row(ex["question"], ex["answer"]) for ex in tqdm(ds["train"], desc="train")]
    # Deterministic shuffle for val carve-out.
    rng = random.Random(args.seed)
    rng.shuffle(train_rows)

    # Choose validation size.
    if args.val_size is not None:
        val_n = min(max(1, args.val_size), len(train_rows) - 1)
    else:
        val_n = max(1, int(len(train_rows) * args.val_ratio))

    val_rows = train_rows[:val_n]
    train_rows = train_rows[val_n:]

    # Convert TEST split (also has answers) with no changes.
    test_rows = [to_row(ex["question"], ex["answer"]) for ex in tqdm(ds["test"], desc="test")]

    # Write out
    write_jsonl(outdir / "train.jsonl", train_rows)
    write_jsonl(outdir / "val.jsonl",   val_rows)
    write_jsonl(outdir / "test.jsonl",  test_rows)

    # Report counts
    print(f"Wrote: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} -> {outdir}")

if __name__ == "__main__":
    main()
