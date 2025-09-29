# eval_inference.py
import json, argparse, os, re, torch, time, random
from contextlib import nullcontext
from typing import Tuple, Any, List, Dict

from load_adapter_for_inference import load_model_with_adapter
from verifier_improved import (
    parse_steps_from_text,
    verifier_pass_fail,
    check_final_answer_from_text,
    canonicalize_problem,
)

# --------------------------- IO ---------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f]

def pick_fields(ex: Dict[str, Any]) -> Tuple[str, Any]:
    """Return (problem_text, gold_answer_or_None) supporting multiple schema variants."""
    prob = ex.get("problem")
    if prob is None:
        prob = ex.get("question")
    if prob is None and "prompt" in ex:
        prob = ex["prompt"]
    if prob is None:
        prob = str(ex)  # last resort

    gold = ex.get("answer")
    if gold is None:
        gold = ex.get("solution")
    if gold is None:
        gold = ex.get("gold")

    return str(prob), gold

# --------------------------- Prompt ---------------------------
def build_prompt(problem: str) -> str:
    return (
        "[ROLE: Solver]\n"
        f"Problem: {problem}\n"
        "Instruction: Provide a CLAIM and numbered PROOF_SKETCH.\n"
        "1) "
    )

# --------------------------- Answer parsing ---------------------------
_final_re = re.compile(r'####\s*([^\n]+)\s*$', re.MULTILINE)

def extract_final_span(txt: str) -> str:
    """Prefer content after the last '#### ...'. Else, last numeric; else stripped text."""
    if not txt:
        return ""
    m = _final_re.search(txt)
    if m:
        return m.group(1).strip()
    nums = re.findall(r'-?\d+\.?\d*', txt)
    return nums[-1] if nums else txt.strip()

def normalize_num(s: str) -> str:
    t = str(s).strip()
    t = re.sub(r'[,\s$]', '', t)  # remove separators for numeric compare
    try:
        x = float(t)
        return str(int(x)) if x.is_integer() else str(x)
    except Exception:
        return s if isinstance(s, str) else str(s)

def majority_vote(samples: List[str]) -> str:
    """Vote over normalized final answers; break ties by lexicographic (stable) then first sample."""
    finals = [extract_final_span(s) for s in samples]
    norms = [normalize_num(x) for x in finals]
    counts: Dict[str, int] = {}
    for n in norms:
        counts[n] = counts.get(n, 0) + 1
    if not counts:
        return samples[0] if samples else ""
    best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return best if best else (norms[0] if norms else "")

# --------------------------- Utils ---------------------------
def _get_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --------------------------- Main ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--adapter_dir', default='outputs/erft_sft')
    parser.add_argument('--data', default='data/maths_test.jsonl')
    parser.add_argument('--out', default='results/eval_traces.jsonl')
    parser.add_argument('--samples', type=int, default=1, help='number of samples per problem')
    parser.add_argument('--batch_size', type=int, default=8, help='problems per batch')
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--limit', type=int, default=None, help='evaluate only this many items')
    parser.add_argument('--no_4bit', action='store_true')

    # Sampling controls
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='>0 enables sampling; 0.0 means greedy')
    parser.add_argument('--top_p', type=float, default=None,
                        help='nucleus sampling; used if temperature>0')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')

    args = parser.parse_args()

    # Seed everything
    set_seed(args.seed)

    # Small speed win on Ampere+:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')  # PyTorch 2.x
        except Exception:
            pass

    print("Loading model...")
    tokenizer, model = load_model_with_adapter(
        args.base_model,
        args.adapter_dir,
        load_in_4bit=not args.no_4bit
    )
    device = _get_device(model)
    print("Model device:", device)

    data = read_jsonl(args.data)
    if args.limit is not None:
        data = data[:args.limit]

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    # Ensure pad token; prefer left padding for decoder-only batching
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    gen_pad_id = tokenizer.pad_token_id
    tokenizer.padding_side = 'left'

    torch.set_grad_enabled(False)

    # Build generation kwargs
    do_sample = args.temperature is not None and args.temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        pad_token_id=gen_pad_id,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            gen_kwargs["top_p"] = args.top_p

    t0_all = time.time()
    total = len(data)

    def prep_example(ex):
        raw_prob, gold = pick_fields(ex)
        prob = canonicalize_problem(raw_prob)
        prompt = build_prompt(prob)
        return prompt, raw_prob, gold  # keep raw_prob for record

    with open(args.out, 'w', encoding='utf-8') as outfh, torch.inference_mode():
        # autocast context
        amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with amp_ctx:
            for start in range(0, total, args.batch_size):
                batch = data[start:start + args.batch_size]
                prompts, raw_probs, golds = zip(*(prep_example(ex) for ex in batch))
                print(f"[{start+1}-{start+len(batch)}/{total}] batching {len(batch)} problems", flush=True)
                t0 = time.time()

                # Repeat prompts 'samples' times to get multiple generations per problem in one call
                repeated_prompts = []
                map_idx = []  # maps each repeated prompt row -> problem index in this batch
                for i, p in enumerate(prompts):
                    for _ in range(args.samples):
                        repeated_prompts.append(p)
                        map_idx.append(i)

                # Tokenize as a single padded batch
                enc = tokenizer(
                    list(repeated_prompts),
                    return_tensors='pt',
                    padding=True,
                    truncation=False,
                )
                enc = {k: v.to(device) for k, v in enc.items()}

                # Generate
                out_ids = model.generate(
                    **enc,
                    **gen_kwargs,
                )

                # Trim prompt per row using attention_mask (sum of non-pad tokens)
                input_lens = enc['attention_mask'].sum(dim=1)
                decoded = []
                for i in range(out_ids.size(0)):
                    gen_only = out_ids[i, input_lens[i]:]
                    txt = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
                    decoded.append(txt)

                # regroup k samples per problem, vote, verify, write
                for i in range(len(batch)):
                    samples_i = [decoded[j] for j in range(len(decoded)) if map_idx[j] == i]
                    pred_norm = majority_vote(samples_i)
                    sample0_final = extract_final_span(samples_i[0]) if samples_i else ""

                    steps = parse_steps_from_text(samples_i[0]) if samples_i else []
                    verif = verifier_pass_fail(steps)
                    verif_rate = sum(1 for v in verif if v.get('ok')) / max(1, len(verif))
                    final_correct = check_final_answer_from_text(samples_i[0] if samples_i else "", golds[i])

                    record = {
                        'problem': raw_probs[i],
                        'gold': golds[i],
                        'pred': pred_norm,
                        'sample0_final': sample0_final,
                        'verif_rate': verif_rate,
                        'final_correct': final_correct,
                        'sample_0': samples_i[0] if samples_i else "",
                        'n_samples': args.samples,
                        'temperature': args.temperature,
                        'top_p': args.top_p,
                        'max_new_tokens': args.max_new_tokens,
                    }
                    outfh.write(json.dumps(record, ensure_ascii=False) + '\n')

                print(f"  batch done in {time.time()-t0:.1f}s", flush=True)

    print(f"All done in {time.time()-t0_all:.1f}s -> {args.out}")

if __name__ == '__main__':
    main()
