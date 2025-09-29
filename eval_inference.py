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
    # problem / question
    prob = ex.get("problem")
    if prob is None:
        prob = ex.get("question")
    if prob is None and "prompt" in ex:
        prob = ex["prompt"]
    if prob is None:
        # last resort: stringify the whole example
        prob = str(ex)

    # gold answer
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
    """
    Prefer content after the last '#### ...' marker.
    Else, fall back to the last numeric span in the text.
    """
    if not txt:
        return ""
    m = _final_re.search(txt)
    if m:
        return m.group(1).strip()
    nums = re.findall(r'-?\d+\.?\d*', txt)
    return nums[-1] if nums else txt.strip()

def normalize_num(s: str) -> str:
    t = str(s).strip()
    # remove spaces, commas, $ for numeric compare
    t = re.sub(r'[,\s$]', '', t)
    try:
        x = float(t)
        return str(int(x)) if x.is_integer() else str(x)
    except Exception:
        return s if isinstance(s, str) else str(s)

def majority_vote(samples: List[str]) -> str:
    """
    Majority vote on normalized final answers extracted per sample.
    If tie, return the final answer from the first sample.
    """
    finals = [extract_final_span(s) for s in samples]
    norms = [normalize_num(x) for x in finals]
    counts: Dict[str, int] = {}
    for n in norms:
        counts[n] = counts.get(n, 0) + 1
    if not counts:
        return samples[0] if samples else ""
    best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    # if the winner is weirdly empty, fall back
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
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--limit', type=int, default=None, help='evaluate only this many items')
    parser.add_argument('--no_4bit', action='store_true')

    # New: sampling controls
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='>0 enables sampling; 0.0 means greedy')
    parser.add_argument('--top_p', type=float, default=None,
                        help='nucleus sampling; used if temperature>0')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')

    args = parser.parse_args()

    # Seed everything
    set_seed(args.seed)

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

    # Make sure pad token is defined
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    gen_pad_id = tokenizer.pad_token_id

    torch.set_grad_enabled(False)

    # Build generation kwargs
    do_sample = args.temperature is not None and args.temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        pad_token_id=gen_pad_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            gen_kwargs["top_p"] = args.top_p

    t0_all = time.time()
    with open(args.out, 'w', encoding='utf-8') as outfh:
        for idx, ex in enumerate(data):
            t0 = time.time()

            raw_prob, gold = pick_fields(ex)
            prob = canonicalize_problem(raw_prob)
            prompt = build_prompt(prob)

            print(f"[{idx+1}/{len(data)}] generating... (len(prompt)={len(prompt)})", flush=True)

            samples: List[str] = []
            with torch.inference_mode():
                # Use new torch.amp autocast API (avoids deprecation warning)
                if device.type == "cuda":
                    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
                else:
                    amp_ctx = nullcontext()

                with amp_ctx:
                    for _ in range(args.samples):
                        inputs = tokenizer(prompt, return_tensors='pt', padding=False)
                        inputs = {k: v.to(device) for k, v in inputs.items()}

                        out_ids = model.generate(
                            **inputs,
                            **gen_kwargs,
                        )[0]

                        gen_only = out_ids[inputs['input_ids'].shape[-1]:]
                        txt = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
                        samples.append(txt)

            pred_norm = majority_vote(samples)
            # For logging also keep the raw first sample’s final span
            sample0_final = extract_final_span(samples[0]) if samples else ""

            # Optional verifier stats on the first sample (as your code had)
            steps = parse_steps_from_text(samples[0]) if samples else []
            verif = verifier_pass_fail(steps)
            verif_rate = sum(1 for v in verif if v.get('ok')) / max(1, len(verif))

            # If your verifier function expects raw text + gold with '####', keep as is:
            final_correct = check_final_answer_from_text(samples[0] if samples else "", gold)

            record = {
                'problem': raw_prob,
                'gold': gold,
                'pred': pred_norm,            # normalized voted prediction
                'sample0_final': sample0_final,
                'verif_rate': verif_rate,
                'final_correct': final_correct,
                'sample_0': samples[0] if samples else "",
                'n_samples': args.samples,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'max_new_tokens': args.max_new_tokens,
            }
            outfh.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f"  done in {time.time()-t0:.1f}s", flush=True)

    print(f"All done in {time.time()-t0_all:.1f}s -> {args.out}")

if __name__ == '__main__':
    main()
