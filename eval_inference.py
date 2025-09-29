# eval_inference.py
import json, argparse, os, re, torch, time
from load_adapter_for_inference import load_model_with_adapter
from verifier_improved import parse_steps_from_text, verifier_pass_fail, check_final_answer_from_text, canonicalize_problem

def read_jsonl(path):
    with open(path,'r',encoding='utf-8') as f:
        return [json.loads(l) for l in f]

def build_prompt(problem):
    return f"[ROLE: Solver]\nProblem: {problem}\nInstruction: Provide a CLAIM and numbered PROOF_SKETCH.\n1) "

def majority_vote(samples):
    for s in samples:
        nums = re.findall(r'-?\d+\.?\d*', s)
        if nums:
            return nums[-1]
    return samples[0] if samples else ''

def _get_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--adapter_dir', default='outputs/erft_sft')
    parser.add_argument('--data', default='data/maths_test.jsonl')
    parser.add_argument('--out', default='results/eval_traces.jsonl')
    parser.add_argument('--samples', type=int, default=1)            # <- start small
    parser.add_argument('--max_new_tokens', type=int, default=128)   # <- start small
    parser.add_argument('--limit', type=int, default=3, help='eval only this many items')  # <- new
    parser.add_argument('--no_4bit', action='store_true')
    args = parser.parse_args()

    print("Loading model...")
    tokenizer, model = load_model_with_adapter(args.base_model, args.adapter_dir, load_in_4bit=not args.no_4bit)
    device = _get_device(model)
    print("Model device:", device)

    data = read_jsonl(args.data)
    if args.limit is not None:
        data = data[:args.limit]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    gen_pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    torch.set_grad_enabled(False)

    t0_all = time.time()
    with open(args.out, 'w', encoding='utf-8') as outfh:
        for idx, ex in enumerate(data):
            t0 = time.time()
            prob = canonicalize_problem(ex['problem'])
            prompt = build_prompt(prob)
            print(f"[{idx+1}/{len(data)}] generating... (len(prompt)={len(prompt)})", flush=True)

            samples = []
            with torch.inference_mode():
                # autocast to fp16 on GPU
                amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if device.type == "cuda" else nullcontext()
                from contextlib import nullcontext
                with amp_ctx:
                    for s in range(args.samples):
                        inputs = tokenizer(prompt, return_tensors='pt', padding=False)
                        inputs = {k: v.to(device) for k, v in inputs.items()}

                        out_ids = model.generate(
                            **inputs,
                            do_sample=True,
                            top_p=0.95,
                            temperature=0.8,
                            max_new_tokens=args.max_new_tokens,
                            pad_token_id=gen_pad_id,
                            eos_token_id=tokenizer.eos_token_id,
                            use_cache=True,
                        )[0]

                        gen_only = out_ids[inputs['input_ids'].shape[-1]:]
                        txt = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
                        samples.append(txt)

            pred = majority_vote(samples)
            steps = parse_steps_from_text(samples[0]) if samples else []
            verif = verifier_pass_fail(steps)
            verif_rate = sum(1 for v in verif if v['ok']) / max(1, len(verif))
            final_correct = check_final_answer_from_text(samples[0] if samples else "", ex.get('answer'))

            record = {
                'problem': ex['problem'],
                'gold': ex.get('answer'),
                'pred': pred,
                'verif_rate': verif_rate,
                'final_correct': final_correct,
                'sample_0': samples[0] if samples else ""
            }
            outfh.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f"  done in {time.time()-t0:.1f}s", flush=True)

    print(f"All done in {time.time()-t0_all:.1f}s -> {args.out}")

if __name__ == '__main__':
    main()
