# eval_inference.py
import json, argparse, os
import torch

from load_adapter_for_inference import load_model_with_adapter
from verifier_improved import (
    parse_steps_from_text,
    verifier_pass_fail,
    check_final_answer_from_text,
    canonicalize_problem,
)

def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f]

def build_prompt(problem):
    return f"[ROLE: Solver]\nProblem: {problem}\nInstruction: Provide a CLAIM and numbered PROOF_SKETCH.\n1) "

def majority_vote(samples):
    import re
    for s in samples:
        nums = re.findall(r"-?\d+\.?\d*", s)
        if nums:
            return nums[-1]
    return samples[0] if samples else ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--adapter_dir', default='outputs/erft_sft')
    parser.add_argument('--data', default='data/maths_test.jsonl')
    parser.add_argument('--out', default='results/eval_traces.jsonl')
    parser.add_argument('--samples', type=int, default=4, help="# stochastic generations (votes) per problem; if 0, use 1")
    parser.add_argument('--max_new_tokens', type=int, default=128)
    args = parser.parse_args()

    # Load tokenizer + model with adapters already applied
    tokenizer, model = load_model_with_adapter(args.base_model, args.adapter_dir)

    # Choose device and move model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # some wrappers may not expose .to, but .to on Module is fine; ensure parameters exist
    try:
        model.to(device)
    except AttributeError:
        # Fallback: no-op if model can't be moved explicitly (unlikely)
        pass
    model.eval()

    data = read_jsonl(args.data)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    n_votes = max(1, args.samples)  # guard against 0
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with open(args.out, 'w', encoding='utf-8') as outfh:
        for ex in data:
            prob = canonicalize_problem(ex['problem'])
            prompt = build_prompt(prob)

            samples = []
            for _ in range(n_votes):
                inputs = tokenizer(prompt, return_tensors='pt')
                # move each tensor to the same device as the model
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        do_sample=True,
                        max_new_tokens=args.max_new_tokens,
                        top_p=0.95,
                        temperature=0.8,
                        pad_token_id=pad_id,
                    )
                txt = tokenizer.decode(out[0], skip_special_tokens=True)
                # strip the prompt if it was echoed
                if prompt in txt:
                    txt = txt.split(prompt, 1)[1].strip()
                samples.append(txt)

            pred = majority_vote(samples)

            # Verification on the first sample (or empty-safe)
            first = samples[0] if samples else ""
            steps = parse_steps_from_text(first)
            verif = verifier_pass_fail(steps)
            verif_rate = (sum(1 for v in verif if v.get('ok')) / max(1, len(verif))) if verif else 0.0

            # gold may be under 'solution' (gsm8k) or 'answer' (other)
            gold = ex.get('solution', ex.get('answer'))
            final_correct = check_final_answer_from_text(first, gold)

            record = {
                'problem': ex['problem'],
                'gold': gold,
                'pred': pred,
                'verif_rate': verif_rate,
                'final_correct': final_correct,
                'samples': samples,  # keep for debugging/analysis
            }
            outfh.write(json.dumps(record, ensure_ascii=False) + '\n')

    print('Done. Results in', args.out)

if __name__ == '__main__':
    main()
