# eval_inference.py
import json, argparse, os
import torch
from transformers import GenerationConfig

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
    parser.add_argument('--base_model', default='meta-llama/Llama-2-7b')
    parser.add_argument('--adapter_dir', default='outputs/erft_sft')
    parser.add_argument('--data', default='data/maths_test.jsonl')
    parser.add_argument('--out', default='results/eval_traces.jsonl')
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    args = parser.parse_args()

    # Load tokenizer + base model with LoRA adapter applied
    tokenizer, model = load_model_with_adapter(args.base_model, args.adapter_dir)

    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    outfh = open(args.out, 'w', encoding='utf-8')

    # Choose device and move model if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model.to(device)
    except AttributeError:
        pass
    model.eval()

    # Make sure PEFT's generate() finds a GenerationConfig
    try:
        gen_cfg = GenerationConfig.from_pretrained(args.base_model)
    except Exception:
        gen_cfg = GenerationConfig()
    model.generation_config = gen_cfg

    data = read_jsonl(args.data)

    for ex in data:
        prob = canonicalize_problem(ex['problem'])
        prompt = build_prompt(prob)

        samples = []
        for _ in range(args.samples):
            inputs = tokenizer(prompt, return_tensors='pt')
            # move inputs to the same device we put the model on
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=args.max_new_tokens,
                top_p=0.95,
                temperature=0.8,
            )
            # Decode only the newly generated tokens
            gen_ids = out[0][inputs['input_ids'].shape[1]:]
            txt = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            # Fallback: if decoding only new tokens fails for some reason, clean by splitting
            if not txt:
                full_txt = tokenizer.decode(out[0], skip_special_tokens=True)
                if prompt in full_txt:
                    txt = full_txt.split(prompt, 1)[1].strip()
                else:
                    txt = full_txt.strip()

            samples.append(txt)

        pred = majority_vote(samples)
        steps = parse_steps_from_text(samples[0]) if samples else []
        verif = verifier_pass_fail(steps)
        verif_rate = sum(1 for v in verif if v.get('ok')) / max(1, len(verif))
        final_correct = check_final_answer_from_text(samples[0], ex.get('answer')) if samples else False

        record = {
            'problem': ex['problem'],
            'gold': ex.get('answer'),
            'pred': pred,
            'verif_rate': verif_rate,
            'final_correct': final_correct,
        }
        outfh.write(json.dumps(record, ensure_ascii=False) + '\n')

    outfh.close()
    print('Done. Results in', args.out)

if __name__ == '__main__':
    main()
