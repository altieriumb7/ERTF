# eval_inference.py
import json, argparse, os
from load_adapter_for_inference import load_model_with_adapter
from verifier_improved import parse_steps_from_text, verifier_pass_fail, check_final_answer_from_text, canonicalize_problem

def read_jsonl(path):
    with open(path,'r',encoding='utf-8') as f:
        return [json.loads(l) for l in f]

def build_prompt(problem):
    return f"[ROLE: Solver]\nProblem: {problem}\nInstruction: Provide a CLAIM and numbered PROOF_SKETCH.\n1) "

def majority_vote(samples):
    for s in samples:
        nums = __import__('re').findall(r'-?\d+\.?\d*', s)
        if nums:
            return nums[-1]
    return samples[0] if samples else ''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='meta-llama/Llama-2-7b')
    parser.add_argument('--adapter_dir', default='outputs/erft_sft')
    parser.add_argument('--data', default='data/maths_test.jsonl')
    parser.add_argument('--out', default='results/eval_traces.jsonl')
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    args = parser.parse_args()

    tokenizer, model = load_model_with_adapter(args.base_model, args.adapter_dir)
    data = read_jsonl(args.data)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    outfh = open(args.out, 'w', encoding='utf-8')

    for ex in data:
        prob = canonicalize_problem(ex['problem'])
        prompt = build_prompt(prob)
        samples = []
        for i in range(args.samples):
            inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
            out = model.generate(**inputs, do_sample=True, max_new_tokens=args.max_new_tokens, top_p=0.95, temperature=0.8)
            txt = tokenizer.decode(out[0], skip_special_tokens=True)
            if prompt in txt:
                txt = txt.split(prompt,1)[1].strip()
            samples.append(txt)
        pred = majority_vote(samples)
        steps = parse_steps_from_text(samples[0])
        verif = verifier_pass_fail(steps)
        verif_rate = sum(1 for v in verif if v['ok']) / max(1, len(verif))
        final_correct = check_final_answer_from_text(samples[0], ex.get('answer'))
        record = {'problem': ex['problem'], 'gold': ex.get('answer'), 'pred': pred, 'verif_rate': verif_rate, 'final_correct': final_correct}
        outfh.write(json.dumps(record, ensure_ascii=False) + '\n')
    outfh.close()
    print('Done. Results in', args.out)

if __name__ == '__main__':
    main()
