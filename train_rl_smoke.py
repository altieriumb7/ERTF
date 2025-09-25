#!/usr/bin/env python3
# Minimal RL loop skeleton (sampling and computing simple rewards) for smoke testing.
import json, random, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from verifier import parse_steps_from_text, verifier_pass_fail, check_final_answer_from_text

def read_jsonl(path):
    with open(path,'r') as f:
        return [json.loads(l) for l in f]

def build_prompt(problem):
    return f"[ROLE: Solver]\nProblem: {problem}\nInstruction: Provide numbered steps.\n1) "

def run_sample(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    out = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.8, top_p=0.95)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text

def compute_reward(text, obj):
    steps = parse_steps_from_text(text)
    verif = verifier_pass_fail(steps)
    verif_rate = sum(1 for v in verif if v['ok']) / max(1, len(verif))
    final_correct = check_final_answer_from_text(text, obj.get('answer'))
    reward = 1.0 * final_correct + 0.5 * verif_rate
    return reward, verif_rate, final_correct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-2-7b')
    parser.add_argument('--data', default='data/maths_train.jsonl')
    parser.add_argument('--iters', type=int, default=10)
    args = parser.parse_args()
    data = read_jsonl(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, load_in_4bit=True, device_map='auto')
    for i in range(args.iters):
        ex = random.choice(data)
        prompt = build_prompt(ex['problem'])
        text = run_sample(model, tokenizer, prompt)
        reward, verif_rate, final_correct = compute_reward(text, ex)
        print(f'Iter {i+1}: reward={reward:.3f}, verif_rate={verif_rate:.2f}, final_correct={final_correct}')
        print('Prompt:', ex['problem'])
        print('Output:', text)
        print('---')

if __name__=='__main__':
    main()
