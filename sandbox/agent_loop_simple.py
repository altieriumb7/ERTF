#!/usr/bin/env python3
# Simple Solver-Verifier loop demo using HF model + sympy
import argparse, json, re
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import sympy as sp

def parse_steps_from_text(text):
    parts = re.split(r'\n\s*\d+\)\s*', text)
    if len(parts) <= 1:
        return [l.strip() for l in text.splitlines() if l.strip()]
    return [p.strip() for p in parts[1:] if p.strip()]

def check_step(step):
    m = re.search(r'([0-9\+\-\*\/\^\(\)\s]+)=\s*([0-9\-\+]+)', step)
    if m:
        try:
            left = sp.sympify(m.group(1).replace('^','**'))
            right = sp.sympify(m.group(2))
            return sp.simplify(left-right) == 0
        except:
            return False
    return False

def build_prompt(problem):
    return f"[ROLE: Solver]\nProblem: {problem}\nInstruction: Provide numbered steps.\n1) "

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-2-7b')
    parser.add_argument('--input', default=None)
    args = parser.parse_args()
    if args.input:
        with open(args.input,'r') as f:
            problem = f.read().strip()
    else:
        problem = 'If John has 3 apples and gets 4 more, how many apples does he have?'
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, load_in_4bit=True, device_map='auto')
    gen_cfg = GenerationConfig(temperature=0.8, top_p=0.95)
    prompt = build_prompt(problem)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    out = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.8)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print('Solver output:', text)
    steps = parse_steps_from_text(text)
    for i,s in enumerate(steps):
        ok = check_step(s)
        print(f'Step {i+1}:', 'PASS' if ok else 'FAIL', '->', s)

if __name__ == '__main__':
    main()
