
import os, json, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sympy.parsing.sympy_parser import parse_expr

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--adapter_solver", type=str, default="adapters/agent_solver")
    p.add_argument("--adapter_verifier", type=str, default="adapters/agent_verifier")
    p.add_argument("--adapter_strategist", type=str, default="adapters/agent_strategist")
    p.add_argument("--out_jsonl", type=str, default="results/dialogs.jsonl")
    p.add_argument("--max_rounds", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=64)
    return p.parse_args()

def load_model_and_tokenizer(model_name):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        max_memory={"cuda:0": "20GiB", "cpu": "48GiB"},
    )
    model.config.use_cache = False  # turn OFF KV cache
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.eval()
    return tok, model

def build_prompt(role, problem, history):
    head = f"[ROLE: {role.upper()}]\n"
    conv = "".join(f"[{r.upper()}]: {t}\n" for (r,t) in history)
    return f"{head}Problem: {problem}\nConversation so far:\n{conv}{head}Output:"

def run_demo(problems, model_name, adapters, out_jsonl, max_rounds, max_new_tokens):
    tok, model = load_model_and_tokenizer(model_name)
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,          # greedy only
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        use_cache=False,
    )

    MAX_PROMPT_TOKENS = 1024

    def safe_generate(prompt_text):
        inputs = tok(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS)
        inputs = {k: v.to(model.device) for k,v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        return tok.decode(out[0], skip_special_tokens=True)

    with open(out_jsonl, "a", encoding="utf-8") as fout:
        for prob in problems:
            convo = []
            log = {"problem": prob, "turns": []}
            for _ in range(max_rounds):
                for role in ["solver","verifier","strategist"]:
                    p = build_prompt(role, prob, convo)
                    text = safe_generate(p)
                    convo.append((role,text))
                    log["turns"].append({"role":role,"text":text})
            fout.write(json.dumps(log, ensure_ascii=False)+"\n")
    print(f"[OK] Saved to {out_jsonl}")

if __name__=="__main__":
    args = parse_args()
    adapters = {"solver":args.adapter_solver,"verifier":args.adapter_verifier,"strategist":args.adapter_strategist}
    problems = ["Compute 3+4*2.","Solve 2x+3=11.","Rectangle sides 3 and 5, area?"]
    run_demo(problems,args.model_name_or_path,adapters,args.out_jsonl,args.max_rounds,args.max_new_tokens)
