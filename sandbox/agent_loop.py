import os, argparse, torch, json
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
    p.add_argument("--max_rounds", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=256)
    return p.parse_args()

def load_model(model_name):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map="auto")
    model.eval()
    return tok, model

def wrap_with_adapter(base_model, adapter_dir):
    if os.path.isdir(adapter_dir):
        return PeftModel.from_pretrained(base_model, adapter_dir).eval()
    return base_model

def build_prompt(role, problem, history):
    head = f"[ROLE: {role.upper()}]\n"
    conv = "".join([f"[{r.upper()}]: {t}\n" for r,t in history])
    return f"{head}Problem: {problem}\nConversation so far:\n{conv}{head}Output:"

def split_steps(text):
    lines = [l.strip("-. \t") for l in text.strip().splitlines() if l.strip()]
    if len(lines) <= 1:
        parts = text.replace(";", "\n").splitlines()
        lines = [p.strip() for p in parts if p.strip()]
    return lines if lines else [text.strip()]

def check_step(s):
    try:
        if "=" in s:
            a,b = s.split("=",1)
            return parse_expr(a.strip()).equals(parse_expr(b.strip()))
        parse_expr(s)
        return True
    except Exception:
        return False

def run_demo(problems, model_name, adapters, out_jsonl, max_rounds, max_new_tokens):
    tok, base = load_model(model_name)
    models = {role: wrap_with_adapter(base, path) for role, path in adapters.items()}
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8, top_p=0.95)

    with open(out_jsonl, "a", encoding="utf-8") as fout:
        for prob in problems:
            convo, ok_cnt, chk_cnt = [], 0, 0
            log = {"problem": prob, "turns": []}
            for _ in range(max_rounds):
                # SOLVER
                p = build_prompt("solver", prob, convo)
                out = models["solver"].generate(**tok(p, return_tensors="pt").to(models["solver"].device), **gen_kwargs)
                stext = tok.decode(out[0], skip_special_tokens=True)
                convo.append(("solver", stext)); log["turns"].append({"role":"solver","text":stext})
                # check steps
                for step in split_steps(stext):
                    chk_cnt += 1
                    ok_cnt += int(check_step(step))

                # VERIFIER
                p = build_prompt("verifier", prob, convo)
                out = models["verifier"].generate(**tok(p, return_tensors="pt").to(models["verifier"].device), **gen_kwargs)
                vtext = tok.decode(out[0], skip_special_tokens=True)
                convo.append(("verifier", vtext)); log["turns"].append({"role":"verifier","text":vtext})

                # STRATEGIST
                p = build_prompt("strategist", prob, convo)
                out = models["strategist"].generate(**tok(p, return_tensors="pt").to(models["strategist"].device), **gen_kwargs)
                ttext = tok.decode(out[0], skip_special_tokens=True)
                convo.append(("strategist", ttext)); log["turns"].append({"role":"strategist","text":ttext})

                if chk_cnt>0 and ok_cnt==chk_cnt:
                    break
            log["verifier_accept_rate"] = ok_cnt/max(1,chk_cnt)
            fout.write(json.dumps(log, ensure_ascii=False)+"\n")
    print(f"[OK] Dialogs saved to {out_jsonl}")

if __name__=="__main__":
    args = parse_args()
    adapters = {"solver":args.adapter_solver, "verifier":args.adapter_verifier, "strategist":args.adapter_strategist}
    problems = [
        "Compute 3 + 4 * 2.",
        "If 2x + 3 = 11, what is x?",
        "A rectangle has sides 3 and 5. What is its area?"
    ]
    run_demo(problems, args.model_name_or_path, adapters, args.out_jsonl, args.max_rounds, args.max_new_tokens)
