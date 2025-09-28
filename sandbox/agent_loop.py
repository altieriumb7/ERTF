cp src/sandbox/agent_loop.py src/sandbox/agent_loop.py.bak 2>/dev/null || true
cat > src/sandbox/agent_loop.py <<'PY'
#!/usr/bin/env python
# coding: utf-8
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
    p.add_argument("--max_rounds", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--sample", action="store_true")
    return p.parse_args()

def load_model_and_tokenizer(model_name: str):
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
    )
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.eval()
    return tok, model

def maybe_wrap_adapter(base_model, adapter_dir):
    if adapter_dir and os.path.isdir(adapter_dir):
        return PeftModel.from_pretrained(base_model, adapter_dir).eval()
    return base_model

def build_prompt(role, problem, history):
    head = f"[ROLE: {role.upper()}]\n"
    conv = "".join(f"[{r.upper()}]: {t}\n" for (r, t) in history)
    return f"{head}Problem: {problem}\nConversation so far:\n{conv}{head}Output:"

def split_steps(text: str):
    lines = [l.strip("-. \t") for l in text.strip().splitlines() if l.strip()]
    if len(lines) <= 1:
        parts = text.replace(";", "\n").splitlines()
        lines = [p.strip() for p in parts if p.strip()]
    return lines if lines else [text.strip()]

def check_step(step_text: str):
    try:
        if "=" in step_text:
            a, b = step_text.split("=", 1)
            return parse_expr(a.strip()).equals(parse_expr(b.strip()))
        parse_expr(step_text)
        return True
    except Exception:
        return False

def run_demo(problems, model_name, adapters, out_jsonl, max_rounds, max_new_tokens, use_sampling=False):
    tok, base = load_model_and_tokenizer(model_name)
    models = {
        "solver": maybe_wrap_adapter(base, adapters.get("solver")),
        "verifier": maybe_wrap_adapter(base, adapters.get("verifier")),
        "strategist": maybe_wrap_adapter(base, adapters.get("strategist")),
    }
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

    # STABLE defaults (greedy). Sampling only if --sample is passed.
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    if use_sampling:
        gen_kwargs.update(dict(do_sample=True, temperature=0.7, top_p=0.9))

    # keep prompt well below ctx window; with 256 new tokens, 3200 is very safe
    MAX_PROMPT_TOKENS = 3200

    def safe_generate(model_key, prompt_text):
        # truncate prompt hard
        inputs = tok(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS)
        inputs = {k: v.to(models[model_key].device) for k, v in inputs.items()}
        try:
            return models[model_key].generate(**inputs, **gen_kwargs)
        except RuntimeError:
            # fallback to strict greedy to avoid sampler math
            safe = dict(gen_kwargs, do_sample=False, temperature=None, top_p=None)
            return models[model_key].generate(**inputs, **safe)

    with open(out_jsonl, "a", encoding="utf-8") as fout:
        for prob in problems:
            convo = []
            ok_cnt, chk_cnt = 0, 0
            log = {"problem": prob, "turns": []}

            for _ in range(max_rounds):
                # SOLVER
                p = build_prompt("solver", prob, convo)
                out = safe_generate("solver", p)
                stext = tok.decode(out[0], skip_special_tokens=True)
                convo.append(("solver", stext))
                log["turns"].append({"role": "solver", "text": stext})
                for step in split_steps(stext):
                    chk_cnt += 1
                    ok_cnt += int(check_step(step))

                # VERIFIER
                p = build_prompt("verifier", prob, convo)
                out = safe_generate("verifier", p)
                vtext = tok.decode(out[0], skip_special_tokens=True)
                convo.append(("verifier", vtext))
                log["turns"].append({"role": "verifier", "text": vtext})

                # STRATEGIST
                p = build_prompt("strategist", prob, convo)
                out = safe_generate("strategist", p)
                ttext = tok.decode(out[0], skip_special_tokens=True)
                convo.append(("strategist", ttext))
                log["turns"].append({"role": "strategist", "text": ttext})

                if chk_cnt > 0 and ok_cnt == chk_cnt:
                    break

            log["verifier_accept_rate"] = ok_cnt / max(1, chk_cnt)
            fout.write(json.dumps(log, ensure_ascii=False) + "\n")
    print(f"[OK] Dialogs saved to {out_jsonl}")

if __name__ == "__main__":
    args = parse_args()
    adapters = {"solver": args.adapter_solver, "verifier": args.adapter_verifier, "strategist": args.adapter_strategist}
    PROBLEMS = [
        "Compute 3 + 4 * 2.",
        "If 2x + 3 = 11, what is x?",
        "A rectangle has sides 3 and 5. What is its area?",
    ]
    run_demo(PROBLEMS, args.model_name_or_path, adapters, args.out_jsonl, args.max_rounds, args.max_new_tokens, use_sampling=args.sample)
PY
