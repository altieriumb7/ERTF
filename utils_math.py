import re, random, numpy as np

BOXED_RE = re.compile(r"\\boxed\\s*{([^}]*)}")

def extract_boxed(text: str) -> str:
    m = BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    last = text.strip().splitlines()[-1]
    return last.strip()

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def canonicalize_answer(ans: str) -> str:
    s = ans.strip()
    s = s.replace(" ", "")
    s = s.replace("\\frac{0}{", "0/")  
    s = s.replace("−", "-")
    return s
