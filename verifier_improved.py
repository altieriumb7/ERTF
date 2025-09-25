# verifier_improved.py
import re, unicodedata
import sympy as sp
from typing import List

WORDS_TO_DIGITS = {
    'zero': '0','one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','seven':'7','eight':'8','nine':'9',
    'ten':'10','eleven':'11','twelve':'12'
}

def normalize_text(t: str) -> str:
    s = t.strip()
    s = unicodedata.normalize("NFKC", s)
    for w,d in WORDS_TO_DIGITS.items():
        s = re.sub(r'\b'+w+r'\b', d, s, flags=re.IGNORECASE)
    return s

def extract_final_numeric(text: str):
    nums = re.findall(r'-?\d+\.?\d*', text)
    if not nums:
        m = re.findall(r'\d+\s*/\s*\d+', text)
        if m:
            try:
                val = float(sp.Rational(m[-1].replace(' ','')))
                return str(val)
            except:
                return None
        return None
    return nums[-1]

def parse_steps_from_text(text: str) -> List[str]:
    parts = re.split(r'\n\s*\d+[)\.\:]\s*', text)
    if len(parts) <= 1:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        candidate_lines = [l for l in lines if re.search(r'[0-9\+\-\*\/\=]', l)]
        return candidate_lines[:12]
    return [p.strip() for p in parts[1:] if p.strip()][:12]

def check_arithmetic_equality(lhs_raw: str, rhs_raw: str, tol=1e-6):
    try:
        lhs = sp.sympify(lhs_raw.replace('^','**'))
        rhs = sp.sympify(rhs_raw.replace('^','**'))
        diff = sp.simplify(lhs - rhs)
        try:
            val = float(diff)
            return abs(val) < tol, f"diff={val}"
        except Exception:
            return diff == 0, f"diff_symbolic={diff}"
    except Exception as e:
        return False, f"parse_error:{e}"

def check_arithmetic_step(step: str):
    s = normalize_text(step)
    m = re.search(r'(.+?)=\s*(.+)$', s)
    if m:
        left = m.group(1).strip()
        right = m.group(2).strip()
        ok, msg = check_arithmetic_equality(left, right)
        return ok, msg
    m2 = re.search(r'compute\s*[:\-]?\s*([0-9\(\)\+\-\*\/\^\.\s,]+)', s, flags=re.I)
    if m2:
        expr = m2.group(1)
        try:
            val = float(sp.N(sp.sympify(expr.replace('^','**'))))
            return True, f"expr-> {val}"
        except Exception as e:
            return False, f"parse_err:{e}"
    num = extract_final_numeric(s)
    if num is not None:
        return True, f"found_numeric {num}"
    return False, "no_arithmetic_found"

def verifier_pass_fail(steps: List[str]):
    results=[]
    for s in steps:
        ok, msg = check_arithmetic_step(s)
        results.append({'step': s, 'ok': bool(ok), 'msg': msg})
    return results

def check_final_answer_from_text(text: str, gold_answer):
    pred = extract_final_numeric(text)
    if pred is None:
        return False
    if gold_answer is None:
        return False
    try:
        return abs(float(pred) - float(gold_answer)) < 1e-6
    except:
        return str(pred).strip() == str(gold_answer).strip()

def canonicalize_problem(text: str):
    s = text.replace('−','-')
    s = re.sub(r'\s+', ' ', s).strip()
    return s
