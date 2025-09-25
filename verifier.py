import re, sympy as sp
def parse_steps_from_text(text):
    parts = re.split(r'\n\s*\d+\)\s*', text)
    if len(parts) <= 1:
        return [l.strip() for l in text.splitlines() if l.strip()]
    return [p.strip() for p in parts[1:] if p.strip()]

def verifier_pass_fail(steps):
    results=[]
    for s in steps:
        m = re.search(r'([0-9\+\-\*\/\^\(\)\s]+)=\s*([0-9\-\+]+)', s)
        if m:
            try:
                left = sp.sympify(m.group(1).replace('^','**'))
                right = sp.sympify(m.group(2))
                ok = sp.simplify(left-right) == 0
                results.append({'step': s, 'ok': bool(ok)})
            except:
                results.append({'step': s, 'ok': False})
        else:
            results.append({'step': s, 'ok': False})
    return results

def check_final_answer_from_text(text, gold):
    import re
    nums = re.findall(r'-?\d+\.?\d*', text)
    if not nums: return False
    pred = nums[-1]
    return str(pred) == str(gold)
