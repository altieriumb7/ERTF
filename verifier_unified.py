from dataclasses import dataclass
from typing import Optional
import math
from utils_math import extract_boxed, canonicalize_answer

@dataclass
class VerifierConfig:
    strict_box: bool = True
    length_penalty: float = 0.0

class Verifier:
    """Unified verifier returning a score in [0,1]."""

    def __init__(self, cfg: Optional[VerifierConfig] = None):
        self.cfg = cfg or VerifierConfig()

    def score(self, prompt: str, scratchpad: str, final_answer: str, meta: dict | None = None) -> float:
        if self.cfg.strict_box and ("\\boxed" not in scratchpad):
            return 0.0

        boxed = extract_boxed(scratchpad)
        a1 = canonicalize_answer(boxed)
        a2 = canonicalize_answer(final_answer)
        agree = float(a1 == a2)

        n = len(scratchpad)
        lp = 1.0
        if self.cfg.length_penalty > 0.0:
            lp = math.exp(-self.cfg.length_penalty * abs(n - 800) / 800)

        return max(0.0, min(1.0, 0.5 * agree + 0.5 * lp))
