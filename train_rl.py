from scripts.eval_math import compare  # import compare function

def compute_episode_reward(ep_record, problem_id=None, gold_map=None, w_final=1.0, w_verif=0.5, w_round=0.05):
    # final correctness: use last solver final string vs gold_map[problem_id]
    final_verified = ep_record.get("final_verified", False)
    # If gold_map available, compare last solver output to gold
    if gold_map and problem_id in gold_map:
        # last solver response
        last_solver = None
        for (role, text) in reversed(ep_record["turns"]):
            if role == "solver":
                last_solver = text; break
        ok, reason = compare(last_solver, gold_map[problem_id])
        final_correct = 1.0 if ok else 0.0
    else:
        final_correct = 1.0 if final_verified else 0.0

    verif_frac = ep_record.get("verifier_ok_frac", 0.0)
    rounds_used = sum(1 for (r, _) in ep_record.get("turns", []) if r == "solver")
    reward = w_final * final_correct + w_verif * verif_frac - w_round * rounds_used
    return float(reward)
