#!/usr/bin/env bash
set -euo pipefail

ROLE="${ROLE:-solver}"   # solver | verifier | strategist
MODEL_ID="${MODEL_ID:-meta-llama/Llama-2-7b-hf}"
PROBLEMS="${PROBLEMS:-data/problems_small.jsonl}"
OUTDIR="${OUTDIR:-ppo_checkpoints}"
BATCH_EPS="${BATCH_EPS:-4}"
UPDATES="${UPDATES:-50}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

mkdir -p "$OUTDIR"

echo "[run_rl] role=$ROLE  model=$MODEL_ID  problems=$PROBLEMS"
accelerate launch train_rl.py \
  --model_name_or_path "$MODEL_ID" \
  --train_role "$ROLE" \
  --adapter_dir_train "adapters/agent_${ROLE}_train" \
  --problems_jsonl "$PROBLEMS" \
  --output_dir "$OUTDIR" \
  --batch_episodes "$BATCH_EPS" \
  --total_updates "$UPDATES" \
  --ppo_epochs "$PPO_EPOCHS"
