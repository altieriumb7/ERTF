#!/usr/bin/env bash
set -euo pipefail

ROLE="${ROLE:-solver}"   # solver | verifier | strategist
MODEL_ID="${MODEL_ID:-meta-llama/Llama-2-7b-hf}"
DATA="${DATA:-data/sft_dialogs_${ROLE}_train.jsonl}"
OUTDIR="${OUTDIR:-adapters/agent_${ROLE}}"
EPOCHS="${EPOCHS:-1}"
BATCH="${BATCH:-1}"
GRAD_ACC="${GRAD_ACC:-16}"
LR="${LR:-2e-4}"
MAXLEN="${MAXLEN:-1024}"

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

mkdir -p "$(dirname "$OUTDIR")"

echo "[run_sft] role=$ROLE  model=$MODEL_ID"
accelerate launch train_sft.py \
  --model_name_or_path "$MODEL_ID" \
  --data "$DATA" \
  --output_dir "$OUTDIR" \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$BATCH" \
  --gradient_accumulation_steps "$GRAD_ACC" \
  --learning_rate "$LR" \
  --max_length "$MAXLEN"
