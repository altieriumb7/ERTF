#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_competition_math.py \
  --hf_path qwedsacf/competition_math \
  --out_dir data/competition_math

# Example SFT (optional): adjust to your training script
# python train_sft.py \
#   --base_model meta-llama/Llama-2-7b-hf \
#   --data data/competition_math/train_sft.jsonl \
#   --save_dir outputs/cm-sft-llama2-7b

python eval_inference_cm.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --hf_path qwedsacf/competition_math \
  --split test \
  --samples 8 --vgd --cove \
  --max_new_tokens 512 \
  --out_csv outputs/competition_math_eval.csv

echo "Results written to outputs/competition_math_eval.csv"
