#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
python3 -m venv "$ROOT_DIR/venv"
source "$ROOT_DIR/venv/bin/activate"
pip install --upgrade pip
if [ -f "$ROOT_DIR/requirements.txt" ]; then
  pip install -r "$ROOT_DIR/requirements.txt"
else
  pip install transformers accelerate bitsandbytes peft trl datasets sympy sentence-transformers faiss-cpu wandb
fi
if [ -n "${HF_TOKEN-}" ]; then
  echo "Logging into Hugging Face using HF_TOKEN..."
  pip install huggingface_hub
  huggingface-cli login --token "$HF_TOKEN"
fi
echo "Running minimal SFT smoke (1 epoch)..."
if [ "${SKIP_SFT-0}" -ne 1 ]; then
  if [ -f "$ROOT_DIR/train_sft.py" ]; then
    python "$ROOT_DIR/train_sft.py" || echo "train_sft.py failed; you can skip SFT by setting SKIP_SFT=1"
  else
    echo "No train_sft.py found; skipping SFT."
  fi
else
  echo "SKIP_SFT set; skipping SFT stage."
fi
echo "Patching rl_config.yaml for conservative smoke settings..."
python - <<'PY'
import ruamel.yaml as yaml, os
p='rl_config.yaml'
if os.path.exists(p):
    y=yaml.round_trip_load(open(p))
    y.setdefault('training',{})
    y['training']['total_updates']=5
    y.setdefault('ppo',{})
    y['ppo']['episodes_per_update']=2
    y.setdefault('generation',{})
    y['generation']['max_new_tokens']=64
    open(p,'w').write(yaml.round_trip_dump(y))
    print('Patched',p)
else:
    print('rl_config.yaml not found; skipping patch.')
PY
echo "Launching multi-turn PPO smoke run (conservative settings)..."
accelerate launch --config_file accelerate_config.yaml train_rl_trl_multiturn_with_val.py \
  --rl_config rl_config.yaml \
  --sft_model_dir outputs/erft_sft \
  --dataset data/maths_train.jsonl \
  --val_dataset data/maths_val.jsonl \
  --output_dir outputs/erft_rl_multiturn_val || echo "PPO run failed; check logs."
echo "Smoke run complete. Check outputs/erft_rl_multiturn_val for checkpoints and latest_ckpt_summary.json"
