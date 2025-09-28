#!/usr/bin/env bash
set -euo pipefail

# --- config you can tweak ---
MODEL_ID="${MODEL_ID:-meta-llama/Llama-2-7b-hf}"
OUT="${OUT:-results/dialogs_llama2_7b.jsonl}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
MAX_NEW="${MAX_NEW:-256}"
SAMPLE="${SAMPLE:-false}"   # set to "true" for sampling

# activate venv (adjust if your venv path differs)
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

mkdir -p results

CMD=( python src/sandbox/agent_loop.py
  --model_name_or_path "$MODEL_ID"
  --out_jsonl "$OUT"
  --max_rounds "$MAX_ROUNDS"
  --max_new_tokens "$MAX_NEW"
)

if [ "$SAMPLE" = "true" ]; then
  CMD+=( --sample )
fi

echo "[run_demo] model=$MODEL_ID  out=$OUT  rounds=$MAX_ROUNDS  max_new=$MAX_NEW  sample=$SAMPLE"
"${CMD[@]}"
