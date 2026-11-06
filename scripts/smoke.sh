#!/usr/bin/env bash
# 6 games (one per task type), RL model, original env. Prints timing to extrapolate full runs.
set -euo pipefail
cd "$(dirname "$0")/.." && source scripts/env.sh
mkdir -p runs && rm -f runs/smoke.jsonl
start=$(date +%s)
python -m alfworld_eval.run_alfworld_eval --model "$RL" --tag rl --split seen --per_type 1 \
  --skills_json "$SKILLS" --perturb none --save_think --out runs/smoke.jsonl
end=$(date +%s)
python -m alfworld_eval.metrics runs/smoke.jsonl
echo "[smoke] 6 episodes in $((end-start)) s (includes model loading)"
