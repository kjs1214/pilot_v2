#!/usr/bin/env bash
# One-time setup inside a VESSL workspace. Usage: bash scripts/setup_workspace.sh
# - Creates a Python 3.10-3.12 venv at $WORK/venv with uv (image Python if compatible, else managed 3.12) (TextWorld/ALFWorld break on Python 3.13:
#   textworld's PDDL text generator raises NameError under 3.13), so the image's own Python version does not matter.
# - Installs vLLM in that fresh venv (vLLM picks its own torch/CUDA build), plus the env packages.
# - Downloads ALFWorld data, SkillRL code, SFT/RL checkpoints, merges the RL FSDP shards.
set -euo pipefail
W=${WORK:-$HOME}
export ALFWORLD_DATA=${ALFWORLD_DATA:-$W/alfworld}
PILOT=$(cd "$(dirname "$0")/.." && pwd)
echo "[setup] WORK=$W  ALFWORLD_DATA=$ALFWORLD_DATA  PILOT=$PILOT"

# 1) uv + Python 3.12 venv
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install -q uv 2>/dev/null || pip install -q uv 2>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
if [ ! -x "$W/venv/bin/python" ]; then
  # use the image's Python if it is 3.10-3.12, otherwise let uv fetch a managed 3.12
  if python3 -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' 2>/dev/null; then
    uv venv --python "$(command -v python3)" --seed "$W/venv"
  else
    uv venv --python 3.12 --seed "$W/venv"
  fi
fi
# shellcheck disable=SC1091
source "$W/venv/bin/activate"
python --version

# 2) packages (vLLM first so it resolves its own torch; fall back to pip if uv's torch-backend flag is unavailable)
uv pip install vllm --torch-backend=auto || pip install vllm
uv pip install textworld alfworld scienceworld textworld_express sentence-transformers huggingface_hub safetensors
python - <<'PY'
import importlib, sys
for m in ["vllm", "torch", "transformers", "textworld", "alfworld", "scienceworld", "textworld_express", "sentence_transformers"]:
    mod = importlib.import_module(m); print(f"[setup] {m} {getattr(mod, '__version__', '')}")
import torch; print("[setup] cuda available:", torch.cuda.is_available())
PY

# 3) code + data + models
cd "$W"
[ -d SkillRL ] || git clone https://github.com/aiming-lab/SkillRL.git
mkdir -p "$ALFWORLD_DATA"
[ -d "$ALFWORLD_DATA/json_2.1.1/valid_seen" ] || alfworld-download --data-dir "$ALFWORLD_DATA"
if [ ! -f "$W/ckpt/rl_merged/config.json" ] || [ ! -d "$W/ckpt/sft" ]; then
  python "$PILOT/scripts/download_models.py" --dest "$W/ckpt"
fi
if [ ! -f "$W/ckpt/rl_merged/config.json" ]; then
  python "$PILOT/scripts/merge_fsdp_ckpt.py" --local_dir "$W/ckpt/rl_raw/actor" --target_dir "$W/ckpt/rl_merged"
  rm -rf "$W/ckpt/rl_raw"
fi
[ -f "$PILOT/configs/skills_mismatched.json" ] || python "$PILOT/scripts/make_skill_variants.py" \
  --src "$W/SkillRL/memory_data/alfworld/claude_style_skills.json" --out "$PILOT/configs"

# 4) Java for ScienceWorld
if ! java -version >/dev/null 2>&1; then
  (apt-get update -qq && apt-get install -y -qq default-jre-headless) || echo "[warn] Java not installed: ScienceWorld needs it"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv || true
df -h "$W" | tail -1
echo "[setup] done. venv: $W/venv  SFT: $W/ckpt/sft/checkpoint-140  RL: $W/ckpt/rl_merged"
