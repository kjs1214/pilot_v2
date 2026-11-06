# source scripts/env.sh  (paths used by all run scripts; activates the venv made by setup_workspace.sh)
export WORK=${WORK:-$HOME}
if [ -f "$WORK/venv/bin/activate" ] && [ -z "${VIRTUAL_ENV:-}" ]; then source "$WORK/venv/bin/activate"; fi
export SKILLRL_ROOT=${SKILLRL_ROOT:-$WORK/SkillRL}
export ALFWORLD_DATA=${ALFWORLD_DATA:-$WORK/alfworld}
export SKILLS=${SKILLS:-$SKILLRL_ROOT/memory_data/alfworld/claude_style_skills.json}
export SFT=${SFT:-$WORK/ckpt/sft/checkpoint-140}
export RL=${RL:-$WORK/ckpt/rl_merged}
export BASE=${BASE:-Qwen/Qwen2.5-7B-Instruct}
export CALIB=${CALIB:-Qwen/Qwen3-30B-A3B}
export PYTHONPATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd):${PYTHONPATH:-}
python -c 'import sys; v=sys.version_info; sys.exit(0 if v < (3,13) else 1)' || echo "[env] WARNING: Python $(python --version) — TextWorld/ALFWorld need <=3.12. Run scripts/setup_workspace.sh (it creates ~/venv with 3.12)."
