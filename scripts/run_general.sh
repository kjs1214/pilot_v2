#!/usr/bin/env bash
# General-skill experiment (the two questions only):
#  (1) premise, in the training dataset (ALFWorld): does removing the GENERAL skills change performance?
#      full bank vs. bank without general skills (task-specific + mistakes kept), seen and unseen, several seeds
#  (2) other datasets with a shared procedure: does adding the SAME general skills (verbatim) help?
#      none vs. general, TWC / CookingWorld / ScienceWorld, several seeds
# Models: rl (trained) and calib (untrained reference). Usage:
#   SEEDS="0 1 2" WORKSPACE_SLUG=... nohup bash scripts/run_general.sh > runs_gen/log.txt 2>&1 &
# TextWorldExpress settings (TWC_PARAMS / CW_PARAMS): preset names or raw gameParams strings
#   default = LIBRARY DEFAULT: twc 3 rooms, 4 items | cookingworld 11 rooms, 3 ingredients, 10 distractors, doors, inventory limit
#   reduced = twc 3 rooms, 2 items | cookingworld 3 rooms, 2 ingredients, 5 distractors, no doors   (early logs called this "original": it is not)
#   easy    = 1 room, 1 item / 1 ingredient
# ScienceWorld runs at LIBRARY DEFAULTS: no simplification (SW_SIMPL=""), 100-step limit (SW_MAX_STEPS=100), scored by its
#   progress score. The only harness choice is the shown action list (SW_LIST=held), validated by analysis/gold_check.py.
#   Non-default options exist (SW_SIMPL=noElectricalAction SW_MAX_STEPS=200 SW_TICKS=500) but are not used in the paper runs.
set -uo pipefail
cd "$(dirname "$0")/.." && source scripts/env.sh
OUT=${OUT:-runs_gen}; SEEDS=${SEEDS:-0 1 2}; MODELS=${MODELS:-rl calib}
mkdir -p "$OUT/alfworld" "$OUT/transfer"
{ [ -f configs/skills_nogeneral.json ] && [ -f configs/skills_nogeneral6.json ] && [ -f configs/skills_genonly.json ] && [ -f configs/skills_genonly6.json ] && [ -f configs/skills_placebo.json ]; } || python scripts/make_skill_variants.py --src "$SKILLS" --out configs
pause_ws () { if [ -n "${WORKSPACE_SLUG:-}" ]; then vesslctl workspace pause "$WORKSPACE_SLUG"; fi; }
trap pause_ws EXIT
model_of () { case $1 in rl) echo "$RL";; base) echo "$BASE";; calib) echo "$CALIB";; esac; }
extra_of () { [ "$1" = calib ] && echo "--chat_template_kwargs {\"enable_thinking\":false} --gpu_mem 0.92" || true; }
for seed in $SEEDS; do
  for tag in $MODELS; do
    for split in ${SPLITS-seen unseen}; do
      for bank in ${BANKS:-full nogen}; do
        out="$OUT/alfworld/${tag}_${split}_${bank}_s${seed}.jsonl"; [ -s "$out" ] && continue
        # with task-specific skills: full | nogen (11 general items removed) | nogen6 (6 general skills removed)
        # without task-specific skills (matched to other datasets): nosk (no skills) | genonly6 (6 general only) | genonly (11 general items only)
        case $bank in
          full) skarg=(--skills_json "$SKILLS");;
          nogen) skarg=(--skills_json configs/skills_nogeneral.json);;
          nogen6) skarg=(--skills_json configs/skills_nogeneral6.json);;
          genonly) skarg=(--skills_json configs/skills_genonly.json);;
          genonly6) skarg=(--skills_json configs/skills_genonly6.json);;
          placebo) skarg=(--skills_json configs/skills_placebo.json);;   # WebShop general skills + mistakes: same format, unrelated content
          nosk) skarg=();;
          *) echo "unknown bank $bank"; exit 1;;
        esac
        echo "[$(date '+%F %T')] alfworld $tag $split $bank seed=$seed"
        python -m alfworld_eval.run_alfworld_eval --model "$(model_of $tag)" --tag $tag --split $split --reps 1 --seed $seed \
          "${skarg[@]}" --perturb none --max_games ${MAX_GAMES:-0} --save_think $(extra_of $tag) --out "$out.tmp" && mv "$out.tmp" "$out"
      done
    done
    for env in ${TRANSFER_ENVS-twc cookingworld scienceworld}; do
      for sk in ${TRANSFER_SKILLS:-none general6 general}; do
        out="$OUT/transfer/${env}_${tag}_${sk}_s${seed}.jsonl"; [ -s "$out" ] && continue
        envopt=(); case $env in
          scienceworld) envopt=(--simplifications "${SW_SIMPL-}" --max_steps ${SW_MAX_STEPS:-100} --sw_tick_limit ${SW_TICKS:-0} --sw_list ${SW_LIST:-held} --max_model_len ${SW_MAX_MODEL_LEN:-16384});;
          twc) envopt=(--game_params "${TWC_PARAMS:-default}");;         # preset: default (library default) | reduced | easy, or a raw string
          cookingworld) envopt=(--game_params "${CW_PARAMS:-default}");;
        esac
        echo "[$(date '+%F %T')] transfer $env $tag $sk seed=$seed"
        python run_eval.py --env $env --model "$(model_of $tag)" --tag $tag --skills $sk --split test \
          --max_variations ${TRANSFER_VARS:-20} --reps 1 --seed $seed --parallel ${PARALLEL:-8} --save_think \
          --max_steps ${MAX_STEPS:-50} --max_actions ${MAX_ACTIONS:-150} --history_length ${HISTORY:-2} --max_model_len ${MAX_MODEL_LEN:-8192} \
          $(extra_of $tag) "${envopt[@]}" --out "$out.tmp" && mv "$out.tmp" "$out"
      done
    done
  done
done
python analysis/general_report.py "$OUT" | tee "$OUT/report.txt"
