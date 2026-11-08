#!/usr/bin/env bash
# General-skill experiment (the two questions only):
#  (1) premise, in the training dataset (ALFWorld): does removing the GENERAL skills change performance?
#      full bank vs. bank without general skills (task-specific + mistakes kept), seen and unseen, several seeds
#  (2) other datasets with a shared procedure: does adding the SAME general skills (verbatim) help?
#      none vs. general, TWC / CookingWorld / ScienceWorld, several seeds
# Models: rl (trained) and calib (untrained reference). Usage:
#   SEEDS="0 1 2" WORKSPACE_SLUG=... nohup bash scripts/run_general.sh > runs_gen/log.txt 2>&1 &
# Original (unsimplified) other-dataset settings: SW_SIMPL="" TWC_PARAMS="numLocations=3, numItemsToPutAway=2, includeDoors=0, limitInventorySize=0"
#   CW_PARAMS="numLocations=3, numIngredients=2, numDistractorItems=5, includeDoors=0, limitInventorySize=0" MAX_ACTIONS=0 MAX_STEPS=100 HISTORY=5
set -uo pipefail
cd "$(dirname "$0")/.." && source scripts/env.sh
OUT=${OUT:-runs_gen}; SEEDS=${SEEDS:-0 1 2}; MODELS=${MODELS:-rl calib}
mkdir -p "$OUT/alfworld" "$OUT/transfer"
[ -f configs/skills_nogeneral.json ] || python scripts/make_skill_variants.py --src "$SKILLS" --out configs
pause_ws () { if [ -n "${WORKSPACE_SLUG:-}" ]; then vesslctl workspace pause "$WORKSPACE_SLUG"; fi; }
trap pause_ws EXIT
model_of () { case $1 in rl) echo "$RL";; base) echo "$BASE";; calib) echo "$CALIB";; esac; }
extra_of () { [ "$1" = calib ] && echo "--chat_template_kwargs {\"enable_thinking\":false} --gpu_mem 0.92" || true; }
for seed in $SEEDS; do
  for tag in $MODELS; do
    for split in ${SPLITS-seen unseen}; do
      for bank in full nogen; do
        out="$OUT/alfworld/${tag}_${split}_${bank}_s${seed}.jsonl"; [ -s "$out" ] && continue
        sk="$SKILLS"; [ $bank = nogen ] && sk=configs/skills_nogeneral.json
        echo "[$(date '+%F %T')] alfworld $tag $split $bank seed=$seed"
        python -m alfworld_eval.run_alfworld_eval --model "$(model_of $tag)" --tag $tag --split $split --reps 1 --seed $seed \
          --skills_json "$sk" --perturb none --max_games ${MAX_GAMES:-0} --save_think $(extra_of $tag) --out "$out.tmp" && mv "$out.tmp" "$out"
      done
    done
    for env in ${TRANSFER_ENVS-twc cookingworld scienceworld}; do
      for sk in ${TRANSFER_SKILLS:-none general}; do
        out="$OUT/transfer/${env}_${tag}_${sk}_s${seed}.jsonl"; [ -s "$out" ] && continue
        envopt=(); case $env in
          scienceworld) envopt=(--simplifications "${SW_SIMPL-noElectricalAction,openDoors,teleportAction}");;
          twc) envopt=(--game_params "${TWC_PARAMS:-numLocations=1, numItemsToPutAway=1, includeDoors=0, limitInventorySize=0}");;
          cookingworld) envopt=(--game_params "${CW_PARAMS:-numLocations=1, numIngredients=1, numDistractorItems=3, includeDoors=0, limitInventorySize=0}");;
        esac
        echo "[$(date '+%F %T')] transfer $env $tag $sk seed=$seed"
        python run_eval.py --env $env --model "$(model_of $tag)" --tag $tag --skills $sk --split test \
          --max_variations ${TRANSFER_VARS:-20} --reps 1 --seed $seed --parallel ${PARALLEL:-8} --save_think \
          --max_steps ${MAX_STEPS:-50} --max_actions ${MAX_ACTIONS:-150} --history_length ${HISTORY:-2} \
          $(extra_of $tag) "${envopt[@]}" --out "$out.tmp" && mv "$out.tmp" "$out"
      done
    done
  done
done
python analysis/general_report.py "$OUT" | tee "$OUT/report.txt"
