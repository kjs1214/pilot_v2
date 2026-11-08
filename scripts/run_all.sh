#!/usr/bin/env bash
# Usage: WORKSPACE_SLUG=<slug> nohup bash scripts/run_all.sh [surface|skillgap|calib|necessity|transfer|all] > runs/run_all.log 2>&1 &
# Pauses the VESSL workspace when finished (or on error) if WORKSPACE_SLUG is set. Finished conditions are skipped on re-run.
#
#   surface   : (Q1) rl/sft/base x {none, using, walkto, moveto(collides with ALFWorld 'move X to Y'), fromtake, names}
#   skillgap  : (Q2) rl/sft x {5 envs} no skills; x {none, moveto, names} mismatched skills; + procedural skills on {none, names}
#   calib     : untrained reference (Qwen3-30B-A3B) x the same perturbations, with and without skills
#   necessity : ALFWorld, EMBEDDING retrieval: library variants (full / minus RETIRE_CATS / +duplicates) x top-k  -> slot competition
#   transfer  : procedure-sharing envs (scienceworld, twc, cookingworld) x base/sft/rl x {none, procedural, mismatch_proc, skillrl_raw}
#               add the competent untrained reference with TRANSFER_MODELS="base sft rl calib"
#               env difficulty: SW_SIMPL (e.g. "noElectricalAction,openDoors,teleportAction"),
#               TWC_PARAMS / CW_PARAMS (TextWorldExpress gameParams, e.g. "numLocations=1, numItemsToPutAway=1, includeDoors=0, limitInventorySize=0")
set -uo pipefail
cd "$(dirname "$0")/.." && source scripts/env.sh
PHASE=${1:-all}; REPS=${REPS:-2}; OUT=${OUT:-runs}
mkdir -p "$OUT/alfworld" "$OUT/necessity" "$OUT/transfer"
pause_ws () { if [ -n "${WORKSPACE_SLUG:-}" ]; then echo "[run_all] pausing workspace $WORKSPACE_SLUG"; vesslctl workspace pause "$WORKSPACE_SLUG"; fi; }
trap pause_ws EXIT
ts () { date '+%F %T'; }
NAMES="--names_map configs/names_map_example.json --perturb_skills"
QWEN3="--chat_template_kwargs {\"enable_thinking\":false} --gpu_mem 0.92"
model_of () { case $1 in rl) echo "$RL";; sft) echo "$SFT";; base) echo "$BASE";; calib) echo "$CALIB";; esac; }
extra_of () { [ "$1" = calib ] && echo "$QWEN3" || true; }

alf () {  # tag perturb cond(full|nosk|mismatch|proc|proc_mismatch) [extra...]
  local tag=$1 pert=$2 cond=$3; shift 3
  local out="$OUT/alfworld/${tag}_seen_${pert}_${cond}.jsonl"
  if [ -s "$out" ]; then echo "[skip] $out"; return; fi
  local a=(); case $cond in
    full) a=(--skills_json "$SKILLS" --skill_cond full);;
    nosk) a=(--skill_cond nosk);;
    mismatch) a=(--skills_json configs/skills_mismatched.json --skill_cond mismatch);;
    proc) a=(--skill_source procedural);;
    proc_mismatch) a=(--skill_source mismatch_proc);;
  esac
  local p=(); [ "$pert" = names ] && p=($NAMES)
  echo "[$(ts)] alfworld $tag $pert $cond"
  python -m alfworld_eval.run_alfworld_eval --model "$(model_of $tag)" --tag "$tag" --split seen --reps "$REPS" \
    "${a[@]}" --perturb "$pert" "${p[@]}" --max_games ${MAX_GAMES:-0} --seed ${SEED:-0} --save_think $(extra_of $tag) "$@" \
    --out "$out.tmp" && mv "$out.tmp" "$out"
}

if [ "$PHASE" = surface ] || [ "$PHASE" = all ]; then
  for tag in ${SURFACE_MODELS:-rl sft base}; do for p in ${SURFACE_PERTS:-none using walkto moveto fromtake names}; do alf $tag $p full; done; done
fi
if [ "$PHASE" = skillgap ] || [ "$PHASE" = all ]; then
  [ -f configs/skills_mismatched.json ] || python scripts/make_skill_variants.py --src "$SKILLS" --out configs
  for tag in ${SKILLGAP_MODELS:-rl sft}; do
    for p in ${NOSK_PERTS:-none using walkto fromtake names}; do alf $tag $p nosk; done
    for p in ${MISMATCH_PERTS:-none walkto names}; do alf $tag $p mismatch; done
    for p in ${PROC_PERTS:-none names}; do alf $tag $p proc; alf $tag $p proc_mismatch; done
  done
fi
if [ "$PHASE" = calib ] || [ "$PHASE" = all ]; then
  for p in ${CALIB_PERTS:-none using walkto moveto fromtake names}; do alf calib $p full; done
  for p in ${CALIB_NOSK_PERTS:-none walkto names}; do alf calib $p nosk; done
  for p in ${CALIB_MISMATCH_PERTS:-none walkto names}; do alf calib $p mismatch; done
  for p in ${CALIB_PROC_PERTS:-none names}; do alf calib $p proc; alf calib $p proc_mismatch; done
fi
if [ "$PHASE" = necessity ] || [ "$PHASE" = all ]; then
  # RETIRE_CATS: categories whose task-specific skills look internalized (small skill gap in Q2), e.g. "heat,cool"
  RETIRE_CATS=${RETIRE_CATS:-heat,cool}
  python necessity/make_library_variants.py --src "$SKILLS" --out configs --minus "$RETIRE_CATS" --plus_dups 1,2
  MINUS="configs/skills_minus_$(echo $RETIRE_CATS | tr , _).json"
  for tag in ${NECESSITY_MODELS:-rl}; do
    for k in ${NECESSITY_K:-3 6}; do
      for lib in "$SKILLS" "$MINUS" configs/skills_plus_dups1.json configs/skills_plus_dups2.json; do
        name=$(basename "$lib" .json); out="$OUT/necessity/${tag}_emb_k${k}_${name}.jsonl"
        if [ -s "$out" ]; then echo "[skip] $out"; continue; fi
        echo "[$(ts)] necessity $tag k=$k $name"
        python -m alfworld_eval.run_alfworld_eval --model "$(model_of $tag)" --tag "$tag" --split seen --reps "$REPS" \
          --skills_json "$lib" --retrieval_mode embedding --top_k $k --perturb none --max_games ${MAX_GAMES:-0} --seed ${SEED:-0} \
          --save_think $(extra_of $tag) --out "$out.tmp" && mv "$out.tmp" "$out"
      done
    done
  done
fi
if [ "$PHASE" = transfer ] || [ "$PHASE" = all ]; then
  for env in ${TRANSFER_ENVS:-scienceworld twc cookingworld}; do
    for tag in ${TRANSFER_MODELS:-base sft rl}; do
      for sk in ${TRANSFER_SKILLS:-none procedural mismatch_proc skillrl_raw}; do
        out="$OUT/transfer/${env}_${tag}_${sk}.jsonl"; [ -s "$out" ] && { echo "[skip] $out"; continue; }
        echo "[$(ts)] transfer $env $tag $sk"
        envopt=()
        case $env in
          scienceworld) envopt=(--simplifications "${SW_SIMPL:-noElectricalAction}");;
          twc)          [ -n "${TWC_PARAMS:-}" ] && envopt=(--game_params "$TWC_PARAMS");;
          cookingworld) [ -n "${CW_PARAMS:-}" ] && envopt=(--game_params "$CW_PARAMS");;
        esac
        python run_eval.py --env $env --model "$(model_of $tag)" --tag $tag --skills $sk --split test \
          --max_variations ${TRANSFER_VARS:-10} --reps "$REPS" --parallel ${PARALLEL:-8} --seed ${SEED:-0} --save_think $(extra_of $tag) \
          "${envopt[@]}" --out "$out.tmp" && mv "$out.tmp" "$out"
      done
    done
  done
fi
echo "[$(ts)] summary"
python -m alfworld_eval.metrics "$OUT/alfworld/*.jsonl" --margin ${MARGIN:-0.1} --names_map configs/names_map_example.json | tee "$OUT/summary_alfworld.txt" || true
python necessity/metrics_slots.py "$OUT/necessity/*.jsonl" --retired "${RETIRE_CATS:-heat,cool}" | tee "$OUT/summary_necessity.txt" || true
python metrics_transfer.py "$OUT/transfer/*.jsonl" | tee "$OUT/summary_transfer.txt" || true
