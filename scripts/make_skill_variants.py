#!/usr/bin/env python3
"""
Build skill-bank variants for the skill-dependence conditions.

  mismatched : task-specific skills are swapped across categories (a fixed derangement), general skills and
               common mistakes unchanged. Same format and roughly the same length as the real bank, but the
               task-specific content is wrong for the task. Compare against the real bank (full) and no skills.

Usage: python scripts/make_skill_variants.py --src $SKILLRL_ROOT/memory_data/alfworld/claude_style_skills.json --out configs
"""
import argparse, copy, json, os

DERANGEMENT = {"pick_and_place": "clean", "clean": "heat", "heat": "cool", "cool": "look_at_obj_in_light",
               "look_at_obj_in_light": "examine", "examine": "pick_and_place"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", default="configs")
    a = ap.parse_args()
    bank = json.load(open(a.src))
    ts = bank["task_specific_skills"]
    missing = [k for k in DERANGEMENT if k not in ts]
    assert not missing, f"categories missing in bank: {missing}"
    mism = copy.deepcopy(bank)
    mism["task_specific_skills"] = {cat: copy.deepcopy(ts[DERANGEMENT[cat]]) for cat in ts}
    mism.setdefault("metadata", {})["variant"] = "mismatched_task_specific:" + json.dumps(DERANGEMENT)
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "skills_mismatched.json")
    json.dump(mism, open(p, "w"), indent=2, ensure_ascii=False)
    # "general" = general_skills + the common_mistakes that template mode injects (first 5): both environment-agnostic
    nog = copy.deepcopy(bank); nog["general_skills"] = []; nog["common_mistakes"] = []
    nog.setdefault("metadata", {})["variant"] = "no_general_skills_and_no_injected_mistakes (task-specific only in prompt)"
    q = os.path.join(a.out, "skills_nogeneral.json")
    json.dump(nog, open(q, "w"), indent=2, ensure_ascii=False); print("wrote", q)
    # SkillRL's official split: remove only general_skills, keep common_mistakes (template mode still injects the first 5)
    nog6 = copy.deepcopy(bank); nog6["general_skills"] = []
    nog6.setdefault("metadata", {})["variant"] = "no_general_skills_only (mistakes kept)"
    q6 = os.path.join(a.out, "skills_nogeneral6.json")
    json.dump(nog6, open(q6, "w"), indent=2, ensure_ascii=False); print("wrote", q6)
    for cat in ts:
        print(f"{cat:22s} <- {DERANGEMENT[cat]:22s} ({len(ts[DERANGEMENT[cat]])} skills, ids {[s['skill_id'] for s in ts[DERANGEMENT[cat]]][:2]}...)")
    print("wrote", p)


if __name__ == "__main__":
    main()
