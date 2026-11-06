#!/usr/bin/env python3
"""
Library variants for the NECESSITY experiment (embedding retrieval, top-k across all categories):
  minus:<cats>  remove the task-specific skills of the given categories (e.g. heat,cool) = retire them
  plus_dups:N   add N near-duplicate copies of every task-specific skill (same meaning, trivially reworded) = growth by redundancy
Usage:
  python necessity/make_library_variants.py --src $SKILLS --out configs --minus heat,cool --plus_dups 1,2
"""
import argparse, copy, json, os


def minus(bank, cats):
    b = copy.deepcopy(bank)
    for c in cats:
        assert c in b["task_specific_skills"], f"unknown category {c}; have {list(b['task_specific_skills'])}"
        b["task_specific_skills"][c] = []
    b.setdefault("metadata", {})["variant"] = f"minus:{','.join(cats)}"
    return b


REWORD = ["In practice, ", "As a rule, ", "Remember: ", "Generally, "]


def plus_dups(bank, n):
    b = copy.deepcopy(bank)
    for cat, skills in b["task_specific_skills"].items():
        extra = []
        for s in skills:
            for i in range(n):
                d = copy.deepcopy(s)
                d["skill_id"] = f"dup{i+1}_{s['skill_id']}"; d["title"] = f"{s['title']} (variant {i+1})"
                d["principle"] = REWORD[i % len(REWORD)] + s["principle"][0].lower() + s["principle"][1:]
                extra.append(d)
        skills.extend(extra)
    b.setdefault("metadata", {})["variant"] = f"plus_dups:{n}"
    return b


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", default="configs")
    ap.add_argument("--minus", default=""); ap.add_argument("--plus_dups", default="")
    a = ap.parse_args(); bank = json.load(open(a.src)); os.makedirs(a.out, exist_ok=True)
    if a.minus:
        cats = a.minus.split(","); p = os.path.join(a.out, f"skills_minus_{'_'.join(cats)}.json")
        json.dump(minus(bank, cats), open(p, "w"), indent=2, ensure_ascii=False); print("wrote", p)
    for n in [int(x) for x in a.plus_dups.split(",") if x]:
        p = os.path.join(a.out, f"skills_plus_dups{n}.json")
        json.dump(plus_dups(bank, n), open(p, "w"), indent=2, ensure_ascii=False); print("wrote", p)


if __name__ == "__main__":
    main()
