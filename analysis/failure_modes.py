#!/usr/bin/env python3
"""
Why did episodes fail? Reads other-dataset logs (run_eval.py output) and prints, per (env, model, skills):
  - how episodes ended: success / ended early by the environment / ran out of steps
  - share of actions that were in the action list shown to the model
  - share of steps repeating the previous action (loops), most frequent actions

Usage: python analysis/failure_modes.py "runs_x/transfer/*.jsonl" [--max_steps 100]
"""
import argparse, glob, json, statistics as st
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pattern"); ap.add_argument("--max_steps", type=int, default=0)
    a = ap.parse_args()
    G = defaultdict(list)
    for f in sorted(glob.glob(a.pattern)):
        for l in open(f):
            r = json.loads(l); G[(r["env"], r["tag"], r["skills"])].append(r)
    if not G:
        print("no records for", a.pattern); return
    for (env, tag, sk), R in sorted(G.items()):
        limit = a.max_steps or max(r["n_steps"] for r in R)
        succ = sum(1 for r in R if r["success"])
        hit = sum(1 for r in R if not r["success"] and r["n_steps"] >= limit)
        early = len(R) - succ - hit
        steps = [s for r in R for s in r["steps"]]
        inlist = sum(1 for s in steps if s.get("in_admissible_shown")) / max(1, len(steps))
        rep = tot = 0
        for r in R:
            acts = [s.get("raw_action") for s in r["steps"]]
            for i in range(1, len(acts)):
                tot += 1; rep += int(acts[i] is not None and acts[i] == acts[i - 1])
        c = Counter((s.get("raw_action") or "<no action parsed>") for s in steps)
        top = ", ".join(f"{k} {v / len(steps):.0%}" for k, v in c.most_common(3))
        distinct = st.median([len({s.get('raw_action') for s in r['steps']}) for r in R])
        print(f"== {env} | {tag} | {sk}   n={len(R)}  success {succ / len(R):.3f}")
        print(f"   ended: success {succ}, ended early by env {early}, ran out of steps ({limit}) {hit}   | mean steps {st.mean(r['n_steps'] for r in R):.0f}")
        print(f"   actions that were in the shown list {inlist:.0%}   | same action as the previous step {rep / max(1, tot):.0%}   | distinct actions per episode (median) {distinct:.0f}")
        print(f"   most frequent actions: {top}")


if __name__ == "__main__":
    main()
