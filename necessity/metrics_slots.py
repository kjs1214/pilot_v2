#!/usr/bin/env python3
"""
Necessity metrics (ALFWorld, embedding retrieval): are retrieval slots taken by skills the policy no longer needs?

Per (tag, library variant, k): success (+CI), mean prompt tokens, and slot composition of the retrieved
task-specific skills: share from the true task category, share from the 'retired' categories (--retired), share of dup_ copies.
Usage: python necessity/metrics_slots.py "runs/necessity/*.jsonl" --retired heat,cool
"""
import argparse, glob, json, re
from collections import defaultdict
import numpy as np

CAT_OF = {"pic": "pick_and_place", "loo": "look_at_obj_in_light", "cle": "clean", "hea": "heat", "coo": "cool", "exa": "examine"}
TRUE_CAT = {"pick": "pick_and_place", "pick2": "pick_and_place", "look": "look_at_obj_in_light", "clean": "clean", "heat": "heat", "cool": "cool"}


def boot_ci(v, n=2000, seed=0):
    v = np.asarray(v, float); rng = np.random.default_rng(seed)
    if len(v) == 0: return (float("nan"), float("nan"))
    m = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cat_of(sid):
    sid = sid.replace("dup1_", "").replace("dup2_", "").replace("dup3_", "")
    return CAT_OF.get(sid[:3])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("paths", nargs="+"); ap.add_argument("--retired", default="")
    a = ap.parse_args(); retired = set(a.retired.split(",")) if a.retired else set()
    recs = []
    for p in a.paths:
        for pp in glob.glob(p): recs += [json.loads(l) for l in open(pp) if l.strip()]
    g = defaultdict(list)
    for r in recs:
        lib = r.get("library", "full"); g[(r["tag"], lib, r.get("top_k"), r.get("retrieval_mode"))].append(r)
    print(f"{'tag':5s} {'library':22s} {'k':>3s} {'mode':9s} {'n':>4s} {'success':>8s} {'95%CI':>16s} {'tokens':>7s} {'true_cat':>8s} {'retired':>8s} {'dups':>6s} {'empty':>6s}")
    for key in sorted(g, key=str):
        rs = g[key]; by_game = defaultdict(list)
        for r in rs: by_game[r["game_id"]].append(float(r["success"]))
        vals = [np.mean(v) for v in by_game.values()]; ci = boot_ci(vals)
        toks = [r["prompt_tokens_mean"] for r in rs if r.get("prompt_tokens_mean")]
        tc = rc = dc = tot = empty = 0
        for r in rs:
            ts = [s for s in r.get("skill_ids", []) if not s.startswith(("gen_", "err_", "proc_"))]
            if not ts: empty += 1
            for s in ts:
                tot += 1; c = cat_of(s)
                tc += (c == TRUE_CAT.get(r["task_type"])); rc += (c in retired); dc += s.startswith("dup")
        d = lambda x, y: x / y if y else float("nan")
        print(f"{key[0]:5s} {str(key[1]):22s} {str(key[2]):>3s} {str(key[3]):9s} {len(by_game):4d} {np.mean(vals):8.3f} [{ci[0]:.3f},{ci[1]:.3f}] "
              f"{(np.mean(toks) if toks else float('nan')):7.0f} {d(tc,tot):8.3f} {d(rc,tot):8.3f} {d(dc,tot):6.3f} {empty/len(rs):6.3f}")
    print("\ntrue_cat = share of retrieved task-specific slots from the task's own category; retired = share from --retired categories;")
    print("dups = share taken by near-duplicate copies (plus_dups libraries); empty = episodes with no task-specific skill retrieved.")


if __name__ == "__main__":
    main()
