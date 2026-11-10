#!/usr/bin/env python3
"""
Report for scripts/run_general.sh — only the two questions.

(1) Training dataset (ALFWorld): effect of removing the general skills = success(full) - success(nogen),
    per model and split, with the run-to-run noise measured from the SAME condition across seeds.
(2) Other datasets: effect of adding the general skills = score/success(general) - (none), per env and model.
Plus the contrast that defines "generalization failure":
    [effect in other dataset] - [effect in ALFWorld]  (>0: general skills matter elsewhere but not where they were learned)

Each seed is one full pass over the games/variations; CIs are bootstrap over games, pooled across seeds.
Noise = spread of the same condition across seeds (games whose outcome flips between two seeds of the same condition).
"""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np


def boot(v, n=4000, seed=0):
    v = np.asarray(v, float)
    if len(v) == 0: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load_alf(out):
    D = defaultdict(lambda: defaultdict(dict))   # (tag, split, bank) -> seed -> game -> success
    for f in glob.glob(os.path.join(out, "alfworld", "*.jsonl")):
        m = re.match(r"(\w+?)_(seen|unseen)_(full|nogen6|nogen)_s(\d+)\.jsonl", os.path.basename(f))
        if not m: continue
        tag, split, bank, seed = m.groups()
        for l in open(f):
            r = json.loads(l); D[(tag, split, bank)][seed][r["game_id"]] = (float(r["success"]), r["task_type"])
    return D


def load_tr(out):
    D = defaultdict(lambda: defaultdict(dict))   # (env, tag, skills) -> seed -> var -> (success, score01)
    for f in glob.glob(os.path.join(out, "transfer", "*.jsonl")):
        m = re.match(r"(\w+?)_(\w+?)_(none|general)_s(\d+)\.jsonl", os.path.basename(f))
        if not m: continue
        env, tag, sk, seed = m.groups()
        for l in open(f):
            r = json.loads(l)
            sc = max(float(r["score"]), 0.0) / (100.0 if env == "scienceworld" else 1.0)
            D[(env, tag, sk)][seed][(r["task_key"], r["variation"])] = (float(r["success"]), sc)
    return D


def per_unit_mean(seeds_dict, idx=0):
    acc = defaultdict(list)
    for s, d in seeds_dict.items():
        for k, v in d.items(): acc[k].append(v[idx])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def noise(seeds_dict, idx=0):
    """fraction of units whose outcome differs between two seeds of the same condition (averaged over seed pairs)"""
    ss = sorted(seeds_dict); fr = []
    for i in range(len(ss)):
        for j in range(i + 1, len(ss)):
            a, b = seeds_dict[ss[i]], seeds_dict[ss[j]]; ks = set(a) & set(b)
            if ks: fr.append(np.mean([a[k][idx] != b[k][idx] for k in ks]))
    return float(np.mean(fr)) if fr else float("nan")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "runs_gen"
    A, T = load_alf(out), load_tr(out)
    print("=== (1) ALFWorld: effect of REMOVING general guidance: success(full) - success(cond) ===")
    print("    nogen  = general skills (6) + injected common mistakes (5) removed   |   nogen6 = only the 6 general skills removed (SkillRL's split)")
    alf_effect = {}
    for (tag, split, cond) in sorted({(k[0], k[1], c) for k in A for c in ("nogen", "nogen6")}):
        f, n = A.get((tag, split, "full")), A.get((tag, split, cond))
        if not f or not n: continue
        mf, mn = per_unit_mean(f), per_unit_mean(n); g = sorted(set(mf) & set(mn))
        d = [mf[x] - mn[x] for x in g]; ci = boot(d)
        if cond == "nogen": alf_effect[(tag, split)] = (np.mean(d), ci)
        print(f"  {tag:5s} {split:6s} {cond:6s} seeds={len(n)} n={len(g)}  full {np.mean([mf[x] for x in g]):.3f}  {cond} {np.mean([mn[x] for x in g]):.3f}"
              f"  effect {np.mean(d):+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  | noise(flip rate same cond) full {noise(f):.2f} {cond} {noise(n):.2f}")
        tt = {}
        for s in f.values():
            for k, v in s.items(): tt[k] = v[1]
        row = []
        for t in sorted(set(tt.values())):
            gg = [x for x in g if tt.get(x) == t]
            if gg:
                dd = [mf[x] - mn[x] for x in gg]; c = boot(dd)
                row.append(f"{t}={np.mean(dd):+.2f}[{c[0]:+.2f},{c[1]:+.2f}]")
        print("        by type: " + "  ".join(row))
    print("\n=== (2) Other datasets: effect of ADDING the same general skills: general - none ===")
    tr_effect = {}
    for (env, tag) in sorted({(k[0], k[1]) for k in T}):
        gS, nS = T.get((env, tag, "general")), T.get((env, tag, "none"))
        if not gS or not nS: continue
        for idx, name in ((0, "success"), (1, "score")):
            mg, mn = per_unit_mean(gS, idx), per_unit_mean(nS, idx); ks = sorted(set(mg) & set(mn))
            d = [mg[k] - mn[k] for k in ks]; ci = boot(d)
            if idx == 0: tr_effect[(env, tag)] = (np.mean(d), ci)
            print(f"  {env:12s} {tag:5s} {name:7s} seeds={len(gS)} n={len(ks)}  none {np.mean([mn[k] for k in ks]):.3f}  general {np.mean([mg[k] for k in ks]):.3f}"
                  f"  effect {np.mean(d):+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]" + (f"  | noise none {noise(nS,0):.2f}" if idx == 0 else ""))
    print("\n=== generalization-failure pattern (per model): effect in other dataset vs effect in ALFWorld seen ===")
    for tag in sorted({k[0] for k in alf_effect}):
        if (tag, "seen") not in alf_effect: continue
        a = alf_effect[(tag, "seen")]
        for (env, t2), e in sorted(tr_effect.items()):
            if t2 != tag: continue
            print(f"  {tag:5s} ALFWorld-seen {a[0]:+.3f} [{a[1][0]:+.3f},{a[1][1]:+.3f}]  vs  {env:12s} {e[0]:+.3f} [{e[1][0]:+.3f},{e[1][1]:+.3f}]")
    print("  (pattern of interest: ALFWorld effect ~0 with a CI inside the noise, other-dataset effect clearly > 0, and larger for rl than calib)")


if __name__ == "__main__":
    main()
