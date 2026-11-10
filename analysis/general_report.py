#!/usr/bin/env python3
"""
Report for scripts/run_general.sh.

(1a) ALFWorld, task-specific skills kept: effect of REMOVING general guidance = success(full) - success(nogen | nogen6)
(1b) ALFWorld, no task-specific skills (matched to other datasets): effect of ADDING general guidance
       = success(genonly | genonly6) - success(nosk)
(2)  Other datasets: effect of ADDING the same general guidance = (general | general6) - none
(3)  Verdict against the pre-registered Step 2 criteria (judged on the 11-item condition):
       (i)  ALFWorld (1b): effect <= +0.03 and CI upper <= +0.06      -> guidance not needed where it was learned
       (ii) other dataset: effect >= +0.10 and CI lower > 0           -> guidance needed again elsewhere
     11 items = SkillRL general skills (6) + injected common mistakes (5); 6 = the general skills only.
(4)  Matched comparison (pairing fixed before results): each other dataset next to the ALFWorld task types that share its
     procedure -- twc <-> pick, pick2 | scienceworld <-> heat, cool | cookingworld <-> pick, heat (weakest match).
     Extra log only: it does not change what is run or the verdict in (3).
(5)  ScienceWorld per task (boil / melt / freeze / change-the-state-of-matter-of). Extra log only.

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
        m = re.match(r"(\w+?)_(seen|unseen)_(full|nogen6|nogen|genonly6|genonly|nosk)_s(\d+)\.jsonl", os.path.basename(f))
        if not m: continue
        tag, split, bank, seed = m.groups()
        for l in open(f):
            r = json.loads(l); D[(tag, split, bank)][seed][r["game_id"]] = (float(r["success"]), r["task_type"])
    return D


def load_tr(out):
    D = defaultdict(lambda: defaultdict(dict))   # (env, tag, skills) -> seed -> var -> (success, score01)
    for f in glob.glob(os.path.join(out, "transfer", "*.jsonl")):
        m = re.match(r"(\w+?)_(\w+?)_(none|general6|general)_s(\d+)\.jsonl", os.path.basename(f))
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


def effect(hi, lo, idx=0):
    """paired effect hi - lo over common units (seed-averaged per unit), with bootstrap CI"""
    mh, ml = per_unit_mean(hi, idx), per_unit_mean(lo, idx); ks = sorted(set(mh) & set(ml))
    if not ks:                                   # nothing in common (e.g. a subset with no games): report as empty, no warnings
        nan = float("nan"); return nan, (nan, nan), 0, nan, nan, ks, mh, ml
    d = [mh[k] - ml[k] for k in ks]
    return float(np.mean(d)), boot(d), len(ks), float(np.mean([mh[k] for k in ks])), float(np.mean([ml[k] for k in ks])), ks, mh, ml


def by_type(f, ks, mh, ml):
    tt = {}
    for s in f.values():
        for k, v in s.items(): tt[k] = v[1]
    row = []
    for t in sorted(set(tt.values())):
        gg = [x for x in ks if tt.get(x) == t]
        if gg:
            dd = [mh[x] - ml[x] for x in gg]; c = boot(dd)
            row.append(f"{t}={np.mean(dd):+.2f}[{c[0]:+.2f},{c[1]:+.2f}]")
    return "  ".join(row)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "runs_gen"
    A, T = load_alf(out), load_tr(out)
    tags_splits = sorted({(k[0], k[1]) for k in A})

    print("=== (1a) ALFWorld, task-specific skills kept: effect of REMOVING general guidance: success(full) - success(cond) ===")
    print("    nogen = 11 items removed (6 general skills + 5 injected mistakes)   |   nogen6 = only the 6 general skills removed")
    for (tag, split) in tags_splits:
        for cond in ("nogen", "nogen6"):
            f, n = A.get((tag, split, "full")), A.get((tag, split, cond))
            if not f or not n: continue
            e, ci, k, mh_, ml_, ks, mh, ml = effect(f, n)
            print(f"  {tag:5s} {split:6s} {cond:6s} seeds={len(n)} n={k}  full {mh_:.3f}  {cond} {ml_:.3f}  effect {e:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]"
                  f"  | noise(flip rate same cond) full {noise(f):.2f} {cond} {noise(n):.2f}")
            print("        by type: " + by_type(f, ks, mh, ml))

    print("\n=== (1b) ALFWorld, NO task-specific skills (matched to other datasets): effect of ADDING general guidance: success(cond) - success(nosk) ===")
    print("    genonly = 11 items only   |   genonly6 = the 6 general skills only   |   nosk = no skills at all")
    alf_add = {}
    for (tag, split) in tags_splits:
        for cond in ("genonly", "genonly6"):
            g, n = A.get((tag, split, cond)), A.get((tag, split, "nosk"))
            if not g or not n: continue
            e, ci, k, mh_, ml_, ks, mh, ml = effect(g, n)
            if cond == "genonly": alf_add[(tag, split)] = (e, ci, min(len(g), len(n)))
            print(f"  {tag:5s} {split:6s} {cond:8s} seeds={len(g)} n={k}  nosk {ml_:.3f}  {cond} {mh_:.3f}  effect {e:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]"
                  f"  | noise nosk {noise(n):.2f} {cond} {noise(g):.2f}")
            print("        by type: " + by_type(g, ks, mh, ml))

    print("\n=== (2) Other datasets: effect of ADDING the same general guidance: success/score(cond) - (none) ===")
    print("    general = 11 items   |   general6 = the 6 general skills only")
    tr_add = {}
    for (env, tag) in sorted({(k[0], k[1]) for k in T}):
        for cond in ("general", "general6"):
            gS, nS = T.get((env, tag, cond)), T.get((env, tag, "none"))
            if not gS or not nS: continue
            for idx, name in ((0, "success"), (1, "score")):
                e, ci, k, mh_, ml_, *_ = effect(gS, nS, idx)
                if cond == "general":
                    tr_add.setdefault((env, tag), {})[name] = (e, ci, min(len(gS), len(nS)), mh_, ml_, k)
                print(f"  {env:12s} {tag:5s} {cond:8s} {name:7s} seeds={len(gS)} n={k}  none {ml_:.3f}  {cond} {mh_:.3f}  effect {e:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]"
                      + (f"  | noise none {noise(nS, 0):.2f}" if idx == 0 else ""))

    print("\n=== (3) Verdict against the pre-registered Step 2 criteria (11-item condition) ===")
    print("    (i) ALFWorld seen (1b): effect <= +0.03 and CI upper <= +0.06    (ii) other dataset: effect >= +0.10 and CI lower > 0")
    any_line = False
    for tag in sorted({k[1] for k in tr_add} | {k[0] for k in alf_add}):
        a = alf_add.get((tag, "seen"))
        if a:
            ok1 = a[0] <= 0.03 and a[1][1] <= 0.06
            print(f"  {tag:5s} (i)  ALFWorld seen: {a[0]:+.3f} [{a[1][0]:+.3f},{a[1][1]:+.3f}] seeds={a[2]}  -> {'MET' if ok1 else 'NOT met'}"); any_line = True
        else:
            print(f"  {tag:5s} (i)  ALFWorld seen: not measured in this folder"); any_line = True
        for (env, t2), d in sorted(tr_add.items()):
            if t2 != tag: continue
            name = "score" if env == "scienceworld" else "success"      # ScienceWorld is judged on its 0-1 progress score
            e, ci, ns, mh_, ml_, k = d[name]
            if mh_ < 0.10 and ml_ < 0.10: v = "FLOOR even with guidance (cannot be judged at this setting)"
            elif e >= 0.10 and ci[0] > 0: v = "MET"
            elif ci[0] > 0: v = "NOT met (weak effect: > 0 but < +0.10)"
            else: v = "NOT met"
            print(f"  {tag:5s} (ii) {env:12s} [{name}]: none {ml_:.3f} -> with guidance {mh_:.3f}, effect {e:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] n={k} seeds={ns}  -> {v}")
    if not any_line: print("  (nothing to judge in this folder)")
    print("  Note: with fewer than 3 seeds or few variations this is a preliminary read, not the final verdict.")

    # ---------------------------------------------------------------- (4) matched comparison (extra log)
    MATCH = {"twc": ["pick", "pick2"], "scienceworld": ["heat", "cool"], "cookingworld": ["pick", "heat"]}
    print("\n=== (4) Matched comparison (extra log; pairing fixed before results) ===")
    print("    each other dataset next to the ALFWorld-seen task types that share its procedure; effect of ADDING guidance (cond - no skills)")
    printed = False
    for tag in sorted({k[1] for k in T}):
        for env in ("twc", "scienceworld", "cookingworld"):
            name = "score" if env == "scienceworld" else "success"; idx = 1 if env == "scienceworld" else 0
            for alf_cond, tr_cond, label in (("genonly", "general", "11 items"), ("genonly6", "general6", "6 items ")):
                gS, nS = T.get((env, tag, tr_cond)), T.get((env, tag, "none"))
                gA, nA = A.get((tag, "seen", alf_cond)), A.get((tag, "seen", "nosk"))
                if not gS or not nS: continue
                e2, ci2, k2, *_ = effect(gS, nS, idx)
                right = f"{env} [{name}] {e2:+.3f} [{ci2[0]:+.3f},{ci2[1]:+.3f}] n={k2}"
                if gA and nA:
                    types = {}
                    for sd in list(gA.values()) + list(nA.values()):
                        for gid, v in sd.items(): types[gid] = v[1]
                    keep = {gid for gid, t in types.items() if t in MATCH[env]}
                    sub = lambda S: {sd: {gid: v for gid, v in d.items() if gid in keep} for sd, d in S.items()}
                    e1, ci1, k1, *_ = effect(sub(gA), sub(nA))
                    left = f"ALFWorld {'+'.join(MATCH[env])} {e1:+.3f} [{ci1[0]:+.3f},{ci1[1]:+.3f}] n={k1}" if k1 else f"ALFWorld {'+'.join(MATCH[env])} (no games)"
                else:
                    left = f"ALFWorld {'+'.join(MATCH[env])} (not measured in this folder)"
                print(f"  {tag:5s} {label}: {left}   vs   {right}"); printed = True
    if not printed: print("  (no other-dataset results in this folder)")

    # ---------------------------------------------------------------- (5) ScienceWorld per task (extra log)
    print("\n=== (5) ScienceWorld per task (extra log): progress score 0-1, cond - none ===")
    printed = False
    for tag in sorted({k[1] for k in T if k[0] == "scienceworld"}):
        nS = T.get(("scienceworld", tag, "none"))
        if not nS: continue
        tasks = sorted({k[0] for d in nS.values() for k in d})
        for task in tasks:
            pick = lambda S: {sd: {k: v for k, v in d.items() if k[0] == task} for sd, d in S.items()}
            base = per_unit_mean(pick(nS), 1); row = f"  {tag:5s} {task:32s} n={len(base)}  none {np.mean(list(base.values())):.3f}"
            for cond in ("general6", "general"):
                gS = T.get(("scienceworld", tag, cond))
                if not gS: continue
                e, ci, k, mh_, ml_, *_ = effect(pick(gS), pick(nS), 1)
                row += f"  | {cond} {mh_:.3f} ({e:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}])"
            print(row); printed = True
    if not printed: print("  (no ScienceWorld results in this folder)")


if __name__ == "__main__":
    main()
