#!/usr/bin/env python3
"""
Transfer metrics for the procedure-sharing environments (run_eval.py logs).
Per env: success / score (+CI over variations) for each tag x skill source; the model ladder (base -> sft -> rl);
the skill gap (source - none) per model; ALFWorld-wording intrusion; and the key contrast
   [gap(rl) - gap(base)]: does the trained model benefit MORE (or less) from procedural skills than the untrained base?
Usage: python metrics_transfer.py "runs/transfer/*.jsonl"
"""
import argparse, glob, json
from collections import defaultdict
import numpy as np


def boot_ci(v, n=2000, seed=0):
    v = np.asarray(v, float); rng = np.random.default_rng(seed)
    if len(v) == 0: return (float("nan"), float("nan"))
    m = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def per_var(recs, key="success"):
    acc = defaultdict(list)
    for r in recs:
        v = float(r[key]) if key == "success" else max(float(r["score"]), 0.0) / (100.0 if r["env"] == "scienceworld" else 1.0)
        acc[(r["task_key"], r["variation"])].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("paths", nargs="+"); a = ap.parse_args()
    recs = []
    for p in a.paths:
        for pp in glob.glob(p): recs += [json.loads(l) for l in open(pp) if l.strip()]
    envs = sorted({r["env"] for r in recs})
    for env in envs:
        R = [r for r in recs if r["env"] == env]
        g = defaultdict(list)
        for r in R: g[(r["tag"], r["skills"])].append(r)
        print(f"\n=== {env} ===  (score = clipped partial score scaled to 0-1; success = task completed)")
        print(f"{'tag':6s} {'skills':14s} {'n':>4s} {'success':>8s} {'95%CI':>16s} {'score':>7s} {'alf_intr':>8s} {'inv_fmt':>8s}")
        for key in sorted(g):
            rs = g[key]; sv = per_var(rs); sc = per_var(rs, "score"); ci = boot_ci(list(sv.values()))
            tot = intr = bad = 0
            for r in rs:
                for s in r["steps"]:
                    tot += 1; intr += bool(s.get("alf_intrusion")); bad += (not s["valid_format"])
            print(f"{key[0]:6s} {key[1]:14s} {len(sv):4d} {np.mean(list(sv.values())):8.3f} [{ci[0]:.3f},{ci[1]:.3f}] "
                  f"{np.mean(list(sc.values())):7.3f} {intr/max(1,tot):8.3f} {bad/max(1,tot):8.3f}")
        print("--- skill gap (source - none), paired by variation ---")
        for tag in ("base", "sft", "rl", "calib"):
            base = per_var(g[(tag, "none")]) if (tag, "none") in g else None
            if not base: continue
            for src in ("procedural", "mismatch_proc", "skillrl_raw"):
                if (tag, src) not in g: continue
                o = per_var(g[(tag, src)]); common = sorted(set(base) & set(o))
                d = [o[k] - base[k] for k in common]; ci = boot_ci(d)
                v = "HELPS" if ci[0] > 0 else ("HURTS" if ci[1] < 0 else "no effect")
                print(f"    {tag:5s} {src:14s} n={len(common):3d} {np.mean(d):+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  {v}")
        for ref in ("base", "calib"):
          for src, ctrl in (("procedural", "none"), ("procedural", "mismatch_proc")):
            need = [("rl", ctrl), ("rl", src), (ref, ctrl), (ref, src)]
            if not all(k in g for k in need):
                continue
            P = {k: per_var(g[k]) for k in need}; common = sorted(set.intersection(*[set(v) for v in P.values()]))
            d = [(P[("rl", src)][k] - P[("rl", ctrl)][k]) - (P[(ref, src)][k] - P[(ref, ctrl)][k]) for k in common]
            ci = boot_ci(d)
            v = "TRAINED LEANS MORE ON PROCEDURAL SKILLS" if ci[0] > 0 else ("trained leans less" if ci[1] < 0 else "no difference")
            print(f"--- key: [{src}-{ctrl}](rl) - [{src}-{ctrl}]({ref})  n={len(common):3d} {np.mean(d):+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  {v}")
        need = [("rl", "none"), ("rl", "procedural"), ("base", "none"), ("base", "procedural")]
        if False and all(k in g for k in need):
            P = {k: per_var(g[k]) for k in need}; common = sorted(set.intersection(*[set(v) for v in P.values()]))
            d = [(P[("rl", "procedural")][k] - P[("rl", "none")][k]) - (P[("base", "procedural")][k] - P[("base", "none")][k]) for k in common]
            ci = boot_ci(d)
            v = "TRAINED LEANS MORE ON PROCEDURAL SKILLS" if ci[0] > 0 else ("trained leans less" if ci[1] < 0 else "no difference")
            print(f"    n={len(common):3d} {np.mean(d):+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  {v}")


if __name__ == "__main__":
    main()
