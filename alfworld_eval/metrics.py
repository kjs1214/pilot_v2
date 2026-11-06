#!/usr/bin/env python3
"""
Aggregate ALFWorld run logs (JSONL from run_alfworld_eval.py).

Reports, per (tag, split, perturb, skills):
  success rate (game-level mean over reps) with bootstrap CI, by task type
  invalid-format rate, off-list rate (raw action not in the admissible list shown)
  intrusion rate: off-list actions that match the ORIGINAL surface (training wording)
  think-intrusion rate: <think> mentions original wording while the env shows the perturbed one
and the ladder: for each tag, perturbed - original (paired by game_id) with CI, plus an equivalence check.

Usage: python -m alfworld_eval.metrics runs/*.jsonl --margin 0.05 [--names_map configs/names_map.json]
"""
from __future__ import annotations
import argparse, json, glob, re, sys, os
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alfworld_eval.perturb import ORIGINAL_FORM

THINK_PATTERNS = {"using": re.compile(r"\b(heat|cool|clean)\b[^.\n]{0,30}\bwith\b"),
                  "moveto": re.compile(r"\bgo to\b"),
                  "fromtake": re.compile(r"\btake\b[^.\n]{0,30}\bfrom\b")}


def load(paths):
    recs = []
    for p in paths:
        for pp in glob.glob(p):
            with open(pp) as f:
                for line in f:
                    if line.strip():
                        recs.append(json.loads(line))
    return recs


def boot_ci(vals, n=2000, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    m = vals[idx].mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def game_level(recs):
    """mean success per game (over reps) -> dict game_id -> value"""
    acc = defaultdict(list)
    for r in recs:
        acc[r["game_id"]].append(float(r["success"]))
    return {g: float(np.mean(v)) for g, v in acc.items()}


def step_rates(recs, perturb, names):
    tot = fmt_bad = off = intr = think_intr = think_tot = 0
    orig_re = ORIGINAL_FORM.get(perturb)
    if perturb == "names" and names:
        orig_re = re.compile(r"\b(" + "|".join(re.escape(k) for k in names) + r")\b")
    tre = THINK_PATTERNS.get(perturb)
    if perturb == "names" and names:
        tre = orig_re
    for r in recs:
        for s in r["steps"]:
            tot += 1
            if not s["valid_format"]:
                fmt_bad += 1
            ra = s["raw_action"]
            if ra is not None and not s["in_admissible_shown"]:
                off += 1
                if orig_re is not None and orig_re.search(ra):
                    intr += 1
            if s.get("think") is not None and tre is not None:
                think_tot += 1
                if tre.search(s["think"]):
                    think_intr += 1
    d = lambda a, b: (a / b) if b else float("nan")
    return {"steps": tot, "invalid_format": d(fmt_bad, tot), "off_list": d(off, tot),
            "intrusion": d(intr, tot), "think_intrusion": d(think_intr, think_tot)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+"); ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--names_map", default=None)
    args = ap.parse_args()
    names = json.load(open(args.names_map)) if args.names_map else None
    recs = load(args.paths)
    groups = defaultdict(list)
    for r in recs:
        r.setdefault("skill_cond", "full" if r.get("skills") else "nosk")
        groups[(r["tag"], r["split"], r["perturb"], r["skill_cond"])].append(r)

    print("\n=== per condition ===")
    print(f"{'tag':6s} {'split':7s} {'perturb':9s} {'skills':8s} {'n_games':7s} {'success':>8s} {'95%CI':>16s} {'inv_fmt':>8s} {'off_list':>8s} {'intrus':>8s} {'think_in':>8s}")
    for key in sorted(groups):
        rs = groups[key]; gl = game_level(rs); vals = list(gl.values())
        ci = boot_ci(vals); sr = step_rates(rs, key[2], names)
        print(f"{key[0]:6s} {key[1]:7s} {key[2]:9s} {str(key[3]):8s} {len(gl):7d} {np.mean(vals):8.3f} [{ci[0]:.3f},{ci[1]:.3f}] "
              f"{sr['invalid_format']:8.3f} {sr['off_list']:8.3f} {sr['intrusion']:8.3f} {sr['think_intrusion']:8.3f}")
        by_type = defaultdict(list)
        for r in rs:
            by_type[r["task_type"]].append(float(r["success"]))
        print("      by type: " + "  ".join(f"{t}={np.mean(v):.2f}(n={len(v)})" for t, v in sorted(by_type.items())))
        routed = [(r["task_type"], r.get("routed_type")) for r in rs if r.get("routed_type")]
        if routed:
            expect = {"pick": "pick_and_place", "pick2": "pick_and_place", "look": "look_at_obj_in_light",
                      "clean": "clean", "heat": "heat", "cool": "cool"}
            mis = [(a, b) for a, b in routed if expect.get(a) != b and not (a == "look" and b == "examine")]
            if mis:
                from collections import Counter
                print("      skill routing (SkillRL template mode) differs from true type: "
                      + ", ".join(f"{a}->{b} x{c}" for (a, b), c in Counter(mis).items()))

    print("\n=== ladder: perturbed - original (paired by game) ===")
    for (tag, split, skills) in sorted({(k[0], k[1], k[3]) for k in groups}):
        base_key = (tag, split, "none", skills)
        if base_key not in groups:
            continue
        g0 = game_level(groups[base_key])
        for key in sorted(k for k in groups if k[0] == tag and k[1] == split and k[3] == skills and k[2] != "none"):
            g1 = game_level(groups[key]); common = sorted(set(g0) & set(g1))
            if not common:
                continue
            diffs = [g1[g] - g0[g] for g in common]
            ci = boot_ci(diffs)
            verdict = ("EQUIVALENT (within ±%.2f)" % args.margin) if (ci[0] > -args.margin and ci[1] < args.margin) \
                else ("DROP" if ci[1] < 0 else ("GAIN" if ci[0] > 0 else "inconclusive"))
            print(f"{tag:6s} {split:7s} skills={skills!s:8s} {key[2]:9s} n={len(common):3d} delta={np.mean(diffs):+.3f} "
                  f"CI=[{ci[0]:+.3f},{ci[1]:+.3f}]  {verdict}")
    skill_dependence(groups, args.margin)
    key_contrasts(groups, args.margin, names)


def skill_dependence(groups, margin):
    """For each (tag, split, perturb): success(full) - success(cond) for cond in nosk/mismatch, paired by game,
    and the change of that gap relative to the original environment (difference in differences)."""
    print("\n=== skill dependence: success(full) - success(cond), paired by game ===")
    keys = {(k[0], k[1]) for k in groups}
    for tag, split in sorted(keys):
        for cond in ("nosk", "mismatch"):
            base = None
            for pert in sorted({k[2] for k in groups if k[0] == tag and k[1] == split}, key=lambda x: (x != "none", x)):
                kf, kc = (tag, split, pert, "full"), (tag, split, pert, cond)
                if kf not in groups or kc not in groups:
                    continue
                gf, gc = game_level(groups[kf]), game_level(groups[kc])
                common = sorted(set(gf) & set(gc))
                if not common:
                    continue
                gaps = {g: gf[g] - gc[g] for g in common}
                vals = list(gaps.values()); ci = boot_ci(vals)
                line = f"{tag:6s} {split:7s} {pert:9s} full-{cond:8s} n={len(common):3d} gap={np.mean(vals):+.3f} CI=[{ci[0]:+.3f},{ci[1]:+.3f}]"
                if pert == "none":
                    base = gaps
                elif base is not None:
                    cg = sorted(set(base) & set(gaps))
                    if cg:
                        dd = [gaps[g] - base[g] for g in cg]; cdd = boot_ci(dd)
                        tagstr = "RELIES MORE ON SKILLS UNDER SHIFT" if cdd[0] > 0 else ("relies less" if cdd[1] < 0 else "no change")
                        line += f"  | vs none: {np.mean(dd):+.3f} [{cdd[0]:+.3f},{cdd[1]:+.3f}] {tagstr}"
                print(line)


def game_rate(recs, perturb, names):
    """per-game mean intrusion rate (fraction of steps whose off-list action matches the ORIGINAL wording)."""
    orig_re = ORIGINAL_FORM.get(perturb)
    if perturb == "names" and names:
        orig_re = re.compile(r"\b(" + "|".join(re.escape(k) for k in names) + r")\b")
    acc = defaultdict(list)
    for r in recs:
        tot = intr = 0
        for s in r["steps"]:
            tot += 1
            ra = s["raw_action"]
            if orig_re is not None and ra is not None and not s["in_admissible_shown"] and orig_re.search(ra):
                intr += 1
        acc[r["game_id"]].append(intr / tot if tot else 0.0)
    return {g: float(np.mean(v)) for g, v in acc.items()}


def key_contrasts(groups, margin, names):
    """The comparisons the pilot is designed for (all paired by game, bootstrap 95% CI over games).
    Main question: an ALFWorld-UNTRAINED model is unaffected by the surface change, the TRAINED model breaks."""
    def gl(tag, pert, cond="full", split="seen"):
        k = (tag, split, pert, cond)
        return game_level(groups[k]) if k in groups else None
    def did(tr, ref, p):
        t0, t1, r0, r1 = gl(tr, "none"), gl(tr, p), gl(ref, "none"), gl(ref, p)
        if not (t0 and t1 and r0 and r1):
            return None
        g = sorted(set(t0) & set(t1) & set(r0) & set(r1))
        return g, [(t1[x] - t0[x]) - (r1[x] - r0[x]) for x in g]
    perts = sorted({k[2] for k in groups if k[2] != "none"})
    print("\n=== KEY CONTRASTS (paired by game; 95% bootstrap CI) ===")
    print("--- Q1: untrained model unaffected by the surface change, trained model breaks / pulls training wording? ---")
    print("[A] untrained reference is unaffected: calib(pert) - calib(none), within ±margin = FAIR")
    for p in perts:
        a_, b_ = gl("calib", "none"), gl("calib", p)
        if a_ and b_:
            g = sorted(set(a_) & set(b_)); d = [b_[x] - a_[x] for x in g]; ci = boot_ci(d)
            ok = "FAIR" if (ci[0] > -margin and ci[1] < margin) else ("HARDER" if ci[1] < 0 else "inconclusive")
            print(f"    {p:9s} n={len(g):3d} {np.mean(d):+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  {ok}")
    print("[B] trained model breaks more than the untrained reference: (trained pert-none) - (calib pert-none), <0 = breaks")
    for tr in ("rl", "sft"):
        for p in perts:
            r = did(tr, "calib", p)
            if r:
                g, d = r; ci = boot_ci(d)
                v = "TRAINED BREAKS" if ci[1] < 0 else ("trained gains" if ci[0] > 0 else "no difference")
                print(f"    {tr:4s} {p:9s} n={len(g):3d} {np.mean(d):+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  {v}")
    print("[C] pulls training wording under perturbation: intrusion(trained) - intrusion(untrained), >0 = pulls old wording")
    for tr in ("rl", "sft"):
        for ref in ("base", "calib"):
            for p in perts:
                kt, kr = (tr, "seen", p, "full"), (ref, "seen", p, "full")
                if kt in groups and kr in groups:
                    it, ir = game_rate(groups[kt], p, names), game_rate(groups[kr], p, names)
                    g = sorted(set(it) & set(ir)); d = [it[x] - ir[x] for x in g]; ci = boot_ci(d)
                    v = "TRAINED INTRUDES" if ci[0] > 0 else ("reference intrudes more" if ci[1] < 0 else "no difference")
                    print(f"    {tr:4s} vs {ref:5s} {p:9s} n={len(g):3d} {np.mean(d):+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]  {v}")
    print("\n--- Q2 (main): small with/without-skill gap in the original env, but a large gap after the surface change? ---")
    print("[D] skill gap = success(full skills) - success(cond); cond = nosk (no skills) or mismatch (wrong task skills, same format)")
    print("    columns: gap in original env | gap in perturbed env | change (pert - none). change > 0 = skills matter more after the change")
    def gap_rows(tr, cond, p):
        need = [(tr, q, c) for q in ("none", p) for c in ("full", cond)]
        G = {n: gl(n[0], n[1], n[2]) for n in need}
        if any(v is None for v in G.values()):
            return None
        g = sorted(set.intersection(*[set(v) for v in G.values()]))
        g0 = [G[(tr, "none", "full")][x] - G[(tr, "none", cond)][x] for x in g]
        g1 = [G[(tr, p, "full")][x] - G[(tr, p, cond)][x] for x in g]
        return g, g0, g1
    dd_cache = {}
    for tr in ("rl", "sft", "calib"):
        for cond in ("nosk", "mismatch", "proc", "proc_mismatch"):
            for p in perts:
                r = gap_rows(tr, cond, p)
                if r is None:
                    continue
                g, g0, g1 = r; ch = [b - a for a, b in zip(g0, g1)]
                c0, c1, cc = boot_ci(g0), boot_ci(g1), boot_ci(ch)
                dd_cache[(tr, cond, p)] = dict(zip(g, ch))
                v = "SKILLS MATTER MORE AFTER CHANGE" if cc[0] > 0 else ("matter less" if cc[1] < 0 else "no change")
                print(f"    {tr:5s} {cond:8s} {p:9s} n={len(g):3d}  none {np.mean(g0):+.3f} [{c0[0]:+.3f},{c0[1]:+.3f}]"
                      f" | pert {np.mean(g1):+.3f} [{c1[0]:+.3f},{c1[1]:+.3f}] | change {np.mean(ch):+.3f} [{cc[0]:+.3f},{cc[1]:+.3f}]  {v}")
    print("[D'] control: is the change specific to TRAINED models? change(trained) - change(calib), > 0 = training-specific")
    for tr in ("rl", "sft"):
        for cond in ("nosk", "mismatch"):
            for p in perts:
                a_, b_ = dd_cache.get((tr, cond, p)), dd_cache.get(("calib", cond, p))
                if not a_ or not b_:
                    continue
                g = sorted(set(a_) & set(b_)); d = [a_[x] - b_[x] for x in g]; ci = boot_ci(d)
                v = "TRAINING-SPECIFIC" if ci[0] > 0 else ("calib larger" if ci[1] < 0 else "no difference")
                print(f"    {tr:4s} {cond:8s} {p:9s} n={len(g):3d} {np.mean(d):+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  {v}")
    print("\n--- secondary ---")
    print("[E] (secondary) where it arises: (rl pert-none) - (sft pert-none), <0 = RL stage adds to the break")
    for p in perts:
        r = did("rl", "sft", p)
        if r:
            g, d = r; ci = boot_ci(d)
            v = "RL ADDS" if ci[1] < 0 else ("SFT stage larger" if ci[0] > 0 else "no difference")
            print(f"    {p:9s} n={len(g):3d} {np.mean(d):+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  {v}")


if __name__ == "__main__":
    main()
