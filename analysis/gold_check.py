#!/usr/bin/env python3
"""Harness validation (no GPU). Replays each environment's own gold solution THROUGH OUR ADAPTER with the evaluation
settings and checks that the harness itself does not prevent success:
  - the gold solution succeeds within the action budget
  - every action it needs is available in the action list we show the model

TextWorldExpress (twc, cw): gold actions must literally be in the shown list.
ScienceWorld (sw): gold strings often use a different referent than the valid-action list ('examine X' vs 'look at X',
  'open door to kitchen' vs 'open door'), and gold paths contain redundant steps (re-opening an open door) and occasional
  invalid steps. So we do a LIST-ONLY replay: at every gold step we execute an equivalent action taken from the shown list.
  A step counts against the list policy only if the full valid list could express it but the shown list could not.

Usage: python analysis/gold_check.py twc|cw [default|reduced|easy]     (default = TextWorldExpress library default)
       python analysis/gold_check.py sw [task,task,...] [extended]    (default: library defaults, score-based check)
"""
import os, sys, argparse, collections, statistics as st
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.adapters import get_adapter

SW_TASKS = "boil,melt,freeze,change-the-state-of-matter-of"


def ns(**kw):
    base = dict(max_steps=100, max_actions=0, split="test", max_variations=20, reps=1, game_params=None,
                simplifications="", tasks=SW_TASKS, sw_list="held", sw_tick_limit=0)
    base.update(kw); return argparse.Namespace(**base)


# ---------------------------------------------------------------- TextWorldExpress
def run_twx(game, params):
    ad = get_adapter(game); a = ns(game_params=params); env = ad.make_env(a); rows = []
    for job in ad.jobs(a):
        obs, info = env.reset(seed=job["variation"], gameFold="test", generateGoldPath=True)
        gold = list(env.getGoldActionSequence()); shown = list(info.get("validActions", []))
        missing = 0; success = False; n = 0
        for act in gold:
            if act not in shown: missing += 1
            obs, shown, valid, done, success, score = ad.step(env, act, a); n += 1
            if done: break
        rows.append((len(gold), n, missing, success))
    env.close()
    L = [r[0] for r in rows]
    print(f"   gold solution length: median {st.median(L):.0f}, min {min(L)}, max {max(L)}  (action budget {a.max_steps})")
    print(f"   gold replay through our adapter: success {sum(1 for r in rows if r[3])}/{len(rows)}")
    print(f"   gold actions NOT in the list shown to the model: {sum(r[2] for r in rows)} of {sum(r[1] for r in rows)} steps")


# ---------------------------------------------------------------- ScienceWorld
def _objwords(s):
    return set(s.replace(" to ", " ").replace(" on ", " ").replace(" in ", " ").split())


def equivalent(g, shown):
    """An action from `shown` with the same effect as gold string g; '__noop__' if g is redundant in this state; None if none."""
    S = set(shown)
    if g in S: return g
    cands = [g.replace("examine ", "look at ", 1) if g.startswith("examine ") else g]
    cands += [c.replace(" in inventory", "") for c in list(cands)]
    for c in cands:
        if c in S: return c
    w = g.split()
    if w[0] in ("open", "close") and "door" in g:              # door referents differ by room: 'door', 'door to X', 'X door'
        target = g.split("door to ")[-1] if "door to " in g else ""
        doors = [a for a in shown if a.split()[0] == w[0] and "door" in a]
        named = [a for a in doors if target and target in a]
        if len(named) == 1: return named[0]
        if len(doors) == 1: return doors[0]
    if g.startswith("pick up ") and ("put down " + g[len("pick up "):]) in S:
        return "__noop__"                                       # gold re-picks something already held
    if w[0] == "open":                                          # gold re-opens something already open
        if "door" in g:
            target = g.split("door to ")[-1] if "door to " in g else ""
            closes = [a for a in shown if a.startswith("close ") and "door" in a]
            if any(target and target in a for a in closes) or (closes and not [a for a in shown if a.startswith("open ") and "door" in a]):
                return "__noop__"
        elif ("close " + g[len("open "):]) in S:
            return "__noop__"
    verb = " ".join(w[:2]) if w[0] in ("look", "pick", "go", "focus", "put") else w[0]
    ow = _objwords(" ".join(cands[-1].split()[len(verb.split()):]))
    pool = [a for a in shown if a.startswith(verb + " ") and ow <= _objwords(a)]
    return pool[0] if len(pool) == 1 else None


def run_sw(tasks, extended=False):
    """Library defaults (no simplification, 100-step limit): many gold solutions do not fit in 100 steps, so the benchmark is
    scored by progress score. We check that executing gold with ONLY listed actions reaches the same score as exact gold.
    extended=True: noElectricalAction, 200 actions, 500 ticks -> every gold solution can complete; checks 36/36 completion."""
    ad = get_adapter("scienceworld")
    a = ns(max_steps=200, sw_tick_limit=500, simplifications="noElectricalAction", tasks=tasks) if extended else ns(max_steps=100, tasks=tasks)
    env = ad.make_env(a); rows = []; sizes = []; pol = collections.Counter(); glitch = 0
    for job in ad.jobs(a):
        task, obs, shown, valid = ad.start(env, job, a, gold=True); sc = 0; n = 0          # (1) exact gold strings
        for g in list(env.get_gold_action_sequence()):
            obs, shown, valid, done, success, sc = ad.step(env, g, a); n += 1
            if done or n >= a.max_steps: break
        exact = sc
        task, obs, shown, valid = ad.start(env, job, a, gold=True); sc = 0; n = 0; pv = 0   # (2) listed actions only
        for g in list(env.get_gold_action_sequence()):
            sizes.append(len(shown)); e = equivalent(g, shown)
            if e == "__noop__": continue
            if e is None:
                if equivalent(g, valid) is None: glitch += 1; continue                    # gold-path glitch: no valid form at all
                pv += 1; pol[" ".join(g.split()[:2])] += 1; e = g
            obs, shown, valid, done, success, sc = ad.step(env, e, a); n += 1
            if done or n >= a.max_steps: break
        rows.append((job["task_key"], exact, sc, pv))
    env.close()
    print(f"   settings: simplifications='{a.simplifications}', action budget {a.max_steps}, tick limit {ad._ticks(a)}, list policy '{a.sw_list}'")
    print(f"   exact gold : mean score {st.mean(r[1] for r in rows):.1f}, completed {sum(r[1] >= 100 for r in rows)}/{len(rows)}")
    print(f"   listed only: mean score {st.mean(r[2] for r in rows):.1f}, completed {sum(r[2] >= 100 for r in rows)}/{len(rows)}")
    print(f"   steps the shown list could not express but the full valid list could (policy gaps): {sum(r[3] for r in rows)}")
    print(f"   gold steps with no valid form at all (gold-path glitches, skipped): {glitch} | shown-list size: median {st.median(sizes):.0f}, max {max(sizes)}")
    for t in tasks.split(","):
        rr = [r for r in rows if r[0] == t]
        print(f"     {t:32s} exact {st.mean(r[1] for r in rr):.1f}  listed-only {st.mean(r[2] for r in rr):.1f}  policy gaps {sum(r[3] for r in rr)}")
    if pol: print("   policy gaps by verb:", dict(pol))


if __name__ == "__main__":
    which = sys.argv[1]
    if which in ("twc", "cw"):
        game = {"twc": "twc", "cw": "cookingworld"}[which]
        preset = sys.argv[2] if len(sys.argv) > 2 else "default"
        ad = get_adapter(game); print(f"== {game}, setting '{preset}': {ad.params(ns(game_params=preset))}")
        run_twx(game, preset)
    else:
        tasks = sys.argv[2] if len(sys.argv) > 2 else SW_TASKS
        ext = len(sys.argv) > 3 and sys.argv[3] == "extended"
        print(f"== ScienceWorld ({tasks}){' [extended budget, non-default]' if ext else ' [library defaults]'}"); run_sw(tasks, ext)
