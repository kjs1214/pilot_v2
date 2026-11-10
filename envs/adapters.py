"""
Uniform episode adapters for the procedure-sharing environments.
Each adapter exposes: name, env_name (for the prompt), jobs(args) -> list of job dicts,
make_env(args) -> env handle, start(env, job) -> (task_text, obs, admissible), step(env, action) -> (obs, admissible, done, success, score),
intrusion patterns (ALFWorld-specific wording that is NOT this env's syntax).
"""
from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple

# ALFWorld command shapes / names that are not valid syntax in the target env (used for intrusion counts)
ALF_WORDING = [re.compile(r"^(heat|cool|clean) .+ with .+$"), re.compile(r"^take .+ from .+$"),
               re.compile(r"^go to .+$"), re.compile(r"^use (desklamp|floorlamp)"),
               re.compile(r"\b(sinkbasin|countertop|garbagecan|desklamp|microwave \d|fridge \d|cabinet \d|drawer \d|stoveburner)\b")]

SW_ALWAYS = ["look around", "inventory", "task", "wait", "wait1"]
SW_PRIORITY = [("focus on", 40), ("go to", 20), ("teleport to", 20), ("open", 20), ("close", 20), ("pick up", 40), ("activate", 20),
               ("deactivate", 20), ("move", 40), ("pour", 20), ("use", 20), ("look at", 30), ("look in", 20),
               ("examine", 20), ("put down", 10), ("read", 10)]


def sw_cap(valid: List[str], cap: int) -> List[str]:
    if cap <= 0 or len(valid) <= cap:
        return list(valid)
    seen, out = set(), []
    def add(a):
        if a not in seen and len(out) < cap:
            seen.add(a); out.append(a)
    for a in SW_ALWAYS:
        if a in valid: add(a)
    pool = sorted(set(valid))
    for verb, k in SW_PRIORITY:
        for a in [x for x in pool if x.startswith(verb + " ")][:k]: add(a)
    low = ("connect ", "disconnect ", "mix ", "eat ", "flush ", "dunk ")
    for a in pool:
        if not a.startswith(low): add(a)
    for a in pool: add(a)
    return out


SW_TWO_OBJ = ("move ", "pour ", "dunk ", "use ", "connect ")


def sw_held_list(env) -> List[str]:
    """ScienceWorld's valid action list, with one rule: two-object actions (move/pour/dunk/use X ... Y) are listed only
    when X is something the agent holds. Everything else is listed as the environment returns it.
    Validated by analysis/gold_check.py: every gold solution can be executed using only actions from this list."""
    wt = env.get_valid_action_object_combinations_with_templates()
    held = {x["obj_ids"][0] for x in wt if x["action"].startswith("put down ") and x.get("obj_ids")}
    out = []
    for x in wt:
        a = x["action"]
        if a.startswith(SW_TWO_OBJ) and len(x.get("obj_ids", [])) >= 2 and x["obj_ids"][0] not in held:
            continue
        out.append(a)
    return out


class ScienceWorldAdapter:
    name, env_name = "scienceworld", "ScienceWorld Environment"
    intrusion = [p for p in ALF_WORDING if "go to" not in p.pattern]   # 'go to LOC' is valid ScienceWorld syntax

    @staticmethod
    def _ticks(args) -> int:
        # ScienceWorld's own step limit (library default 100). It counts internal ticks: one 'wait' costs 10.
        # By default it equals the action budget (--max_steps); --sw_tick_limit > 0 overrides it (non-default, must be reported).
        t = int(getattr(args, "sw_tick_limit", 0) or 0)
        return t if t > 0 else int(args.max_steps)

    def _shown(self, env, valid, args) -> List[str]:
        if getattr(args, "sw_list", "held") == "held":
            return sw_held_list(env)
        return sw_cap(valid, args.max_actions)

    def jobs(self, args) -> List[Dict]:
        from scienceworld import ScienceWorldEnv
        env = ScienceWorldEnv("", envStepLimit=self._ticks(args))
        out = []
        for t in args.tasks.split(","):
            env.load(t, 0, args.simplifications, False)
            vs = {"train": env.get_variations_train, "dev": env.get_variations_dev, "test": env.get_variations_test}[args.split]()
            if args.max_variations > 0: vs = vs[: args.max_variations]
            out += [{"task_key": t, "variation": v, "rep": r} for r in range(args.reps) for v in vs]
        env.close()
        return out

    def make_env(self, args):
        from scienceworld import ScienceWorldEnv
        return ScienceWorldEnv("", envStepLimit=self._ticks(args))

    def start(self, env, job, args, gold: bool = False):
        env.load(job["task_key"], job["variation"], args.simplifications, gold)
        obs, info = env.reset()
        task = env.get_task_description()
        valid = list(info.get("valid", env.get_valid_action_object_combinations()))
        return task, f"{obs}\n\nYour task is to: {task}", self._shown(env, valid, args), valid

    def step(self, env, action, args):
        obs, reward, done, info = env.step(action)
        score = int(info.get("score", 0)); valid = list(info.get("valid", []))
        return obs, self._shown(env, valid, args), valid, bool(done) or score >= 100, score >= 100, score


class TWXAdapter:
    """TextWorldExpress games: twc (tidy-up) and cookingworld."""
    # TextWorldExpress LIBRARY DEFAULTS (what the games generate with gameParams=""; verified with getGenerationProperties()).
    DEFAULT_PARAMS = {"twc": "numLocations=3, numItemsToPutAway=4, includeDoors=0, limitInventorySize=0",
                      "cookingworld": "numLocations=11, numIngredients=3, numDistractorItems=10, includeDoors=1, limitInventorySize=1"}
    # Settings we used earlier. NOTE: 'reduced' was mislabelled "original" in early runs/logs; it is easier than the library default.
    REDUCED_PARAMS = {"twc": "numLocations=3, numItemsToPutAway=2, includeDoors=0, limitInventorySize=0",
                      "cookingworld": "numLocations=3, numIngredients=2, numDistractorItems=5, includeDoors=0, limitInventorySize=0"}
    EASY_PARAMS = {"twc": "numLocations=1, numItemsToPutAway=1, includeDoors=0, limitInventorySize=0",
                   "cookingworld": "numLocations=1, numIngredients=1, numDistractorItems=3, includeDoors=0, limitInventorySize=0"}
    PRESETS = {"default": DEFAULT_PARAMS, "reduced": REDUCED_PARAMS, "easy": EASY_PARAMS}

    def params(self, args) -> str:
        gp = getattr(args, "game_params", None)
        if not gp: return self.DEFAULT_PARAMS[self.game]
        return self.PRESETS[gp][self.game] if gp in self.PRESETS else gp      # a preset name or a raw gameParams string
    intrusion = ALF_WORDING   # 'go to' is not TWX syntax (it uses 'move west'), so it counts

    def __init__(self, game: str):
        self.name = game; self.game = game
        self.env_name = {"twc": "TextWorld Commonsense Environment", "cookingworld": "CookingWorld Environment"}[game]

    def jobs(self, args) -> List[Dict]:
        from textworld_express import TextWorldExpressEnv
        env = TextWorldExpressEnv(envStepLimit=args.max_steps)
        env.load(gameName=self.game, gameParams=self.params(args))
        seeds = {"train": env.getValidSeedsTrain, "dev": env.getValidSeedsDev, "test": env.getValidSeedsTest}[args.split]()
        if args.max_variations > 0: seeds = seeds[: args.max_variations]
        env.close()
        return [{"task_key": "*", "variation": s, "rep": r} for r in range(args.reps) for s in seeds]

    def make_env(self, args):
        from textworld_express import TextWorldExpressEnv
        env = TextWorldExpressEnv(envStepLimit=args.max_steps)
        env.load(gameName=self.game, gameParams=self.params(args))
        return env

    def start(self, env, job, args):
        obs, info = env.reset(seed=job["variation"], gameFold=args.split, generateGoldPath=False)
        task = env.getTaskDescription()
        valid = list(info.get("validActions", []))
        return task, f"{obs}\n\nYour task is to: {task}", valid, valid

    def step(self, env, action, args):
        obs, reward, done, info = env.step(action)
        valid = list(info.get("validActions", [])); score = float(info.get("score", 0.0))
        return obs, valid, valid, bool(done or info.get("tasksuccess") or info.get("taskfailure")), bool(info.get("tasksuccess")), score


def get_adapter(name: str):
    if name == "scienceworld": return ScienceWorldAdapter()
    if name in ("twc", "cookingworld"): return TWXAdapter(name)
    raise ValueError(name)
