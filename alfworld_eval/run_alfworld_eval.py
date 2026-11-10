#!/usr/bin/env python3
"""
Line 1: ALFWorld model ladder under surface perturbations (procedure unchanged).

Runs a model (base / SFT / RL / calibration) on a set of ALFWorld games with the exact SkillRL
prompt pipeline (template mode skills, history_length=2, <think>/<action> tags), optionally under a
surface perturbation, and logs every step so intrusion / collapse metrics can be computed later.

Example:
  python -m alfworld_eval.run_alfworld_eval --model /ckpt/rl_merged --tag rl --split seen \
      --skills_json $SKILLRL/memory_data/alfworld/claude_style_skills.json --perturb moveto --reps 2 \
      --out runs/rl_seen_moveto.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, time, re
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.prompting import History, build_prompt, extract_task, parse_response
from common.llm import make_llm
from alfworld_eval.perturb import Perturbation
from alfworld_eval.games import list_games, select_games


def make_env(game_file: str, max_steps: int):
    import textworld, textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    env_id = textworld.gym.register_games([game_file], request_infos, batch_size=1, asynchronous=False,
                                          max_episode_steps=max_steps,
                                          wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
    return textworld.gym.make(env_id)


def skill_ids(retrieved: Dict[str, Any]) -> List[str]:
    ids = []
    for v in retrieved.values():
        if isinstance(v, list):
            for it in v:
                if isinstance(it, dict):
                    ids.append(it.get("skill_id") or it.get("mistake_id") or it.get("title") or "?")
    return ids


class Episode:
    def __init__(self, game: Dict, rep: int, max_steps: int, pert: Perturbation, skills_mem, args):
        self.game, self.rep, self.max_steps, self.pert, self.args = game, rep, max_steps, pert, args
        self.env = make_env(game["game_file"], max_steps)
        obs, infos = self.env.reset()
        self.raw_obs = obs[0]
        self.admissible = list(infos["admissible_commands"][0])
        self.task_orig = extract_task(self.raw_obs)          # original wording (used for skill routing)
        self.history = History()
        self.done, self.won, self.t = False, False, 0
        self.steps: List[Dict[str, Any]] = []
        self.prompt_tokens: List[int] = []
        self.skill_ids: List[str] = []
        self.skills_text: Optional[str] = None
        self.routed_type: Optional[str] = None
        if args.skill_source in ("procedural", "mismatch_proc"):
            from skills.sources import ProceduralSkills
            ps = ProceduralSkills(); mm = args.skill_source == "mismatch_proc"
            txt = ps.text("alfworld", game["short"], mismatch=mm)
            self.skill_ids = ps.ids("alfworld", game["short"], mismatch=mm); self.routed_type = "procedural"
            self.skills_text = pert.text(txt) if args.perturb_skills else txt
        elif skills_mem is not None:
            retrieved = skills_mem.retrieve(task_description=self.task_orig, top_k=args.top_k,
                                            similarity_threshold=0.7, max_tokens=2000, include_examples=False)
            self.skill_ids = skill_ids(retrieved)
            self.routed_type = retrieved.get("task_type")
            txt = skills_mem.format_for_prompt(retrieved)
            self.skills_text = pert.text(txt) if args.perturb_skills else txt

    def prompt(self) -> str:
        obs_shown = self.pert.text(self.raw_obs)
        task_shown = self.pert.text(self.task_orig)
        adm_shown = self.pert.admissible(self.admissible)
        return build_prompt(task_shown, obs_shown, adm_shown, self.history, self.args.history_length,
                            init=(self.t == 0), skills_text=self.skills_text)

    def step(self, response: str):
        raw_action, has_think, think = parse_response(response)
        adm_shown = self.pert.admissible(self.admissible)
        if raw_action is None:
            # mirror SkillRL's projection: an unparsable response is sent as its last 30 chars (-> 'Nothing happens.')
            env_action = response.lower()[-30:]
            valid_format = False
        elif self.pert.kind != "none" and raw_action not in adm_shown:
            # Under a perturbation the admissible commands are the perturbed ones shown to the model. An off-list action
            # (e.g. the ORIGINAL wording 'go to cabinet 1' while the list says 'move to cabinet 1') does not exist in the
            # perturbed interface, so it must fail like any invalid command instead of being silently executed.
            env_action = "__offlist_under_perturbation__"
            valid_format = has_think
        else:
            env_action = self.pert.action_reverse(raw_action)
            valid_format = has_think
        in_adm_shown = raw_action in adm_shown if raw_action is not None else False
        in_adm_env = env_action in self.admissible
        obs, scores, dones, infos = self.env.step([env_action])
        self.won = bool(infos["won"][0])
        new_obs = obs[0]
        # history stores what the MODEL saw/said (perturbed surface), as SkillRL stores pre-step obs + action
        self.history.store(self.pert.text(self.raw_obs), raw_action if raw_action is not None else env_action)
        self.steps.append({"t": self.t, "raw_action": raw_action, "env_action": env_action,
                           "valid_format": valid_format, "in_admissible_shown": in_adm_shown,
                           "in_admissible_env": in_adm_env, "think": think if self.args.save_think else None,
                           "obs_after": new_obs if self.args.save_obs else None,
                           "n_admissible": len(self.admissible)})
        self.raw_obs = new_obs
        self.admissible = list(infos["admissible_commands"][0])
        self.t += 1
        self.done = bool(dones[0]) or self.won or self.t >= self.max_steps

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    def record(self) -> Dict[str, Any]:
        return {"tag": self.args.tag, "model": self.args.model, "split": self.args.split, "perturb": self.pert.kind,
                "skills": bool(self.skills_text is not None), "skill_cond": self.args.skill_cond, "routed_type": self.routed_type, "game_id": self.game["id"], "task_type": self.game["short"],
                "rep": self.rep, "success": self.won, "n_steps": self.t, "skill_ids": self.skill_ids,
                "retrieval_mode": self.args.retrieval_mode, "skill_source": self.args.skill_source, "top_k": self.args.top_k,
                "library": (os.path.splitext(os.path.basename(self.args.skills_json))[0] if self.args.skills_json else "none"),
                "prompt_tokens_mean": (sum(self.prompt_tokens) / len(self.prompt_tokens)) if self.prompt_tokens else None,
                "task": self.task_orig, "steps": self.steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--tag", required=True, help="label for this model (base/sft/rl/calib)")
    ap.add_argument("--split", default="seen", choices=["train", "seen", "unseen"])
    ap.add_argument("--alfworld_data", default=os.environ.get("ALFWORLD_DATA"))
    ap.add_argument("--per_type", type=int, default=0); ap.add_argument("--max_games", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0); ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--skills_json", default=None, help="SkillRL claude_style_skills.json; omit for skill-free")
    ap.add_argument("--skill_cond", default=None, help="label for the skill condition (full / nosk / mismatch); default from skills_json")
    ap.add_argument("--skillrl_root", default=os.environ.get("SKILLRL_ROOT"), help="repo root for agent_system imports")
    ap.add_argument("--top_k", type=int, default=6); ap.add_argument("--history_length", type=int, default=2)
    ap.add_argument("--retrieval_mode", default="template", choices=["template", "embedding"],
                    help="SkillRL retrieval: template (category keywords, all category skills) or embedding (top-k across all categories)")
    ap.add_argument("--embedding_model", default="Qwen/Qwen3-Embedding-0.6B"); ap.add_argument("--task_specific_top_k", type=int, default=None)
    ap.add_argument("--skill_source", default="skillrl", choices=["skillrl", "procedural", "mismatch_proc"],
                    help="skillrl = bank given by --skills_json; procedural/mismatch_proc = environment-agnostic procedural skills (skills/procedural_skills.json)")
    ap.add_argument("--perturb", default="none", choices=["none", "using", "moveto", "walkto", "fromtake", "names"])
    ap.add_argument("--names_map", default=None); ap.add_argument("--perturb_skills", action="store_true",
                    help="apply the names map to skill text too (recommended with --perturb names)")
    ap.add_argument("--max_steps", type=int, default=50); ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max_tokens", type=int, default=512); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tp", type=int, default=1); ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=8192); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chat_template_kwargs", default="{}", help='e.g. \'{"enable_thinking": false}\' for Qwen3')
    ap.add_argument("--save_think", action="store_true"); ap.add_argument("--save_obs", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.skillrl_root:
        sys.path.insert(0, args.skillrl_root)
    assert args.alfworld_data, "set ALFWORLD_DATA or --alfworld_data"
    games = select_games(list_games(args.alfworld_data, args.split), args.per_type, args.max_games, args.offset)
    assert games, "no games found"
    pert = Perturbation(args.perturb, args.names_map)
    if args.skill_cond is None:
        if args.skill_source != "skillrl":
            args.skill_cond = {"procedural": "proc", "mismatch_proc": "proc_mismatch"}[args.skill_source]
        else:
            b = os.path.basename(args.skills_json) if args.skills_json else ""
            LABELS = [("mismatch", "mismatch"), ("nogeneral6", "nogen6"), ("nogeneral", "nogen"), ("genonly6", "genonly6"), ("genonly", "genonly"), ("placebo", "placebo")]
            args.skill_cond = "nosk" if not b else next((lab for key, lab in LABELS if key in b), "full")
    skills_mem = None
    if args.skills_json and args.skill_source == "skillrl":
        assert args.skillrl_root, "set SKILLRL_ROOT or --skillrl_root (needed to load SkillRL's SkillsOnlyMemory)"
        from common.skills_loader import load_skills_only_memory
        SkillsOnlyMemory = load_skills_only_memory(args.skillrl_root)
        if args.retrieval_mode == "embedding":
            skills_mem = SkillsOnlyMemory(skills_json_path=args.skills_json, retrieval_mode="embedding",
                                          embedding_model_path=args.embedding_model, task_specific_top_k=args.task_specific_top_k)
        else:
            skills_mem = SkillsOnlyMemory(skills_json_path=args.skills_json, retrieval_mode="template")
    llm = make_llm(args.model, args.tokenizer, tensor_parallel=args.tp, gpu_memory_utilization=args.gpu_mem,
                   max_model_len=args.max_model_len, seed=args.seed,
                   chat_template_kwargs=json.loads(args.chat_template_kwargs))

    todo = [(g, r) for r in range(args.reps) for g in games]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fout = open(args.out, "a")
    active: List[Episode] = []
    n_done, n_succ, t0 = 0, 0, time.time()
    print(f"[run] {len(games)} games x {args.reps} reps, perturb={args.perturb}, skills={bool(skills_mem)}", flush=True)
    while todo or active:
        while todo and len(active) < args.batch:
            g, r = todo.pop(0)
            active.append(Episode(g, r, args.max_steps, pert, skills_mem, args))
        prompts = [ep.prompt() for ep in active]
        tok = getattr(llm, "tokenizer", None)
        if tok is not None:
            for ep, pr in zip(active, prompts):
                ep.prompt_tokens.append(len(tok(pr, add_special_tokens=False)["input_ids"]))
        responses = llm.generate(prompts, temperature=args.temperature, max_tokens=args.max_tokens)
        still = []
        for ep, resp in zip(active, responses):
            ep.step(resp)
            if ep.done:
                fout.write(json.dumps(ep.record(), ensure_ascii=False) + "\n"); fout.flush()
                n_done += 1; n_succ += int(ep.won); ep.close()
            else:
                still.append(ep)
        active = still
        if n_done and n_done % 20 == 0:
            print(f"[run] done={n_done} success={n_succ/n_done:.3f} elapsed={time.time()-t0:.0f}s", flush=True)
    fout.close()
    print(f"[run] finished: {n_done} episodes, success={n_succ/max(1,n_done):.3f}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
