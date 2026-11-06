#!/usr/bin/env python3
"""
Cross-environment runner (procedure-sharing envs): scienceworld | twc | cookingworld.
Same prompt skeleton as the ALFWorld runs; skills come from --skills:
  none | procedural | mismatch_proc | skillrl_raw   (see skills/sources.py)

Example:
  python run_eval.py --env scienceworld --model ~/ckpt/rl_merged --tag rl --skills procedural \
      --tasks boil,melt,freeze,change-the-state-of-matter-of --split test --max_variations 10 --reps 2 --out runs/transfer/sw_rl_procedural.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, time
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.prompting import History, build_prompt, parse_response
from common.llm import make_llm
from envs.adapters import get_adapter
from skills.sources import ProceduralSkills, SkillRLRaw


class Episode:
    def __init__(self, adapter, env, job, args, skills_text, skill_ids, tok):
        self.ad, self.env, self.job, self.args = adapter, env, job, args
        self.task, self.obs, self.shown, self.valid = adapter.start(env, job, args)
        self.history, self.t, self.done, self.success, self.score = History(), 0, False, False, 0
        self.skills_text, self.skill_ids, self.tok = skills_text, skill_ids, tok
        self.steps: List[Dict[str, Any]] = []

    def prompt(self):
        return build_prompt(self.task, self.obs, self.shown, self.history, self.args.history_length,
                            init=(self.t == 0), skills_text=self.skills_text, env_name=self.ad.env_name)

    def step(self, response: str):
        raw, has_think, think = parse_response(response)
        env_action = raw if raw is not None else response.lower()[-30:]
        shown_before = self.shown
        obs, shown, valid, done, success, score = self.ad.step(self.env, env_action, self.args)
        intr = raw is not None and (raw not in shown_before) and any(p.search(raw) for p in self.ad.intrusion)
        self.history.store(self.obs, env_action)
        self.steps.append({"t": self.t, "raw_action": raw, "valid_format": has_think and raw is not None,
                           "in_admissible_shown": raw in shown_before if raw is not None else False,
                           "alf_intrusion": intr, "score": score, "n_valid": len(self.valid), "n_shown": len(shown_before),
                           "think": think if self.args.save_think else None})
        self.obs, self.shown, self.valid, self.score = obs, shown, valid, score
        self.t += 1
        self.done = done or self.t >= self.args.max_steps
        self.success = success

    def record(self):
        return {"env": self.ad.name, "tag": self.args.tag, "model": self.args.model, "skills": self.args.skills,
                "task_key": self.job["task_key"], "variation": self.job["variation"], "rep": self.job["rep"],
                "success": self.success, "score": self.score, "n_steps": self.t, "skill_ids": self.skill_ids,
                "task": self.task, "steps": self.steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["scienceworld", "twc", "cookingworld"])
    ap.add_argument("--model", required=True); ap.add_argument("--tokenizer", default=None); ap.add_argument("--tag", required=True)
    ap.add_argument("--skills", default="none", choices=["none", "procedural", "mismatch_proc", "skillrl_raw"])
    ap.add_argument("--skillrl_root", default=os.environ.get("SKILLRL_ROOT")); ap.add_argument("--skills_json", default=os.environ.get("SKILLS"))
    ap.add_argument("--tasks", default="boil,melt,freeze,change-the-state-of-matter-of", help="scienceworld task names")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"]); ap.add_argument("--max_variations", type=int, default=10)
    ap.add_argument("--reps", type=int, default=1); ap.add_argument("--simplifications", default="noElectricalAction")
    ap.add_argument("--game_params", default=None, help="TextWorldExpress gameParams override")
    ap.add_argument("--max_steps", type=int, default=50); ap.add_argument("--max_actions", type=int, default=150)
    ap.add_argument("--history_length", type=int, default=2); ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max_tokens", type=int, default=512); ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.85); ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--chat_template_kwargs", default="{}")
    ap.add_argument("--parallel", type=int, default=8); ap.add_argument("--save_think", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ad = get_adapter(args.env)
    jobs = ad.jobs(args)
    print(f"[{args.env}] {len(jobs)} episodes, skills={args.skills}", flush=True)
    proc = ProceduralSkills()
    raw = SkillRLRaw(args.skillrl_root, args.skills_json) if args.skills == "skillrl_raw" else None

    def skills_for(job):
        if args.skills == "none": return None, []
        if args.skills in ("procedural", "mismatch_proc"):
            mm = args.skills == "mismatch_proc"
            return proc.text(ad.name, job["task_key"], mismatch=mm), proc.ids(ad.name, job["task_key"], mismatch=mm)
        cat = proc.raw_category(ad.name, job["task_key"]); return raw.text_and_ids(cat)

    llm = make_llm(args.model, args.tokenizer, tensor_parallel=args.tp, gpu_memory_utilization=args.gpu_mem,
                   max_model_len=args.max_model_len, seed=args.seed, chat_template_kwargs=json.loads(args.chat_template_kwargs))
    n_par = max(1, min(args.parallel, len(jobs)))
    envs = [ad.make_env(args) for _ in range(n_par)]
    free, active, todo = list(range(n_par)), [], list(jobs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fout = open(args.out, "a"); t0 = time.time(); done_n = 0; succ = 0
    while todo or active:
        while todo and free:
            job = todo.pop(0); ei = free.pop(0); st, ids = skills_for(job)
            active.append((ei, Episode(ad, envs[ei], job, args, st, ids, None)))
        resps = llm.generate([ep.prompt() for _, ep in active], temperature=args.temperature, max_tokens=args.max_tokens)
        still = []
        for (ei, ep), resp in zip(active, resps):
            ep.step(resp)
            if ep.done:
                fout.write(json.dumps(ep.record(), ensure_ascii=False) + "\n"); fout.flush()
                done_n += 1; succ += int(ep.success); free.append(ei)
                if done_n % 10 == 0:
                    print(f"[{args.env}] {done_n}/{len(jobs)} success={succ/done_n:.3f} elapsed={time.time()-t0:.0f}s", flush=True)
            else:
                still.append((ei, ep))
        active = still
    fout.close()
    for e in envs:
        try: e.close()
        except Exception: pass
    print(f"[{args.env}] finished: {done_n} episodes, success={succ/max(1,done_n):.3f}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
