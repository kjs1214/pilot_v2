"""
Skill text sources for cross-environment runs.

  none         : no skills
  procedural   : environment-agnostic procedural skills selected by procedure tags (procedure_map.json)
  skillrl_raw  : SkillRL's ALFWorld bank, template mode, with the ALFWorld category forced per env/task
                 (procedure_map.json -> skillrl_raw_category). Tests whether ALFWorld-worded skills transfer.
  mismatch_proc: procedural skills whose tags do NOT apply to this task (same format; wrong content)

All sources return text formatted like SkillRL's prompt section so the prompt skeleton is identical.
"""
from __future__ import annotations
import json, os
from typing import List, Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))


def _fmt_block(title: str, items: List[Dict]) -> str:
    lines = [f"### {title}"]
    for s in items:
        lines.append(f"- **{s['title']}**: {s['principle']}")
    return "\n".join(lines)


class ProceduralSkills:
    def __init__(self, path: Optional[str] = None, map_path: Optional[str] = None):
        self.bank = json.load(open(path or os.path.join(HERE, "procedural_skills.json")))["skills"]
        self.pmap = json.load(open(map_path or os.path.join(HERE, "procedure_map.json")))

    def tags_for(self, env: str, task_key: str) -> List[str]:
        m = self.pmap.get(env, {})
        return list(m.get(task_key, m.get("*", [])))

    def select(self, env: str, task_key: str, mismatch: bool = False) -> List[Dict]:
        tags = set(self.tags_for(env, task_key))
        if mismatch:
            pool = [s for s in self.bank if not (set(s["tags"]) & tags)]
            n = len([s for s in self.bank if set(s["tags"]) & tags])
            # same number of skill entries as the matched set: cycle through the non-applicable skills if there are fewer
            chosen = [pool[i % len(pool)] for i in range(max(1, n))] if pool else []
        else:
            chosen = [s for s in self.bank if set(s["tags"]) & tags]
        return chosen

    def text(self, env: str, task_key: str, mismatch: bool = False) -> str:
        return _fmt_block("Procedural Principles", self.select(env, task_key, mismatch))

    def ids(self, env: str, task_key: str, mismatch: bool = False) -> List[str]:
        return [s["skill_id"] for s in self.select(env, task_key, mismatch)]

    def raw_category(self, env: str, task_key: str) -> Optional[str]:
        m = self.pmap.get("skillrl_raw_category", {}).get(env, {})
        return m.get(task_key, m.get("*"))


class SkillRLRaw:
    """SkillRL bank in template mode with a forced ALFWorld category (for non-ALFWorld envs)."""

    def __init__(self, skillrl_root: str, skills_json: str, top_k: int = 6):
        import sys
        sys.path.insert(0, os.path.dirname(HERE))
        from common.skills_loader import load_skills_only_memory
        SOM = load_skills_only_memory(skillrl_root)
        self.mem = SOM(skills_json_path=skills_json, retrieval_mode="template")
        self.top_k = top_k
        # probe sentences that the keyword detector maps to each category
        self.probe = {"heat": "heat some apple and put it in fridge", "cool": "cool some apple and put it in fridge",
                      "clean": "clean some apple and put it in fridge", "pick_and_place": "put a apple in fridge",
                      "look_at_obj_in_light": "look at apple under the desklamp", "examine": "examine the apple with the desklamp"}

    def retrieve(self, category: str):
        r = self.mem.retrieve(task_description=self.probe[category], top_k=self.top_k)
        assert r.get("task_type") == category, (r.get("task_type"), category)
        return r

    def text_and_ids(self, category: str):
        r = self.retrieve(category)
        ids = []
        for v in r.values():
            if isinstance(v, list):
                ids += [it.get("skill_id") or it.get("mistake_id") for it in v if isinstance(it, dict)]
        return self.mem.format_for_prompt(r), ids
