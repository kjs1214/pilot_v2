"""
Prompt construction that mirrors SkillRL's ALFWorld prompt pipeline
(agent_system/environments/env_manager.py + prompts/alfworld.py + memory/memory.py).

If the SkillRL repo is importable we reuse its templates; otherwise we fall back to
verbatim copies (identical to the repo at the time of writing).
"""
from __future__ import annotations
from typing import List, Dict, Optional
import re

try:
    from agent_system.environments.prompts.alfworld import (
        ALFWORLD_TEMPLATE_NO_HIS, ALFWORLD_TEMPLATE, ALFWORLD_TEMPLATE_WITH_MEMORY,
    )
except Exception:  # verbatim fallback copies
    ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""
    ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""
    ALFWORLD_TEMPLATE_WITH_MEMORY = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}

## Retrieved Relevant Experience

{retrieved_memories}

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


class History:
    """Same formatting as SkillRL SimpleMemory.fetch()."""

    def __init__(self):
        self.records: List[Dict[str, str]] = []

    def store(self, text_obs: str, action: str):
        self.records.append({"text_obs": text_obs, "action": action})

    def __len__(self):
        return len(self.records)

    def fetch(self, history_length: int):
        recent = self.records[-history_length:] if history_length > 0 else []
        start_idx = len(self.records) - len(recent)
        lines = []
        for j, rec in enumerate(recent):
            step_num = start_idx + j + 1
            lines.append(f"[Observation {step_num}: '{rec['text_obs']}', Action {step_num}: '{rec['action']}']")
        return "\n".join(lines), len(recent)


def format_admissible(admissible: List[str]) -> str:
    # identical to SkillRL: exclude 'help'
    return "\n ".join(f"'{s}'" for s in admissible if s != "help")


def build_prompt(task: str, obs: str, admissible: List[str], history: History,
                 history_length: int, init: bool, skills_text: Optional[str],
                 env_name: str = "ALFRED Embodied Environment") -> str:
    adm = format_admissible(admissible)
    if init or history_length <= 0:
        p = ALFWORLD_TEMPLATE_NO_HIS.format(current_observation=obs, admissible_actions=adm)
    else:
        action_history, valid_len = history.fetch(history_length)
        kw = dict(task_description=task, step_count=len(history), history_length=valid_len,
                  action_history=action_history, current_step=len(history) + 1,
                  current_observation=obs, admissible_actions=adm)
        if skills_text is not None:
            p = ALFWORLD_TEMPLATE_WITH_MEMORY.format(retrieved_memories=skills_text, **kw)
        else:
            p = ALFWORLD_TEMPLATE.format(**kw)
    if env_name != "ALFRED Embodied Environment":
        p = p.replace("ALFRED Embodied Environment", env_name)
    return p


def extract_task(text_obs: str) -> str:
    i = text_obs.find("Your task is to: ")
    if i == -1:
        raise ValueError("Task description not found in text observation.")
    return text_obs[i + len("Your task is to: "):].strip()


_THINK = re.compile(r"<think>(.*?)</think>", re.S)
_ACTION = re.compile(r"<action>(.*?)</action>", re.S)


def parse_response(text: str):
    """Mirror of alfworld_projection: lowercase the text and take the <action>..</action> span.
    Returns (action_or_None, has_think, think_text)."""
    low = text.lower()
    tm = _THINK.search(text)
    think_text = tm.group(1).strip() if tm else ""
    m = _ACTION.search(low)
    if m is None:
        return None, tm is not None, think_text
    return m.group(1).strip(), tm is not None, think_text
