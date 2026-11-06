"""Batched generation with vLLM (offline), one user turn rendered with the model's chat template (as SkillRL does)."""
from __future__ import annotations
from typing import List, Optional, Dict, Any


class VLLMChat:
    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None, tensor_parallel: int = 1,
                 gpu_memory_utilization: float = 0.85, max_model_len: int = 6144, seed: int = 0,
                 chat_template_kwargs: Optional[Dict[str, Any]] = None, dtype: str = "bfloat16"):
        from vllm import LLM
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=True)
        self.llm = LLM(model=model_path, tokenizer=tokenizer_path or model_path,
                       tensor_parallel_size=tensor_parallel, gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, seed=seed, dtype=dtype, trust_remote_code=True)
        self.chat_template_kwargs = chat_template_kwargs or {}

    def render(self, user_text: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}], add_generation_prompt=True, tokenize=False,
            **self.chat_template_kwargs)

    def generate(self, user_texts: List[str], temperature: float = 0.4, max_tokens: int = 512,
                 top_p: float = 1.0) -> List[str]:
        from vllm import SamplingParams
        if not user_texts:
            return []
        sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        outs = self.llm.generate([self.render(t) for t in user_texts], sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


class MockChat:
    """CPU-only stand-in for VLLMChat (set PILOT_MOCK_LLM=1): picks a random admissible action from the prompt.
    Use it to dry-run the whole pipeline (env, perturbations, logging, metrics) without a GPU."""

    def __init__(self, *a, seed: int = 0, **k):
        import random
        self.rng = random.Random(seed)

    def generate(self, user_texts, temperature=0.4, max_tokens=512, top_p=1.0):
        import re
        outs = []
        for p in user_texts:
            m = re.search(r"Your admissible actions of the current situation are: \[(.*)\]\.", p, re.S)
            acts = re.findall(r"'([^']*)'", m.group(1)) if m else ["look"]
            a = self.rng.choice(acts) if acts else "look"
            outs.append(f"<think>mock: {a}</think>\n<action>{a}</action>")
        return outs


def make_llm(*args, **kwargs):
    import os
    if os.environ.get("PILOT_MOCK_LLM"):
        return MockChat(*args, **kwargs)
    return VLLMChat(*args, **kwargs)
