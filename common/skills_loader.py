"""
Load SkillRL's SkillsOnlyMemory without importing agent_system.memory/__init__.py
(that __init__ also imports RetrievalMemory, which requires sentence-transformers + faiss).
Only base.py and skills_only_memory.py are needed for template mode.
"""
from __future__ import annotations
import importlib.util
import os
import sys
import types


def load_skills_only_memory(skillrl_root: str):
    mem_dir = os.path.join(skillrl_root, "agent_system", "memory")
    pkg_name = "_skillrl_memory"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [mem_dir]
        sys.modules[pkg_name] = pkg
        for mod in ("base", "skills_only_memory"):
            spec = importlib.util.spec_from_file_location(f"{pkg_name}.{mod}", os.path.join(mem_dir, f"{mod}.py"))
            m = importlib.util.module_from_spec(spec)
            sys.modules[f"{pkg_name}.{mod}"] = m
            spec.loader.exec_module(m)
    return sys.modules[f"{pkg_name}.skills_only_memory"].SkillsOnlyMemory
