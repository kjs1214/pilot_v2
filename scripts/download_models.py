#!/usr/bin/env python3
"""Download SkillRL checkpoints (SFT full, RL model shards only) with huggingface_hub (no CLI naming issues)."""
import argparse, os
from huggingface_hub import snapshot_download
ap = argparse.ArgumentParser(); ap.add_argument("--dest", default=os.path.expanduser("~/ckpt")); a = ap.parse_args()
os.makedirs(a.dest, exist_ok=True)
print("[download] SFT ->", snapshot_download("Jianwen/Alfworld-7B-SFT", local_dir=os.path.join(a.dest, "sft")))
print("[download] RL (model shards + config/tokenizer, no optimizer) ->",
      snapshot_download("Jianwen/Alfworld-7B-RL", local_dir=os.path.join(a.dest, "rl_raw"),
                        allow_patterns=["actor/model_*", "actor/extra_state_*", "actor/*.json", "actor/*.txt", "actor/*.jinja"]))
