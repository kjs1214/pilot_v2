#!/usr/bin/env python3
"""
Standalone FSDP -> HuggingFace merge for verl actor checkpoints (e.g. Jianwen/Alfworld-7B-RL/actor).
Follows the logic of verl's scripts/model_merger.py (FSDP path, no TP) but only needs
torch + safetensors (no ray / tensordict / verl install).

  python scripts/merge_fsdp_ckpt.py --local_dir ~/ckpt/rl_raw/actor --target_dir ~/ckpt/rl_merged

Input dir must contain model_world_size_{W}_rank_{r}.pt for r in [0, W) plus the HF config/tokenizer
files that verl saves next to them (config.json, tokenizer*.json, ...). Output: bf16 safetensors shards
+ index + copied config/tokenizer files, loadable by vLLM / transformers.
"""
from __future__ import annotations
import argparse, glob, json, os, re, shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

try:
    from torch.distributed.tensor import DTensor  # torch >= 2.4
except Exception:  # pragma: no cover
    from torch.distributed._tensor import DTensor

COPY_PATTERNS = ["config.json", "generation_config.json", "tokenizer*.json", "vocab.json", "merges.txt",
                 "special_tokens_map.json", "added_tokens.json", "chat_template.jinja", "*.model"]


def world_size(local_dir: Path) -> int:
    for f in os.listdir(local_dir):
        m = re.match(r"model_world_size_(\d+)_rank_0\.pt", f)
        if m:
            return int(m.group(1))
    raise FileNotFoundError(f"no model_world_size_*_rank_0.pt in {local_dir}")


def local_and_placement(t):
    """Return (local_tensor, placement_or_None) for a DTensor or plain tensor."""
    if isinstance(t, DTensor):
        pl = tuple(t.placements)
        names = t.device_mesh.mesh_dim_names or ("fsdp",)
        if names and names[0] in ("dp", "ddp"):       # replicated dp dim can be discarded (as in verl)
            pl = pl[1:]
        assert len(pl) == 1, f"only 1-D FSDP meshes are supported, got placements {pl}"
        return t._local_tensor, pl[0]
    return t, None


def merge(tensors, placement):
    if placement is None:                 # non-DTensor: FSDP shards along dim 0 (as in verl)
        return torch.cat(tensors, dim=0).contiguous()
    if placement.is_replicate():
        return tensors[0]
    if placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    raise NotImplementedError(f"unsupported placement {placement}")


def save_sharded(state, out: Path, max_shard_bytes: int = 4 * 1024 ** 3):
    keys = sorted(state)
    shards, cur, cur_bytes = [], {}, 0
    for k in keys:
        nb = state[k].numel() * state[k].element_size()
        if cur and cur_bytes + nb > max_shard_bytes:
            shards.append(cur); cur, cur_bytes = {}, 0
        cur[k] = state[k]; cur_bytes += nb
    if cur:
        shards.append(cur)
    weight_map, total = {}, 0
    n = len(shards)
    for i, sh in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{n:05d}.safetensors"
        save_file({k: v.contiguous() for k, v in sh.items()}, str(out / name), metadata={"format": "pt"})
        for k, v in sh.items():
            weight_map[k] = name; total += v.numel() * v.element_size()
    with open(out / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", required=True); ap.add_argument("--target_dir", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()
    src, out = Path(os.path.expanduser(args.local_dir)), Path(os.path.expanduser(args.target_dir))
    out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    W = world_size(src)
    print(f"[merge] world_size={W}")
    shards = []
    for r in range(W):
        p = src / f"model_world_size_{W}_rank_{r}.pt"
        print(f"[merge] loading {p.name}", flush=True)
        shards.append(torch.load(p, map_location="cpu", weights_only=False))
    keys = list(shards[0].keys())
    merged = {}
    for k in keys:
        parts, pl0 = [], "unset"
        for r in range(W):
            loc, pl = local_and_placement(shards[r].pop(k))
            if pl0 == "unset":
                pl0 = pl
            parts.append(loc.to(dtype))
        merged[k] = merge(parts, pl0)
        del parts
    del shards
    # strip possible wrapper prefixes (defensive; verl normally saves clean HF names)
    clean = {}
    for k, v in merged.items():
        k2 = k.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
        clean[k2] = v
    n = save_sharded(clean, out)
    copied = []
    for pat in COPY_PATTERNS:
        for f in glob.glob(str(src / pat)):
            shutil.copy2(f, out / os.path.basename(f)); copied.append(os.path.basename(f))
    cfg_p = out / "config.json"
    if cfg_p.exists():
        cfg = json.load(open(cfg_p)); cfg["torch_dtype"] = args.dtype; json.dump(cfg, open(cfg_p, "w"), indent=2)
    print(f"[merge] wrote {len(clean)} tensors into {n} shard(s) at {out}; copied: {copied}")
    # quick sanity: compare against config if available
    if cfg_p.exists():
        c = json.load(open(cfg_p))
        emb = [k for k in clean if k.endswith("embed_tokens.weight")]
        if emb:
            print(f"[merge] embed_tokens shape={tuple(clean[emb[0]].shape)} (config vocab={c.get('vocab_size')}, hidden={c.get('hidden_size')})")


if __name__ == "__main__":
    main()
