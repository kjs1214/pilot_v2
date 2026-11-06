"""Deterministic enumeration of ALFWorld game files, mirroring AlfredTWEnv.collect_game_files()."""
from __future__ import annotations
import json
import os
from typing import List, Dict

TASK_TYPES = {1: "pick_and_place_simple", 2: "look_at_obj_in_light", 3: "pick_clean_then_place_in_recep",
              4: "pick_heat_then_place_in_recep", 5: "pick_cool_then_place_in_recep", 6: "pick_two_obj_and_place"}
TYPE2SHORT = {"pick_and_place_simple": "pick", "look_at_obj_in_light": "look", "pick_clean_then_place_in_recep": "clean",
              "pick_heat_then_place_in_recep": "heat", "pick_cool_then_place_in_recep": "cool", "pick_two_obj_and_place": "pick2"}
SPLIT_DIR = {"train": "train", "eval_in_distribution": "valid_seen", "eval_out_of_distribution": "valid_unseen",
             "seen": "valid_seen", "unseen": "valid_unseen"}


def list_games(alfworld_data: str, split: str) -> List[Dict]:
    root = os.path.join(alfworld_data, "json_2.1.1", SPLIT_DIR[split])
    games = []
    for r, dirs, files in os.walk(root):
        if "traj_data.json" in files:
            gf = os.path.join(r, "game.tw-pddl")
            if not os.path.exists(gf):
                continue
            with open(os.path.join(r, "traj_data.json")) as f:
                td = json.load(f)
            tt = td.get("task_type")
            if tt not in TYPE2SHORT:
                continue
            games.append({"game_file": gf, "task_type": tt, "short": TYPE2SHORT[tt],
                          "id": os.path.relpath(r, root)})
    games.sort(key=lambda g: g["id"])
    return games


def select_games(games: List[Dict], per_type: int = 0, max_games: int = 0, offset: int = 0) -> List[Dict]:
    if per_type > 0:
        out, cnt = [], {}
        for g in games:
            if cnt.get(g["short"], 0) < per_type:
                out.append(g); cnt[g["short"]] = cnt.get(g["short"], 0) + 1
        games = out
    games = games[offset:]
    if max_games > 0:
        games = games[:max_games]
    return games
