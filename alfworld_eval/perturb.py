"""
Surface perturbations for ALFWorld that keep the procedure identical.

Two kinds:
  1) Command-wording rules (AgentRM-style, single factor each):
       using    : 'clean/cool/heat X with Y'  -> 'clean/cool/heat X using Y'
       moveto   : 'go to X'                   -> 'move to X'
       fromtake : 'take X from Y'             -> 'from Y take X'
     Forward rewrite is applied to the admissible-action list shown to the model
     (and to the model's own past actions stored in the history); the reverse mapping
     turns the model's output back into the original command before env.step().
  2) Name substitution ('names'): a JSON map {original: replacement} applied to every
     text the model sees (observation, task sentence, admissible list, skill text) and
     reversed on the model's output. Replacements must be unique strings that do not
     collide with other ALFWorld names (checked at load time).
"""
from __future__ import annotations
import json
import re
from typing import Dict, List, Optional, Tuple

COMMAND_RULES = {
    "using": [(r"^(clean|cool|heat) (.+?) with (.+)$", r"\1 \2 using \3")],
    "moveto": [(r"^go to (.+)$", r"move to \1")],
    "fromtake": [(r"^take (.+?) from (.+)$", r"from \2 take \1")],
}
REVERSE_RULES = {
    "using": [(r"^(clean|cool|heat) (.+?) using (.+)$", r"\1 \2 with \3")],
    "moveto": [(r"^move to (.+)$", r"go to \1")],
    "fromtake": [(r"^from (.+?) take (.+)$", r"take \2 from \1")],
}
# What the ORIGINAL surface looks like (used by metrics to detect intrusion of training wording)
ORIGINAL_FORM = {
    "using": re.compile(r"^(clean|cool|heat) .+ with .+$"),
    "moveto": re.compile(r"^go to .+$"),
    "fromtake": re.compile(r"^take .+ from .+$"),
}


def _apply(rules, s: str) -> str:
    for pat, rep in rules:
        s2 = re.sub(pat, rep, s)
        if s2 != s:
            return s2
    return s


class Perturbation:
    def __init__(self, kind: str = "none", names_map_path: Optional[str] = None):
        assert kind in ("none", "using", "moveto", "fromtake", "names"), kind
        self.kind = kind
        self.names: Dict[str, str] = {}
        if kind == "names":
            assert names_map_path, "--names_map is required for kind=names"
            self.names = json.load(open(names_map_path))
            self._check_names()
            # longest-first to avoid partial overlaps, word-boundary matching
            self._fwd = [(re.compile(r"\b" + re.escape(k) + r"\b"), v) for k, v in
                         sorted(self.names.items(), key=lambda kv: -len(kv[0]))]
            self._rev = [(re.compile(r"\b" + re.escape(v) + r"\b"), k) for k, v in
                         sorted(self.names.items(), key=lambda kv: -len(kv[1]))]

    def _check_names(self):
        vals = list(self.names.values())
        assert len(set(vals)) == len(vals), "replacement strings must be unique"
        for k, v in self.names.items():
            for k2 in self.names:
                if k != k2 and (k in k2 or k2 in k):
                    raise ValueError(f"original names overlap: {k!r} / {k2!r} (ambiguous replacement)")
            for v2 in vals:
                if v != v2 and (v in v2 or v2 in v):
                    raise ValueError(f"replacements overlap: {v!r} / {v2!r} (ambiguous reverse mapping)")

    # ---- forward: what the model sees ----
    def text(self, s: str) -> str:
        """Apply to observations / task sentence / skill text (names only; command rules don't appear there)."""
        if self.kind == "names":
            for pat, rep in self._fwd:
                s = pat.sub(rep, s)
        return s

    def action_forward(self, a: str) -> str:
        """Apply to an admissible command (or a past env action) so it looks like the perturbed surface."""
        if self.kind in COMMAND_RULES:
            return _apply(COMMAND_RULES[self.kind], a)
        if self.kind == "names":
            return self.text(a)
        return a

    def admissible(self, cmds: List[str]) -> List[str]:
        return [self.action_forward(c) for c in cmds]

    # ---- reverse: what the env receives ----
    def action_reverse(self, a: str) -> str:
        if self.kind in REVERSE_RULES:
            return _apply(REVERSE_RULES[self.kind], a)
        if self.kind == "names":
            for pat, rep in self._rev:
                a = pat.sub(rep, a)
        return a

    def original_form_regex(self):
        """Regex matching the ORIGINAL surface for intrusion detection (None for 'none')."""
        if self.kind in ORIGINAL_FORM:
            return ORIGINAL_FORM[self.kind]
        if self.kind == "names":
            return re.compile(r"\b(" + "|".join(re.escape(k) for k in self.names) + r")\b")
        return None
