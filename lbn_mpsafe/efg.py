"""Generic extensive-form game (EFG) utilities.

Tree nodes are lightweight dicts:
  chance   : {"kind": "C", "probs": [...], "children": [...]}
  decision : {"kind": "D", "player": i, "infoset": str,
              "actions": [...], "children": [...]}
  frozen   : {"kind": "F", "player": i, "infoset": str,
              "probs": [...], "children": [...]}   # plays a FIXED policy
  terminal : {"kind": "T", "payoff": np.ndarray(n_players)}

A strategy profile is a dict: infoset -> np.ndarray of action probs.
All algorithms are exact full-tree walks (games here are tiny).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

Strategy = Dict[str, np.ndarray]


def chance_node(probs: List[float], children: List[dict]) -> dict:
    return {"kind": "C", "probs": list(probs), "children": children}


def decision_node(player: int, infoset: str, actions: List[str],
                  children: List[dict]) -> dict:
    return {"kind": "D", "player": player, "infoset": infoset,
            "actions": actions, "children": children}


def frozen_node(player: int, infoset: str, probs: np.ndarray,
                children: List[dict]) -> dict:
    return {"kind": "F", "player": player, "infoset": infoset,
            "probs": np.asarray(probs, dtype=float), "children": children}


def terminal_node(payoff: np.ndarray) -> dict:
    return {"kind": "T", "payoff": np.asarray(payoff, dtype=float)}


def collect_infosets(root: dict) -> Dict[str, Tuple[int, int]]:
    """Return infoset -> (player, n_actions) for all decision nodes."""
    out: Dict[str, Tuple[int, int]] = {}
    stack = [root]
    while stack:
        nd = stack.pop()
        if nd["kind"] == "D":
            out[nd["infoset"]] = (nd["player"], len(nd["actions"]))
        if nd["kind"] != "T":
            stack.extend(nd["children"])
    return out


def uniform_strategy(root: dict) -> Strategy:
    return {I: np.ones(na) / na for I, (_, na) in collect_infosets(root).items()}


def expected_utility(root: dict, sigma: Strategy) -> np.ndarray:
    """Exact expected payoff vector under profile sigma."""
    if root["kind"] == "T":
        return root["payoff"]
    if root["kind"] == "C":
        probs = root["probs"]
    elif root["kind"] == "F":
        probs = root["probs"]
    else:
        probs = sigma[root["infoset"]]
    out = None
    for p, ch in zip(probs, root["children"]):
        if p <= 0.0:
            continue
        v = expected_utility(ch, sigma)
        out = p * v if out is None else out + p * v
    if out is None:  # degenerate; take first child
        out = expected_utility(root["children"][0], sigma)
    return out


# ---------------------------------------------------------------------------
# Vanilla CFR (multiplayer, regret matching, simultaneous updates)
# ---------------------------------------------------------------------------
class CFRTrainer:
    """Vanilla CFR on an explicit tree. Works for n>=2 players.

    For n>2 this converges to (approximate) CCE-like play; per-player
    exploitability is measured separately.
    """

    def __init__(self, root: dict, n_players: int) -> None:
        self.root = root
        self.n = n_players
        info = collect_infosets(root)
        self.player_of = {I: p for I, (p, _) in info.items()}
        self.regret: Dict[str, np.ndarray] = {
            I: np.zeros(na) for I, (_, na) in info.items()}
        self.strat_sum: Dict[str, np.ndarray] = {
            I: np.zeros(na) for I, (_, na) in info.items()}

    def _current(self, infoset: str) -> np.ndarray:
        r = np.maximum(self.regret[infoset], 0.0)
        tot = r.sum()
        if tot <= 0.0:
            return np.ones(len(r)) / len(r)
        return r / tot

    def _walk(self, nd: dict, reach: np.ndarray, chance_reach: float
              ) -> np.ndarray:
        if nd["kind"] == "T":
            return nd["payoff"]
        if nd["kind"] in ("C", "F"):
            probs = nd["probs"]
            out = np.zeros(self.n)
            for p, ch in zip(probs, nd["children"]):
                if p <= 0.0:
                    continue
                out += p * self._walk(ch, reach, chance_reach * p)
            return out
        pl, I = nd["player"], nd["infoset"]
        strat = self._current(I)
        child_vals = np.zeros((len(nd["children"]), self.n))
        node_val = np.zeros(self.n)
        for a, ch in enumerate(nd["children"]):
            r2 = reach.copy()
            r2[pl] *= strat[a]
            child_vals[a] = self._walk(ch, r2, chance_reach)
            node_val += strat[a] * child_vals[a]
        cf_reach = chance_reach * np.prod(
            [reach[q] for q in range(self.n) if q != pl])
        self.regret[I] += cf_reach * (child_vals[:, pl] - node_val[pl])
        self.strat_sum[I] += reach[pl] * cf_reach * strat
        return node_val

    def run(self, iters: int) -> None:
        for _ in range(iters):
            self._walk(self.root, np.ones(self.n), 1.0)

    def average_strategy(self) -> Strategy:
        out: Strategy = {}
        for I, s in self.strat_sum.items():
            tot = s.sum()
            out[I] = s / tot if tot > 0 else np.ones(len(s)) / len(s)
        return out


# ---------------------------------------------------------------------------
# Exact best response for one player, others fixed
# ---------------------------------------------------------------------------
def best_response(root: dict, sigma: Strategy, p: int, n_players: int,
                  minimize: bool = False, obj_player: Optional[int] = None
                  ) -> Tuple[float, Strategy]:
    """Exact BR of player p to sigma_{-p} w.r.t. payoff coordinate obj_player
    (defaults to p's own coordinate).

    Returns (root value of coordinate obj under (br_p, sigma_{-p}), br strat).
    If minimize=True, p picks actions to MINIMIZE the objective coordinate
    (used for team-minimisation / paper-style vulnerability estimates).
    Handles sequential own decisions by depth-ordered backward induction
    over p's infosets (perfect recall assumed).
    """
    obj = p if obj_player is None else obj_player
    # Pass 1: annotate histories -> (node, pi_{-p}, own-depth) and group
    # p's decision nodes by infoset with own action depth.
    entries: Dict[str, List[Tuple[dict, float]]] = {}
    depth_of: Dict[str, int] = {}

    def down(nd: dict, pi_others: float, own_depth: int) -> None:
        if nd["kind"] == "T":
            return
        if nd["kind"] in ("C", "F"):
            for pr, ch in zip(nd["probs"], nd["children"]):
                if pr > 0.0:
                    down(ch, pi_others * pr, own_depth)
            return
        if nd["player"] == p:
            entries.setdefault(nd["infoset"], []).append((nd, pi_others))
            d = depth_of.get(nd["infoset"], own_depth)
            depth_of[nd["infoset"]] = max(d, own_depth)
            for ch in nd["children"]:
                down(ch, pi_others, own_depth + 1)
        else:
            probs = sigma[nd["infoset"]]
            for pr, ch in zip(probs, nd["children"]):
                if pr > 0.0:
                    down(ch, pi_others * pr, own_depth)

    down(root, 1.0, 0)
    choice: Dict[str, int] = {}

    def eval_p(nd: dict, forced: Optional[Tuple[dict, int]]) -> float:
        """Objective coordinate value, using `choice` for decided infosets."""
        if nd["kind"] == "T":
            return float(nd["payoff"][obj])
        if nd["kind"] in ("C", "F"):
            return sum(pr * eval_p(ch, forced)
                       for pr, ch in zip(nd["probs"], nd["children"])
                       if pr > 0.0)
        if nd["player"] == p:
            if forced is not None and nd is forced[0]:
                return eval_p(nd["children"][forced[1]], None)
            a = choice[nd["infoset"]]
            return eval_p(nd["children"][a], forced)
        probs = sigma[nd["infoset"]]
        return sum(pr * eval_p(ch, forced)
                   for pr, ch in zip(probs, nd["children"]) if pr > 0.0)

    for I in sorted(entries, key=lambda k: -depth_of[k]):
        nodes = entries[I]
        na = len(nodes[0][0]["children"])
        vals = np.zeros(na)
        for a in range(na):
            for nd, pi_o in nodes:
                vals[a] += pi_o * eval_p(nd["children"][a], None)
        choice[I] = int(np.argmin(vals) if minimize else np.argmax(vals))

    br: Strategy = {}
    for I, a in choice.items():
        na = len(entries[I][0][0]["children"])
        v = np.zeros(na)
        v[a] = 1.0
        br[I] = v
    full = dict(sigma)
    full.update(br)
    val = expected_utility(root, full)[obj]
    return float(val), br


def per_player_br_gain(root: dict, sigma: Strategy, n_players: int
                       ) -> np.ndarray:
    """Task metric: Vul_i = max_{s_i'} u_i(s_i', sigma_{-i}) - u_i(sigma)."""
    base = expected_utility(root, sigma)
    out = np.zeros(n_players)
    for p in range(n_players):
        v, _ = best_response(root, sigma, p, n_players)
        out[p] = v - base[p]
    return out
