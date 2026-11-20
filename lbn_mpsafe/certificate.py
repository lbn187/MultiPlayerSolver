from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .efg import (Strategy, best_response, chance_node, collect_infosets,
                  decision_node, expected_utility, frozen_node)
from .games import Game
from .resolve import blueprint_cfvs, find_subgame_entries

def player_gains(game: Game, sigma: Strategy) -> np.ndarray:
    n = game.n_players
    return np.array([best_response(game.root, sigma, p, n)[0]
                     - expected_utility(game.root, sigma)[p]
                     for p in range(n)])

def gap(game: Game, sigma: Strategy) -> float:
    return float(player_gains(game, sigma).max())

def subgame_infosets(game: Game, pub: str, player: int) -> List[str]:
    info = collect_infosets(game.root)
    uni = {I: np.ones(na) / na for I, (_, na) in info.items()}
    keys = set()
    for e in find_subgame_entries(game, pub, uni):
        for I, (pl, _) in collect_infosets(e["node"]).items():
            if pl == player:
                keys.add(I)
    return sorted(keys)

def br_margins(game: Game, pub: str, resolver: int, bp: Strategy,
               resolved: Strategy) -> Dict[str, float]:
    n = game.n_players
    entries = find_subgame_entries(game, pub, bp)
    if not entries:
        return {}
    cfv, _ = blueprint_cfvs(game, entries, bp)
    groups: Dict[str, List[Tuple[dict, float]]] = {}
    for e in entries:
        for j in range(n):
            if j == resolver:
                continue
            key = "G%d|%s" % (j, e["privates"][j])
            w = e["chance"] * float(np.prod(
                [e["reach"][q] for q in range(n) if q != j]))
            groups.setdefault(key, []).append((e["node"], w))
    out: Dict[str, float] = {}
    for key, occ in sorted(groups.items()):
        j = int(key[1:].split("|")[0])
        tot = sum(w for _, w in occ)
        probs = ([w / tot for _, w in occ] if tot > 1e-12
                 else [1.0 / len(occ)] * len(occ))
        root = chance_node(probs, [nd for nd, _ in occ])
        sbv, _ = best_response(root, resolved, j, n)
        out[key] = float(cfv[key] - sbv)
    return out

def margin_deficits(game: Game, pub: str, resolver: int, bp: Strategy,
                    resolved: Strategy) -> Dict[str, float]:
    n = game.n_players
    entries = find_subgame_entries(game, pub, bp)
    m_hat = br_margins(game, pub, resolver, bp, resolved)
    mass: Dict[str, float] = {}
    for e in entries:
        for j in range(n):
            if j == resolver:
                continue
            key = "G%d|%s" % (j, e["privates"][j])
            mass[key] = mass.get(key, 0.0) + e["chance"] * float(np.prod(
                [e["reach"][q] for q in range(n) if q != j]))
    out: Dict[int, float] = {}
    for key, m in m_hat.items():
        j = int(key[1:].split("|")[0])
        out[j] = out.get(j, 0.0) + mass[key] * max(0.0, -m)
    return {str(j): float(v) for j, v in sorted(out.items())}

def activation_term(game: Game, pub: str, resolver: int,
                    bp: Strategy) -> float:
    n = game.n_players
    s_set = set(subgame_infosets(game, pub, resolver))
    frozen_root = _freeze_outside(game.root, resolver, s_set, bp)
    val, _ = best_response(frozen_root, bp, resolver, n)
    return float(val - expected_utility(game.root, bp)[resolver])

def _freeze_outside(nd: dict, player: int, s_set: set,
                    sigma: Strategy) -> dict:
    if nd["kind"] == "T":
        return nd
    kids = [_freeze_outside(c, player, s_set, sigma) for c in nd["children"]]
    if nd["kind"] in ("C", "F"):
        out = dict(nd)
        out["children"] = kids
        return out
    if nd["player"] == player and nd["infoset"] not in s_set:
        return frozen_node(player, nd["infoset"] + "#frozen",
                           sigma[nd["infoset"]], kids)
    return decision_node(nd["player"], nd["infoset"], nd["actions"], kids)

def certificate(game: Game, pub: str, resolver: int, bp: Strategy,
                resolved: Strategy) -> dict:
    a0 = player_gains(game, bp)
    a1 = player_gains(game, resolved)
    u0 = expected_utility(game.root, bp)
    u1 = expected_utility(game.root, resolved)
    g = u1 - u0
    m_j = margin_deficits(game, pub, resolver, bp, resolved)
    m = {int(k): v for k, v in m_j.items()}
    others = [p for p in range(game.n_players) if p != resolver]
    b_dir = max([-g[resolver]] + [m[j] - g[j] for j in others])
    b_cache = max([a0[resolver] - g[resolver]]
                  + [a0[j] + m[j] - g[j] for j in others]) - float(a0.max())
    dvul = float(a1.max() - a0.max())
    return {"dgul": dvul, "gap_bp": float(a0.max()),
            "gap_resolved": float(a1.max()),
            "A0": a0.tolist(), "A1": a1.tolist(), "gains": g.tolist(),
            "M": {str(k): v for k, v in m.items()},
            "B_dir": float(b_dir), "B_cache": float(b_cache),
            "covers": bool(dvul <= b_cache + 1e-9),
            "harmful": bool(dvul > 0.02)}

def gate(game: Game, pub: str, resolver: int, bp: Strategy,
         resolved: Strategy, tol: float = 0.02) -> dict:
    cert = certificate(game, pub, resolver, bp, resolved)
    cert["accept"] = bool(cert["B_cache"] <= tol)
    cert["played"] = resolved if cert["accept"] else bp
    return cert
