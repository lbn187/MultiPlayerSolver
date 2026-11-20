from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .efg import (CFRTrainer, Strategy, chance_node, collect_infosets,
                 decision_node, expected_utility, frozen_node, terminal_node)
from .games import Game

def find_subgame_entries(game: Game, pub: str, sigma: Strategy
                         ) -> List[dict]:
    out: List[dict] = []

    def down(nd: dict, reach: np.ndarray, ch: float, d_idx: int,
             privs: tuple) -> None:
        if nd.get("pub") == pub:
            out.append({"deal": d_idx, "privates": privs, "node": nd,
                        "chance": ch, "reach": reach.copy()})
            return
        if nd["kind"] == "T":
            return
        if nd["kind"] in ("C", "F"):
            for pr, c in zip(nd["probs"], nd["children"]):
                if pr > 0:
                    down(c, reach, ch * pr, d_idx, privs)
            return
        pl = nd["player"]
        probs = sigma[nd["infoset"]]
        for pr, c in zip(probs, nd["children"]):
            r2 = reach.copy()
            r2[pl] *= pr
            down(c, r2, ch, d_idx, privs)

    for d_idx, (prob, privs, sub) in enumerate(game.deals):
        down(sub, np.ones(game.n_players), prob, d_idx, privs)
    return out

def blueprint_cfvs(game: Game, entries: List[dict], sigma: Strategy
                   ) -> Tuple[Dict[str, float], Dict[str, bool]]:
    n = game.n_players
    num: Dict[str, float] = {}
    den: Dict[str, float] = {}
    raw: Dict[str, List[float]] = {}
    for e in entries:
        ev = expected_utility(e["node"], sigma)
        for j in range(n):
            key = f"G{j}|{e['privates'][j]}"
            w = e["chance"] * float(np.prod(
                [e["reach"][q] for q in range(n) if q != j]))
            num[key] = num.get(key, 0.0) + w * float(ev[j])
            den[key] = den.get(key, 0.0) + w
            raw.setdefault(key, []).append(float(ev[j]))
    cfv, zflag = {}, {}
    for key in num:
        if den[key] > 1e-12:
            cfv[key] = num[key] / den[key]
            zflag[key] = False
        else:
            cfv[key] = float(np.mean(raw[key]))
            zflag[key] = True
    return cfv, zflag

def _copy_subgame(nd: dict, resolver: int, flags: Dict[int, str],
                  cfv_of: Dict[int, float], blueprint: Strategy) -> dict:
    if nd["kind"] == "T":
        pay = nd["payoff"].copy()
        for j, fl in flags.items():
            if fl == "T":
                pay[j] = cfv_of[j]
        return terminal_node(pay)
    if nd["kind"] in ("C", "F"):
        kids = [_copy_subgame(c, resolver, flags, cfv_of, blueprint)
                for c in nd["children"]]
        return chance_node(list(nd["probs"]), kids)
    pl = nd["player"]
    kids = [_copy_subgame(c, resolver, flags, cfv_of, blueprint)
            for c in nd["children"]]
    if pl != resolver and flags.get(pl) == "T":
        return frozen_node(pl, nd["infoset"] + "#bp",
                           blueprint[nd["infoset"]], kids)
    return decision_node(pl, nd["infoset"], nd["actions"], kids)

def build_gadget(game: Game, pub: str, resolver: int, blueprint: Strategy
                 ) -> Optional[dict]:
    n = game.n_players
    entries = find_subgame_entries(game, pub, blueprint)
    if not entries:
        return None
    cfv, zflag = blueprint_cfvs(game, entries, blueprint)
    opps = [q for q in range(n) if q != resolver]
    weights = [e["chance"] * e["reach"][resolver] for e in entries]
    tot = sum(weights)
    if tot <= 1e-12:
        return None
    weights = [w / tot for w in weights]

    kids = []
    for e in entries:
        privs = e["privates"]

        def chain(k: int, flags: Dict[int, str]) -> dict:
            if k == len(opps):
                cfv_of = {j: cfv[f"G{j}|{privs[j]}"] for j in opps}
                return _copy_subgame(e["node"], resolver, flags, cfv_of,
                                     blueprint)
            j = opps[k]
            ch_T = chain(k + 1, {**flags, j: "T"})
            ch_F = chain(k + 1, {**flags, j: "F"})
            return decision_node(j, f"G{j}|{privs[j]}|{pub}", ["T", "F"],
                                 [ch_T, ch_F])

        kids.append(chain(0, {}))
    root = chance_node(weights, kids)
    return {"root": root, "entries": entries, "cfv": cfv, "zflag": zflag,
            "opps": opps, "pub": pub, "resolver": resolver}

def cf_action_values(root: dict, sigma: Strategy, target: str, n: int
                     ) -> Optional[np.ndarray]:
    found: List[Tuple[dict, float]] = []

    def down(nd: dict, pi_minus: float, pl_target: List[int]) -> None:
        if nd["kind"] == "T" or pi_minus <= 0:
            return
        if nd["kind"] in ("C", "F"):
            for pr, c in zip(nd["probs"], nd["children"]):
                down(c, pi_minus * pr, pl_target)
            return
        if nd["infoset"] == target:
            found.append((nd, pi_minus))
            pl_target.append(nd["player"])
            return
        probs = sigma[nd["infoset"]]
        for pr, c in zip(probs, nd["children"]):
            down(c, pi_minus * pr, pl_target)

    pl_holder: List[int] = []
    down(root, 1.0, pl_holder)
    if not found:
        return None
    pl = pl_holder[0]
    na = len(found[0][0]["children"])
    vals = np.zeros(na)
    den = 0.0
    for nd, w in found:
        den += w
        for a, ch in enumerate(nd["children"]):
            vals[a] += w * expected_utility(ch, sigma)[pl]
    if den <= 1e-12:
        return None
    return vals / den

def resolve_subgame(game: Game, pub: str, resolver: int, blueprint: Strategy,
                    iters: int = 4000) -> Optional[dict]:
    gad = build_gadget(game, pub, resolver, blueprint)
    if gad is None:
        return None
    trainer = CFRTrainer(gad["root"], game.n_players)
    trainer.run(iters)
    sg = trainer.average_strategy()

    margins: Dict[str, float] = {}
    ginfo = collect_infosets(gad["root"])
    for I in ginfo:
        if I.startswith("G"):
            v = cf_action_values(gad["root"], sg, I, game.n_players)
            if v is not None:
                margins[I] = float(v[0] - v[1])

    orig_info = collect_infosets(game.root)
    resolved = {k: v.copy() for k, v in blueprint.items()}
    n_spliced = 0
    for I, s in sg.items():
        if I in orig_info and orig_info[I][0] == resolver:
            resolved[I] = s.copy()
            n_spliced += 1
    return {"resolved": resolved, "margins": margins, "gadget": gad,
            "n_spliced": n_spliced, "cfv": gad["cfv"], "zflag": gad["zflag"]}

def vul_paper_est(root: dict, sigma: Strategy, i: int, n: int,
                  restarts: int = 4, rounds: int = 10,
                  seed: int = 0) -> float:
    from .efg import best_response
    rng = np.random.default_rng(seed)
    info = collect_infosets(root)
    opps = [q for q in range(n) if q != i]
    base = float(expected_utility(root, sigma)[i])
    best_min = np.inf
    for r in range(restarts):
        cur = {k: v.copy() for k, v in sigma.items()}
        if r > 0:
            for I, (pl, na) in info.items():
                if pl in opps:
                    v = np.zeros(na)
                    v[rng.integers(0, na)] = 1.0
                    cur[I] = v
        prev = np.inf
        for _ in range(rounds):
            for j in opps:
                _, brj = best_response(root, cur, j, n, minimize=True,
                                       obj_player=i)
                cur.update(brj)
            val = float(expected_utility(root, cur)[i])
            best_min = min(best_min, val)
            if val >= prev - 1e-12:
                break
            prev = val
    return base - best_min
