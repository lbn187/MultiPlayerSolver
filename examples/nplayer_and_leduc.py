"""Resolve one subgame of n-player Kuhn and one of three-player Leduc.

Small iteration counts keep the run near a minute; raise ``--iters`` for
tighter blueprints.

    python3 examples/nplayer_and_leduc.py [--kuhn-iters 120] [--leduc-iters 10]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import mpsafe as M

TOL = 0.02


def second_round_public_states(root):
    out = set()

    def walk(nd):
        if nd["kind"] == "T":
            return
        pub = nd.get("pub") or ""
        if "|" in pub:
            out.add(pub)
        for c in nd["children"]:
            walk(c)

    walk(root)
    return sorted(out)


def run_one(game, pub, resolver, bp, iters):
    t0 = time.time()
    res = M.resolve_subgame(game, pub, resolver, bp, iters=iters)
    if res is None:
        return None
    cert = M.gate(game, pub, resolver, bp, res["resolved"], tol=TOL)
    cert["seconds"] = time.time() - t0
    return cert


def tree_size(game):
    dec = [0]
    infosets = set()

    def walk(nd):
        if nd["kind"] == "T":
            return
        if nd["kind"] == "D":
            dec[0] += 1
            infosets.add(nd["infoset"])
        for c in nd["children"]:
            walk(c)

    walk(game.root)
    return dec[0], len(infosets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kuhn-iters", type=int, default=60)
    ap.add_argument("--leduc-iters", type=int, default=5)
    ap.add_argument("--resolve-iters", type=int, default=10)
    args = ap.parse_args()

    print("game                 decisions  information sets")
    for n in (3, 4, 5, 6):
        dec, inf = tree_size(M.build_kuhn(n))
        print("Kuhn n=%d          %11d %17d" % (n, dec, inf))
    dec, inf = tree_size(M.build_leduc3())
    print("Leduc n=3         %11d %17d" % (dec, inf))

    print()
    for n, iters in ((3, args.kuhn_iters), (4, args.kuhn_iters)):
        game = M.build_kuhn(n)
        trainer = M.CFRTrainer(game.root, game.n_players)
        trainer.run(iters)
        bp = trainer.average_strategy()
        cert = run_one(game, "b", 0, bp, args.resolve_iters)
        if cert is None:
            continue
        print("Kuhn n=%d  blueprint gap %.4f  update at 'b': dGap %+.4f  "
              "B_cache %+.4f  %s  (%.1fs)" %
              (n, M.gap(game, bp), cert["dgul"], cert["B_cache"],
               "played" if cert["accept"] else "rejected", cert["seconds"]))

    game = M.build_leduc3()
    t0 = time.time()
    trainer = M.CFRTrainer(game.root, game.n_players)
    trainer.run(args.leduc_iters)
    bp = trainer.average_strategy()
    print("Leduc  blueprint: %d CFR iterations in %.0fs, gap %.4f" %
          (args.leduc_iters, time.time() - t0, M.gap(game, bp)))
    pub = second_round_public_states(game.root)[0]
    cert = run_one(game, pub, 0, bp, args.resolve_iters)
    if cert is not None:
        print("Leduc  update at %r: dGap %+.4f  B_dir %+.4f  B_cache %+.4f  "
              "%s  (%.1fs)" %
              (pub, cert["dgul"], cert["B_dir"], cert["B_cache"],
               "played" if cert["accept"] else "rejected", cert["seconds"]))


if __name__ == "__main__":
    main()
