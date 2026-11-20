from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import lbn_mpsafe as M

TOL = 0.02

def public_states(root):
    out = set()

    def walk(nd):
        if nd["kind"] == "T":
            return
        pub = nd.get("pub")
        if pub and pub != "<root>":
            out.add(pub)
        for c in nd["children"]:
            walk(c)

    walk(root)
    return sorted(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200,
                    help="CFR iterations for the blueprint and the gadget")
    args = ap.parse_args()

    game = M.build_kuhn3()
    trainer = M.CFRTrainer(game.root, game.n_players)
    trainer.run(args.iters)
    bp = trainer.average_strategy()

    print("blueprint gap: %.4f" % M.gap(game, bp))
    print("%-5s %-8s %9s %9s %9s %7s" %
          ("pub", "resolver", "dGap", "B_dir", "B_cache", "played"))
    rows = []
    for pub in public_states(game.root):
        for i in range(game.n_players):
            res = M.resolve_subgame(game, pub, i, bp, iters=args.iters)
            if res is None:
                continue
            cert = M.gate(game, pub, i, bp, res["resolved"], tol=TOL)
            rows.append(cert)
            print("%-5s %-8d %+9.4f %+9.4f %+9.4f %7s" %
                  (pub, i, cert["dgul"], cert["B_dir"], cert["B_cache"],
                   "update" if cert["accept"] else "blueprint"))

    harmful = [c for c in rows if c["harmful"]]
    blocked = [c for c in harmful if not c["accept"]]
    played = [c for c in rows if c["accept"]]
    print()
    print("updates: %d   harmful: %d   blocked by the gate: %d" %
          (len(rows), len(harmful), len(blocked)))
    print("worst gap increase among played updates: %+.4f" %
          (max([c["dgul"] for c in played], default=0.0)))
    print("covers (dGap <= B_cache): %d/%d" %
          (sum(1 for c in rows if c["covers"]), len(rows)))

if __name__ == "__main__":
    main()
