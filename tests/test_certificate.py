"""Checks that the certificate behaves as the theorem says on three-player Kuhn."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

import mpsafe as M

TOL = 0.02


def _setup(iters=150):
    game = M.build_kuhn3()
    trainer = M.CFRTrainer(game.root, game.n_players)
    trainer.run(iters)
    return game, trainer.average_strategy()


def _updates(game, bp, iters=150, pubs=("b", "bf", "kb")):
    for pub in pubs:
        for i in range(game.n_players):
            res = M.resolve_subgame(game, pub, i, bp, iters=iters)
            if res is not None:
                yield pub, i, res["resolved"]


def test_bounds_are_ordered_and_cover():
    game, bp = _setup()
    seen = 0
    for pub, i, resolved in _updates(game, bp):
        cert = M.certificate(game, pub, i, bp, resolved)
        seen += 1
        assert cert["B_cache"] <= cert["B_dir"] + 1e-9, (pub, i, cert)
        assert cert["dgul"] <= cert["B_cache"] + 1e-9, (pub, i, cert)
    assert seen > 0


def test_gate_blocks_every_harmful_update():
    game, bp = _setup()
    for pub, i, resolved in _updates(game, bp):
        cert = M.gate(game, pub, i, bp, resolved, tol=TOL)
        if not cert["accept"]:
            assert cert["B_cache"] > TOL
        if cert["harmful"]:
            assert not cert["accept"], (pub, i, cert)


def test_resolve_changes_only_the_resolver():
    game, bp = _setup()
    for pub, i, resolved in _updates(game, bp, pubs=("b",)):
        if pub != "b":
            continue
        info = M.collect_infosets(game.root)
        for key, strat in resolved.items():
            if np.allclose(strat, bp[key]):
                continue
            owner = info[key][0]
            assert owner == i, (key, owner, i)


if __name__ == "__main__":
    for fn in (test_bounds_are_ordered_and_cover,
               test_gate_blocks_every_harmful_update,
               test_resolve_changes_only_the_resolver):
        fn()
        print("ok", fn.__name__)
