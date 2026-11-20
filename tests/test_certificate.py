"""Checks that the certificate behaves as the theorem says on three-player Kuhn."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

import lbn_mpsafe as M

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




def _tree_size(game):
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


def test_game_family_sizes():
    assert _tree_size(M.build_kuhn(3)) == (288, 48)
    assert set(M.collect_infosets(M.build_kuhn(3).root)) == \
        set(M.collect_infosets(M.build_kuhn3().root))
    assert _tree_size(M.build_kuhn(4)) == (3840, 160)
    assert _tree_size(M.build_kuhn(5)) == (57600, 480)
    assert _tree_size(M.build_kuhn(6)) == (967680, 1344)
    assert _tree_size(M.build_leduc3()) == (135288, 8406)


def test_leduc_certificate_covers():
    game = M.build_leduc3()
    bp = M.uniform_strategy(game.root)
    pubs = set()

    def walk(nd):
        if nd["kind"] == "T":
            return
        if "|" in (nd.get("pub") or ""):
            pubs.add(nd["pub"])
        for c in nd["children"]:
            walk(c)

    walk(game.root)
    pub = sorted(pubs)[0]
    res = M.resolve_subgame(game, pub, 0, bp, iters=5)
    assert res is not None
    cert = M.certificate(game, pub, 0, bp, res["resolved"])
    assert cert["B_cache"] <= cert["B_dir"] + 1e-9
    assert cert["covers"]


if __name__ == "__main__":
    for fn in (test_bounds_are_ordered_and_cover,
               test_gate_blocks_every_harmful_update,
               test_resolve_changes_only_the_resolver,
               test_game_family_sizes,
               test_leduc_certificate_covers):
        fn()
        print("ok", fn.__name__)
