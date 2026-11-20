"""Safe subgame resolving in multiplayer games.

A compact reference implementation of the paper's core pipeline:

    blueprint -> multiplayer gadget resolve -> post-hoc certificate -> gate

Only the base solver is included: extensive-form game representation, CFR,
the CFR-D style multiplayer resolving gadget, best responses, and the
certificate of Theorem 1.  The adversarial search campaign and the
experimental drivers of the paper are not part of this package.
"""
from .efg import (CFRTrainer, Strategy, best_response, collect_infosets,
                  expected_utility, per_player_br_gain, uniform_strategy)
from .games import Game, build_kuhn, build_kuhn3, build_leduc3, eval_pure
from .resolve import build_gadget, resolve_subgame
from .certificate import (activation_term, br_margins, certificate, gap, gate,
                          margin_deficits, player_gains)

__all__ = [
    "CFRTrainer", "Strategy", "best_response", "collect_infosets",
    "expected_utility", "per_player_br_gain", "uniform_strategy",
    "Game", "build_kuhn", "build_kuhn3", "build_leduc3", "eval_pure",
    "build_gadget", "resolve_subgame",
    "activation_term", "br_margins", "certificate", "gap", "gate",
    "margin_deficits", "player_gains",
]
__version__ = "0.2.0"
