from .efg import (CFRTrainer, Strategy, best_response, collect_infosets,
                  expected_utility, per_player_br_gain, uniform_strategy)
from .games import Game, build_kuhn, build_kuhn3, build_leduc3, eval_pure

try:
    from .resolve import build_gadget, resolve_subgame
    from .certificate import (activation_term, br_margins, certificate, gap,
                              gate, margin_deficits, player_gains)
except ImportError as exc:
    raise ImportError(
        "the core modules of lbn_mpsafe ship as compiled extensions for "
        "Linux x86_64 on CPython 3.10 and later; this platform has no build "
        "(original error: %s)" % exc
    ) from exc

__all__ = [
    "CFRTrainer", "Strategy", "best_response", "collect_infosets",
    "expected_utility", "per_player_br_gain", "uniform_strategy",
    "Game", "build_kuhn", "build_kuhn3", "build_leduc3", "eval_pure",
    "build_gadget", "resolve_subgame",
    "activation_term", "br_margins", "certificate", "gap", "gate",
    "margin_deficits", "player_gains",
]
__version__ = "0.3.0"
