"""Game builders: 3-player Kuhn poker and a synthetic polymatrix+perturbation
family. Both are returned as a Game object exposing per-deal subtrees so that
subgames (identified by PUBLIC action history) can be extracted for resolving.

Infoset keys: "P{player}|{private}|{public_history}".
Every non-terminal node carries a "pub" field (public history string).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .efg import chance_node, decision_node, terminal_node


@dataclass
class Game:
    name: str
    n_players: int
    root: dict
    # deals: list of (prob, privates tuple (one label per player), deal_root)
    deals: List[Tuple[float, tuple, dict]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 3-player Kuhn poker
# ---------------------------------------------------------------------------
# Betting grammar (one bet max, 1-chip ante, 1-chip bet):
#   ""    -> P0 {k,b}
#   "k"   -> P1 {k,b}
#   "kk"  -> P2 {k,b};  "kkk" showdown
#   "b"   -> P1 {f,c} -> P2 {f,c} -> terminal
#   "kb"  -> P2 {f,c} -> P0 {f,c} -> terminal
#   "kkb" -> P0 {f,c} -> P1 {f,c} -> terminal
_KUHN_NEXT: Dict[str, Tuple[int, List[str]]] = {
    "": (0, ["k", "b"]),
    "k": (1, ["k", "b"]),
    "kk": (2, ["k", "b"]),
    "b": (1, ["f", "c"]),
    "bf": (2, ["f", "c"]),
    "bc": (2, ["f", "c"]),
    "kb": (2, ["f", "c"]),
    "kbf": (0, ["f", "c"]),
    "kbc": (0, ["f", "c"]),
    "kkb": (0, ["f", "c"]),
    "kkbf": (1, ["f", "c"]),
    "kkbc": (1, ["f", "c"]),
}
_KUHN_TERMINAL = {"kkk", "bff", "bfc", "bcf", "bcc", "kbff", "kbfc", "kbcf",
                  "kbcc", "kkbff", "kkbfc", "kkbcf", "kkbcc"}

# who played b / c / f in a completed history
_KUHN_ACTOR_SEQ: Dict[str, List[int]] = {}


def _actor_seq(hist: str) -> List[int]:
    """Actor of each position in history string."""
    seq: List[int] = []
    h = ""
    for ch in hist:
        seq.append(_KUHN_NEXT[h][0])
        h += ch
    return seq


def _kuhn_payoff(hist: str, cards: Tuple[int, int, int]) -> np.ndarray:
    actors = _actor_seq(hist)
    contrib = np.ones(3)  # antes
    bettor = None
    active = {0, 1, 2}
    for pos, ch in enumerate(hist):
        p = actors[pos]
        if ch == "b":
            bettor = p
            contrib[p] += 1.0
        elif ch == "c":
            contrib[p] += 1.0
        elif ch == "f":
            active.discard(p)
    if bettor is not None:
        # only bettor + callers contest the pot
        pass  # folders already removed; checkers before the bet never fold?
    # Players who neither folded nor (when a bet occurred) called are the
    # bettor themselves; grammar guarantees everyone facing a bet acts f/c.
    pot = contrib.sum()
    contenders = sorted(active)
    winner = max(contenders, key=lambda p: cards[p])
    pay = -contrib
    pay[winner] += pot
    return pay


def _kuhn_subtree(hist: str, cards: Tuple[int, int, int]) -> dict:
    if hist in _KUHN_TERMINAL:
        return terminal_node(_kuhn_payoff(hist, cards))
    player, acts = _KUHN_NEXT[hist]
    children = [_kuhn_subtree(hist + a, cards) for a in acts]
    nd = decision_node(player, f"P{player}|{cards[player]}|{hist}",
                       acts, children)
    nd["pub"] = hist
    return nd


def build_kuhn3(deck_size: int = 4) -> Game:
    deck = list(range(deck_size))
    deals = list(itertools.permutations(deck, 3))
    prob = 1.0 / len(deals)
    deal_entries = []
    children = []
    for cards in deals:
        sub = _kuhn_subtree("", cards)
        deal_entries.append((prob, cards, sub))
        children.append(sub)
    root = chance_node([prob] * len(deals), children)
    root["pub"] = "<root>"
    return Game(f"kuhn3_d{deck_size}", 3, root, deal_entries)


def kuhn_infoset_keys(deck_size: int, player: int) -> List[str]:
    states = [h for h, (p, _) in _KUHN_NEXT.items() if p == player]
    return [f"P{player}|{c}|{h}" for c in range(deck_size) for h in states]


# ---------------------------------------------------------------------------
# Synthetic family: EXACTLY constant-sum polymatrix at lam=0, plus lam *
# non-polymatrix perturbation.
#   chance: types t_i ~ U{0,1} iid AND a PUBLIC signal X in {A,B} uniform
#   round: P0, P1, P2 act in {0,1} sequentially WITHOUT observing each other
#          (each sees only own type + X) -> effectively simultaneous.
# Payoff: u_i(z) = sum_{j != i} M[X][(i,j)][(t_i,a_i),(t_j,a_j)]
#                  + lam * T[i, t, X, a]
# Base part is pairwise zero-sum per signal (M[X][(i,j)] = -M[X][(j,i)]^T).
# Because actions condition only on own type and the CHANCE signal X, the
# induced normal form is exactly CSP at lam=0 (expectation over independent
# chance factorizes pairwise); the LP verifies delta(0)=0.
# The subgame used for resolving is the public-chance subtree X=B, mirroring
# resolving after a public board card in poker.
# ---------------------------------------------------------------------------
def build_synthetic(lam: float, seed: int) -> Game:
    rng = np.random.default_rng(1000 + seed)
    # per-signal pairwise zero-sum matrices over local states s_i=(t_i,a_i)
    M = {}
    for X in ("A", "B"):
        for i in range(3):
            for j in range(i + 1, 3):
                A = rng.uniform(-1.0, 1.0, size=(4, 4))
                M[(X, i, j)] = A
                M[(X, j, i)] = -A.T
    # non-polymatrix perturbation over full outcome (i, t0,t1,t2, X, a0,a1,a2)
    T = rng.uniform(-1.0, 1.0, size=(3, 2, 2, 2, 2, 2, 2, 2))

    def sidx(t: int, a: int) -> int:
        return 2 * t + a

    def payoff(ts: Tuple[int, int, int], X: str,
               acts: Tuple[int, int, int]) -> np.ndarray:
        xi = 0 if X == "A" else 1
        out = np.zeros(3)
        for i in range(3):
            for j in range(3):
                if j == i:
                    continue
                out[i] += M[(X, i, j)][sidx(ts[i], acts[i]),
                                       sidx(ts[j], acts[j])]
            out[i] += lam * T[(i,) + ts + (xi,) + acts]
        return out

    deal_entries = []
    children = []
    combos = list(itertools.product([0, 1], repeat=3))
    prob = 1.0 / (len(combos) * 2)
    for ts in combos:
        for X in ("A", "B"):
            def p2_node(a0: int, a1: int) -> dict:
                kids = [terminal_node(payoff(ts, X, (a0, a1, a2)))
                        for a2 in (0, 1)]
                nd = decision_node(2, f"P2|{ts[2]}|{X}", ["0", "1"], kids)
                nd["pub"] = X
                return nd

            def p1_node(a0: int) -> dict:
                kids = [p2_node(a0, a1) for a1 in (0, 1)]
                nd = decision_node(1, f"P1|{ts[1]}|{X}", ["0", "1"], kids)
                nd["pub"] = X
                return nd

            nd0 = decision_node(0, f"P0|{ts[0]}|{X}", ["0", "1"],
                                [p1_node(0), p1_node(1)])
            nd0["pub"] = X
            deal_entries.append((prob, ts, nd0))
            children.append(nd0)
    root = chance_node([prob] * len(children), children)
    root["pub"] = "<root>"
    g = Game(f"synth_lam{lam}_s{seed}", 3, root, deal_entries)
    g.meta = {"lam": lam, "seed": seed, "T_maxabs": float(np.abs(T).max())}
    return g


# ---------------------------------------------------------------------------
# Pure-strategy evaluation (needed for delta LP)
# ---------------------------------------------------------------------------
def eval_pure(game: Game, pure: List[Dict[str, int]]) -> np.ndarray:
    """Expected payoff of a pure profile (infoset -> action idx per player)."""
    out = np.zeros(game.n_players)
    for prob, _, sub in game.deals:
        nd = sub
        while nd["kind"] != "T":
            a = pure[nd["player"]][nd["infoset"]]
            nd = nd["children"][a]
        out += prob * nd["payoff"]
    return out

# ---------------------------------------------------------------------------
# n-player Kuhn poker (n = 3 .. 6) and three-player Leduc poker
# ---------------------------------------------------------------------------
# Kuhn poker, n players: a deck of n+1 distinct cards gives one private card
# to each player, every player antes 1, and one betting round allows at most
# one bet of 1.  At n = 3 this matches build_kuhn3.
#
# Leduc poker, three players: a six-card deck holds two copies of each of
# three ranks.  Every player antes 1 and receives one private card, a public
# card is dealt from the three remaining cards, and two betting rounds use
# bet sizes 2 and 4 with at most one raise per round.  At showdown a pair
# with the public card beats any other hand, a higher private rank beats a
# lower one, and equal hands split the pot.

def _payoff_fold(contrib, folded, n):
    alive = [p for p in range(n) if p not in folded]
    pay = np.array([-float(c) for c in contrib])
    pay[alive[0]] += float(sum(contrib))
    return pay


def _payoff_kuhn(cards, contrib, folded, n):
    alive = [p for p in range(n) if p not in folded]
    if len(alive) == 1:
        return _payoff_fold(contrib, folded, n)
    best = max(alive, key=lambda p: cards[p])
    pay = np.array([-float(c) for c in contrib])
    pay[best] += float(sum(contrib))
    return pay


def _payoff_leduc(cards, board, contrib, folded, n):
    alive = [p for p in range(n) if p not in folded]
    if len(alive) == 1:
        return _payoff_fold(contrib, folded, n)
    strength = [(1, cards[p]) if cards[p] == board else (0, cards[p])
                for p in range(n)]
    top = max(strength[p] for p in alive)
    winners = [p for p in alive if strength[p] == top]
    pay = np.array([-float(c) for c in contrib])
    pot = float(sum(contrib))
    for p in winners:
        pay[p] += pot / len(winners)
    return pay


class _BettingGame:
    """Shared betting engine: seat-ordered action, one raise per round.

    ``pending`` lists the players who still owe an action in this round, in
    the order they act.  A bet or raise puts every other live player back on
    the list, starting from the seat after the bettor, which is the standard
    poker order.
    """

    def __init__(self, n, cards, bet_sizes, max_bets, showdown, board=None,
                 has_board=False):
        self.has_board = has_board
        self.n = n
        self.cards = cards
        self.bet_sizes = bet_sizes
        self.max_bets = max_bets
        self.showdown = showdown
        self.board = board
        self.infosets = set()

    def _terminal(self, contrib, folded):
        pay = self.showdown(self.cards, self.board, contrib, folded)
        return terminal_node(pay)

    def node(self, contrib, folded, pending, bets, rnd, hist):
        n = self.n
        actor = pending[0]
        key = "P%d|%d|%s" % (actor, self.cards[actor], hist)
        self.infosets.add(key)
        to_call = max(contrib) - contrib[actor]
        size = self.bet_sizes[rnd]
        if to_call > 0:
            acts = ["f", "c"] + (["r"] if bets < self.max_bets else [])
        else:
            acts = ["k"] + (["b"] if bets < self.max_bets else [])
        kids = []
        def advance(c2, f2, p2, b2, h2):
            live2 = [p for p in range(n) if p not in f2]
            if len(live2) == 1:
                return self._terminal(c2, f2)
            if p2:
                return self.node(c2, f2, p2, b2, rnd, h2)
            if rnd + 1 < len(self.bet_sizes):
                return self._next_round(c2, f2, h2 + "|", rnd + 1)
            return self._terminal(c2, f2)

        for a in acts:
            c2 = list(contrib)
            f2 = set(folded)
            h2 = hist + a
            if a == "f":
                f2.add(actor)
                kids.append(advance(c2, f2, pending[1:], bets, h2))
                continue
            if a in ("b", "r"):
                c2[actor] = max(c2) + size
                live = [p for p in range(n) if p not in f2 and p != actor]
                p2 = sorted(live, key=lambda p: (p - actor) % n)
                kids.append(advance(c2, f2, p2, bets + 1, h2))
                continue
            if a == "c":
                c2[actor] = max(c2)
            kids.append(advance(c2, f2, pending[1:], bets, h2))
        nd = decision_node(actor, key, acts, kids)
        nd["pub"] = hist
        return nd

    def _first_actor(self, folded):
        return min(p for p in range(self.n) if p not in folded)

    def _next_round(self, contrib, folded, hist, rnd):
        first = self._first_actor(folded)
        pending = [p for p in range(self.n) if p not in folded and p >= first]
        pending += [p for p in range(self.n) if p not in folded and p < first]
        if not self.has_board:
            return self.node(contrib, folded, pending, 0, rnd, hist)
        deal = self._board_chance()
        probs = [w for w, _ in deal]
        kids = []
        for _, b in deal:
            g = _BettingGame(self.n, self.cards, self.bet_sizes,
                             self.max_bets, self.showdown, b, True)
            kids.append(g.node(contrib, folded, pending, 0, rnd, hist + str(b)))
        return chance_node(probs, kids)

    def _board_chance(self):
        rest = [i for i in range(6) if i not in self.cards]
        counts: Dict[int, int] = {}
        for i in rest:
            counts[i // 2] = counts.get(i // 2, 0) + 1
        return [(c / len(rest), b) for b, c in sorted(counts.items())]


def _pending_all(n, folded):
    first = min(p for p in range(n) if p not in folded)
    out = [p for p in range(n) if p not in folded and p >= first]
    out += [p for p in range(n) if p not in folded and p < first]
    return out


def build_kuhn(n: int = 3, deck_size: Optional[int] = None) -> Game:
    """n-player Kuhn poker: deck n+1, ante 1, one round, at most one bet of 1."""
    if deck_size is None:
        deck_size = n + 1
    def showdown(cards, board, contrib, folded):
        return _payoff_kuhn(cards, contrib, folded, n)
    deals = list(itertools.permutations(range(deck_size), n))
    prob = 1.0 / len(deals)
    entries, children = [], []
    for cards in deals:
        g = _BettingGame(n, list(cards), [1], 1, showdown)
        sub = g.node([1] * n, set(), _pending_all(n, set()), 0, 0, "")
        entries.append((prob, tuple(cards), sub))
        children.append(sub)
    root = chance_node([prob] * len(deals), children)
    root["pub"] = "<root>"
    return Game("kuhn%d_d%d" % (n, deck_size), n, root, entries)


def build_leduc3() -> Game:
    """Three-player Leduc: deck of six cards (three ranks, two copies each)."""
    def showdown(cards, board, contrib, folded):
        return _payoff_leduc(cards, board, contrib, folded, 3)
    deals = list(itertools.permutations(range(6), 3))
    prob = 1.0 / len(deals)
    entries, children = [], []
    for cards in deals:
        g = _BettingGame(3, list(cards), [2, 4], 2, showdown, None, True)
        sub = g.node([1, 1, 1], set(), _pending_all(3, set()), 0, 0, "")
        entries.append((prob, tuple(cards), sub))
        children.append(sub)
    root = chance_node([prob] * len(deals), children)
    root["pub"] = "<root>"
    return Game("leduc3", 3, root, entries)
