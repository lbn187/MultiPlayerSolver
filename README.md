# MultiPlayerSolver

Reference implementation of **safe subgame resolving in multiplayer games**.

A solver computes a full-game *blueprint* offline and refines it in real time
inside the reached *subgame*. In two-player zero-sum games this refinement is
safe: it never increases exploitability. With three or more players that
guarantee fails, and a resolve can raise the profile's **best-response gap**
without any warning.

This package implements the pair of pieces that make multiplayer resolving
safe again:

1. a **CFR-D style multiplayer resolving gadget** (`mpsafe.resolve`), and
2. a **post-hoc certificate** (`mpsafe.certificate`) that bounds how much the
   delivered update can raise the best-response gap, from one exact subgame
   best response per opponent.

The certificate turns a resolve into a decision. Run it after the resolve;
if the bound is within a tolerance, play the update, otherwise keep the
blueprint.

## Install

```bash
pip install -r requirements.txt
pip install -e .          # optional, makes `import mpsafe` work anywhere
```

Only `numpy` is required.

## Quickstart

```python
import mpsafe as M

game = M.build_kuhn3()                      # three-player Kuhn poker
trainer = M.CFRTrainer(game.root, game.n_players)
trainer.run(200)
bp = trainer.average_strategy()             # blueprint

res = M.resolve_subgame(game, pub="b", resolver=1, blueprint=bp, iters=200)
cert = M.gate(game, "b", 1, bp, res["resolved"], tol=0.02)
print(cert["dgul"], cert["B_cache"], cert["accept"])
```

`certificate()` returns the measured gap change `dgul`, the blueprint and
resolved best-response gains `A0` and `A1`, the realized gains `gains`, the
margin deficits `M`, and the two bounds:

| quantity | meaning |
|---|---|
| `B_dir` | `max(-g_i, max_j [M_j - g_j])` |
| `B_cache` | baseline-aware form, uses the blueprint's stored best-response gains |
| `covers` | whether `dgul <= B_cache` holds for this update |
| `harmful` | whether `dgul` exceeds the tolerance |

The theorem guarantees `dgul <= B_cache <= B_dir` for any update that changes
only the resolver's strategy inside the subgame.

## Example

`python3 examples/kuhn3_gate.py` resolves every public subgame of three-player
Kuhn poker for every resolver and applies the gate at tolerance 0.02:

```
updates: 33   harmful: 10   blocked by the gate: 10
worst gap increase among played updates: +0.0155
covers (dGap <= B_cache): 33/33
```

Ten of the 33 resolves raise the gap by more than the tolerance, the gate
rejects all ten, and no played update exceeds the tolerance.

## API

| function | purpose |
|---|---|
| `build_kuhn3(deck_size=4)` | three-player Kuhn poker |
| `CFRTrainer(root, n_players).run(iters)` | vanilla CFR, average strategy |
| `resolve_subgame(game, pub, resolver, blueprint, iters)` | multiplayer gadget resolve |
| `player_gains(game, sigma)` | best-response gain of every player |
| `gap(game, sigma)` | best-response gap of a profile |
| `br_margins(game, pub, resolver, bp, resolved)` | BR-certified margin per entry information set |
| `margin_deficits(...)` | mass-weighted margin deficit per opponent |
| `activation_term(game, pub, resolver, bp)` | value the blueprint promises but does not realize |
| `certificate(...)`, `gate(...)` | bounds and the play-or-keep decision |

Games are small on purpose: every quantity is computed by exact full-tree
walks, with no sampling.

## Scope

This repository ships the base solver only: game representation, CFR, the
resolving gadget, best responses, and the certificate. The adversarial search
campaign, the numerical regression suites and the experiment drivers behind
the paper are not included.

The guarantee covers full-depth public subgames, where margins are evaluated
by exact subgame best responses. Depth-limited resolving with learned leaf
values is outside its scope.

## Tests

```bash
python3 tests/test_certificate.py
```

The tests check that `B_cache <= B_dir`, that the certificate covers every
delivered update, that the gate rejects every harmful update, and that a
resolve changes only the resolver's strategy.

## Citation

```bibtex
@article{li2026safe,
  title  = {Safe Subgame Resolving in Multiplayer Games},
  author = {Li, Boning and Huang, Longbo},
  year   = {2026}
}
```

## License

MIT. See `LICENSE`.
