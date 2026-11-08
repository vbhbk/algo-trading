# gw_scalp: take, then recycle capacity (session gw)

Engine: logs/gw/scalp/body.py + make.py (builds strategies/gw_scalp*.py with DirectionModel from strategy_v8_r3,
plus r3a ridge models 0.5/1/2 s from logs/gw/models_r3a.json).
- Take: IOC at the touch when |s| >= TAKE_Z (s = p25 + A50*p50 + A1*p1), size min(TAKE_MAX, CAP - exposure(side)).
  Cap: pos + resting orders on that side + IOC in flight <= 95. Resting orders on the take side are cancelled.
- Exit: passive limit at the touch on the reducing side when z = sign(pos)*s <= EXIT_Z (EXIT_Z_HOLD after HOLD_US),
  kept (hysteresis) until z >= HOLD_Z; re-joined if the touch moves. IOC unwind when z < -UNWIND_Z. Optional EXIT_IOC,
  2 s-model target (TARGET_K), SLIP on takes.

## round3_a tuning (pnl k$, all breach 0, q95 95)
| var | config | pnl |
|---|---|---|
| gw_scalp (v0) | p25, Z .6, passive exit when z<=.5 (any) / after 1 s z<=1.5, no hysteresis | 194.5 |
| a | same, exit only z<=0 early, hold 2 s | 194.0 |
| b | IOC exit (z<=0, after 2 s z<=.6) | 162.8 |
| c | passive exit only when z<=-0.2 | 221.4 |
| d | no passive exit (take + IOC unwind only = flip) | 224.2 |
| e | d, Z .5 | 222.7 |
| f | passive exit down to 2 s target (K 200), z<=0 | 203.5 |
| g | c + hysteresis (keep until z>=.6) | 223.3 |
| h | g, exit z<=.2 | 204.8 |
| i | d, Z .8 | 185.7 |
| j | d, take slip 1 tick | 156.0 |
| k | g, TAKE_MAX 95 | 222.8 |
| l | s=p25+.5 p1, Z .9, exit -.2, hold .6 | 232.0 |
| m | s=p25+.5 p1, Z 1.2, exit -.3/.9 | 229.1 |
| **n (chosen)** | s=p25+p1, Z 1.35, exit z<=-.3, keep until z>=.9 | **237.2** |
| o | n without passive exit | 232.8 |
| p | n, Z 1.1 | 211.8 |
| q | n, Z 1.6 | 236.4 |
| r | Z 1.5, exit -.6 | 228.5 |
| s | Z 1.5, exit 0.0 | 199.6 |

Decomposition (decomp.py): v0 passive exits (joined at back of queue, signal decayed) markout +0.30/lot at fill but
-0.81/lot at 1 s -> exits are filled exactly when the move continues (momentum), and the re-take pays spread.
g (exit only when z<=-0.2): passive +0.35 at fill, -0.14 at 1 s, ~0 at 5 s: roughly neutral.

## Final (strategies/gw_scalp.py == gw_scalp_n)
| round | pnl | breach_penalty | fills | abs_inventory_q95 |
|---|---|---|---|---|
| round3_a | 237 158 | 0 | 4118 | 95 |
| round3_b | 232 540 | 0 | 4320 | 95 |
| round2_public | 216 897 | 0 | 8549 | 95 |
Baseline gw_base95: 236 863 / 224 864 / 279 844.

## Conclusions
- Capacity recycling via passive exits does NOT pay in the real sim: back-of-queue exit fills are adversely selected
  (momentum continues after the signal decays); the offline "free passive exit" estimate was far too optimistic.
- Every earlier exit (passive z<=0.5/0.2/0, IOC exit, 2 s target) lost 20-60k; the best engine holds +-95 until a strong
  opposite signal (essentially a flip) with a passive exit only when the prediction is already against the position.
- Adding the 1 s ridge to the take signal (p25 + p1) was the only real gain (+10-13k on r3a vs p25 alone).
- Out of sample: +7.7k vs baseline on round3_b, but -63k on round2_public (no passive market making; r2 regime differs).
