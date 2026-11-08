# gw_v8exec: v8 engine, taking/execution/capacity changes (tuned on round3_a only)

Engine: strategies/gw_v8exec.py = gw_base95 + parameters (defaults reproduce base95):
TAKE_MIN (partial takes), TAKE_BLEND (0.25/0.5 s ridge blend, 0.5 s weights from models_r3a.json), THROUGH_Z/THROUGH_TICKS
(IOC priced through the touch), SIZE_FULL_Z (size ~ |p|), HOLD_US/EXIT_Z (target 0 + looser unwind after the hold),
EXIT_TICKS/EXIT_FRAC/EXIT_TTL_US (passive profit-taking LIMIT after each take, maker shrunk to make room),
MAKER_CAP (passive quotes may only move position to +-MAKER_CAP), TAKE_LOTS up to 190 (one IOC for a full reversal).
Variants: python logs/gw/v8exec/mk.py TAG K=V ... -> strategies/gw_v8exec_TAG.py; run+decomp: logs/gw/v8exec/btl.sh.
Diagnostics: inv.py (time at |pos| buckets), split.py (markout/lot by add/reduce/flip), block.py (signal vs position).

## round3_a results (base95 = 236 863, all pen 0, q95 95, q50 95)
| variant | params | pnl |
|---|---|---|
| p20 | TAKE_MIN 20 | 237 583 |
| tm40 | p20 + TARGET_MAX 40 | 231 001 |
| th09 | p20 + IOC 1 tick through at abs(p)>=0.9 | 234 413 |
| h2e3 | p20 + HOLD 2 s, EXIT_Z 0.3 | 236 377 |
| h2em1 | p20 + HOLD 2 s, EXIT_Z -0.1 (timed taker exit) | 225 420 |
| sz12 | p20 + size ~ abs(p)/1.2 | 233 020 |
| bl5 | p20 + blend 0.5, Z 0.7 | 237 454 |
| z5c50 | p20 + Z 0.5, cooldown 50 ms | 237 846 |
| ex1 | p20 + passive exit at take px +1 (TTL 3 s) | 228 838 |
| ex3 | p20 + passive exit +3 (TTL 5 s) | 229 030 |
| mc30 | p20 + MAKER_CAP 30 | 233 049 |
| z4 | p20 + Z 0.4, cd 50 ms | 220 598 |
| z8 | p20 + Z 0.8 | 222 999 |
| tl190 | p20 + TAKE_LOTS 190 | 237 334 |
| **c1** | TAKE_MIN 20, TAKE_LOTS 190, blend 0.5, Z 0.7 | **241 412** |
| c2 | c1 with Z 0.8 | 236 813 |
| c3 | c1 with blend 1.0, Z 0.8 | 236 659 |

## Final check (chosen = strategies/gw_v8exec_c1.py)
| round | pnl | breach_penalty | fills | abs_inventory_q95 |
|---|---|---|---|---|
| round3_a | 241 412 | 0 | 3899 | 95 |
| round3_b | 232 072 | 0 | 3931 | 95 |
| round2_public | 274 390 | 0 | 8238 | 95 |
base95: 236 863 / 224 864 / 279 844 -> c1 +4.5k / +7.2k / -5.5k: noise-level (pnl se ~18-25k).

## Findings
- Execution space is flat: 17 variants within -16k..+5k of base on r3a; only TAKE_Z matters (0.4 and 0.8 both -15k).
- Pinning at +-95 is NOT caused by the 2 s target or passive refills: |p|>=0.6 episodes arrive every ~0.4 s, so takes
  themselves keep capacity in use (pos at cap 62-68% of time in every variant, incl. TARGET_MAX 40 / timed exits).
  At strong signals (abs p>=1.2) 66% of episodes start already at cap in the signal direction, i.e. the position already rides the move.
- Recycling/exits lose: timed taker exit -11k, passive profit-taking exits (+1/+3 ticks) -8k (they fill but displace
  maker fills and exit before the move ends; 2 s moves after strong signals are 1.5-3.6 ticks).
- Passive fills that flip the position have -0.2 tick/lot markout at 1-5 s, but capping them (MAKER_CAP 30) halves
  passive volume and costs ~4k net.
