# Brief for strategy agents (session gw)
Repo /home/user/algo-trading. Read AGENTS.md rules (one .py, one BaseStrategy subclass, no ctor args, allowed imports
only: stdlib/numpy/pandas/scipy/sklearn/simulator.models/simulator.strategy; forbidden names: os sys io socket subprocess
threading importlib inspect gc open eval exec getattr type, dunders, str.format). Never edit simulator/ or explorer/.
Only create files strategies/gw_<name>*.py and logs/gw/<name>/. Do NOT commit.
Inventory must never exceed |100| (we keep 95): |position + own resting orders on a side + IOC in flight| <= 95 always.

## Backtest
logs/gw/bt.sh strategies/gw_X.py round3_a        # prints pnl, se, penalty, fills, q95, q50 (~1 min per 3000 s file)
CPU: 4 cores shared by 3 agents -> run AT MOST 1 backtest at a time (one file per bt.sh call).
Protocol: tune ONLY on round3_a. Then ONE final check of the chosen variant on round3_b and round2_public.
Report: round, pnl, breach_penalty, fills, abs_inventory_q95 for round3_a, round3_b, round2_public.

## Baseline (strategies/gw_base95.py = strategy_v8_r3 with TAKE_LOTS 95, models fit on round3_a)
round3_a 236 863 | round3_b 224 864 | round2_public 279 844. Must beat this.

## What the data analysis found (logs/gw/an*.py, logs/gw/decomp.py)
- Exchange: 5 ms order latency; PnL = cash + mark-out of inventory (IOC-liquidation against book). Spread 1 tick ~82% of time.
- Edge is SHORT-TERM only: ridge models for horizons 0.25 s / 0.5 s / 1 s generalise (R3b R^2 .25/.22/.15), 2 s weakly,
  >=5 s do not generalise (R^2 <= 0). Mean-reversion overlays (mid-EMA) and vol-scaled thresholds did NOT help.
- Baseline PnL decomposition on round3_b: TAKER (IOC) fills 200k lots: markout -0.49 tick/lot at fill, +1.02 at 1 s,
  peak +1.27 at 2 s, flat after (~+1.0..1.1 at 5-60 s). PASSIVE fills 161k lots: +0.37/lot at fill, only +0.1/lot at 1 s
  (adverse selection). => almost all money is from IOC taking on the 0.25 s prediction; holding > 2-3 s adds nothing.
- Baseline inventory is pinned at +-95 (q50 = 95): capacity is mostly blocked, few new takes possible.
- Pure IOC "flip" (hold +-95 by sign of p25 with hysteresis 0.6, strategies/gw_flip.py): r3a 200k, r3b 204k, r2 194k
  (worse: pays spread on both legs). Adding passive pre-positioning (gw_flipmk.py): r3b 223k, r2 178k.
- v8 engine with TARGET_K=0 (no holding target): r3b 177k (worse); TAKE_Z 0.5: r3b 234k (noise-level gain).
- Offline (optimistic) sim: take on |p25|>=0.6, hold H s, exit — with taker exit ~250-275k, with free passive exit
  at touch up to 330-390k for H 0.5-1 s. So the lever = recycle capacity: take, then exit cheaply (passively) after ~1-2 s.
Models: logs/gw/models_r3a.json has ridge (b, w) on the 33 DirectionModel features (dm.x) for H 0.25,0.5,1,2,5,10,30 s,
fit on round3_a. strategy_v8_r3.py's DirectionModel W is the 0.25 s model; LONG_MODELS[2] the 2 s model (both r3a-fit).
Per-event features dump: logs/gw/feat/<round>.npz (t us, bid, ask, bq, aq, sv signed mkt aggr vol, p, pl (r3full weights!), X).
logs/gw/make_flip.py shows how to build a standalone file containing DirectionModel.
