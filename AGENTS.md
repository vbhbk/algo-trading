# Agent rules: Qminers Algo Trade Arena

Goal: maximise the PnL of ONE strategy file. Score = cash + markout - penalty.
Inventory beyond +/-100 costs $10,000 per second: never allow it.

- Never edit simulator/ or explorer/: the official scoring runs this exact code.
- Only touch your own files: strategies/<your-branch-suffix>_*.py and logs/<your-branch-suffix>/.
- Data is shared and read-only: ../algo-trading/data/*.csv. Do not tune on sample.csv (synthetic, trending);
  the real rounds look nothing like it.
- Backtest:
  uv run exchange-simulator --strategy strategies/X.py --backtest ../algo-trading/data/ROUND.csv --log-dir logs/<suffix>/X --no-logs
  Drop --no-logs only when you want to inspect a run with: uv run backtest-explorer --log-dir ...
- Submission rules: one .py file, one BaseStrategy subclass, no constructor args. Imports: stdlib, numpy, pandas,
  scipy, scikit-learn, simulator.models, simulator.strategy.BaseStrategy only. Forbidden: os, sys, io, socket,
  subprocess, threading, importlib, inspect, gc, open, eval, exec, getattr, type, dunder attributes, str.format.
- Report every result as: round, pnl, breach_penalty, fills, abs_inventory_q95.
- Do not commit unless asked.
