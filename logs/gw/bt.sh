#!/usr/bin/env bash
# usage: logs/gw/bt.sh strategies/X.py round3_b round2_public ...   -> runs in parallel, prints metrics
cd "$(dirname "$0")/../.."
s=$1; shift; n=$(basename "$s" .py)
for r in "$@"; do
  uv run exchange-simulator --strategy "$s" --backtest data/$r.csv --log-dir logs/gw/runs/$n/$r --no-logs > logs/gw/runs/$n.$r.out 2>&1 &
done; wait
for r in "$@"; do
  python3 -c "import json,sys;d=json.load(open('logs/gw/runs/$n/$r/result.json'));print(f\"$n $r pnl={d['pnl']:.0f} se={d['pnl_se']:.0f} pen={d['breach_penalty']:.0f} fills={d['fills']} q95={d['abs_inventory_q95']} q50={d['abs_inventory_q50']}\")" 2>/dev/null || { echo "$n $r FAILED"; tail -3 logs/gw/runs/$n.$r.out; }
done
