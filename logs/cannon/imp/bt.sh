#!/bin/bash
# usage: bt.sh NAME DATASET...   (runs sequentially)
n=$1; shift
for d in "$@"; do
  uv run exchange-simulator --strategy strategies/cannon_imp_${S:-$n}.py --backtest data/$d.csv --log-dir logs/cannon/imp/$n/$d --no-logs > /dev/null 2>&1
  python3 -c "import json;r=json.load(open('logs/cannon/imp/$n/$d/result.json'));print('$n','$d',int(r['pnl']),r['breach_penalty'],r['fills'],r['abs_inventory_q95'],r['abs_inventory_q50'])"
done
