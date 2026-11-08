#!/usr/bin/env bash
# with logs + decomp: logs/gw/v8exec/btl.sh strategies/X.py round3_a
cd "$(dirname "$0")/../../.."
s=$1; r=$2; n=$(basename "$s" .py); d=logs/gw/v8exec/runs/$n/$r
rm -rf $d; mkdir -p $(dirname $d); uv run exchange-simulator --strategy "$s" --backtest data/$r.csv --log-dir $d > $d.out 2>&1 || { tail -5 $d.out; exit 1; }
python3 -c "import json;d=json.load(open('$d/result.json'));print(f\"$n $r pnl={d['pnl']:.0f} se={d['pnl_se']:.0f} pen={d['breach_penalty']:.0f} fills={d['fills']} vol={d['traded_volume']} q95={d['abs_inventory_q95']} q50={d['abs_inventory_q50']}\")"
uv run python logs/gw/decomp.py $d $r 2>&1 | grep -v "^ *taker markout/lot H=0.05\|H=10s\|H=60s"
rm -f $d/events.csv $d/book_deltas.csv $d/run.jsonl
