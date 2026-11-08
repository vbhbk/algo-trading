#!/bin/bash
# usage: bt.sh strategy_name dataset
s=$1; d=$2; o=logs/cannon/r4/${s}_$d
uv run exchange-simulator --strategy strategies/$s.py --backtest data/$d.csv --log-dir $o --no-logs > $o.out 2>&1
