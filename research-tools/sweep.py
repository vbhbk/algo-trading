"""Parameter grid search for a strategy file, run in parallel over several datasets.

Usage (from your worktree root):
    uv run python C:/Projects/max-research-tools/sweep.py \
        --strategy strategies/max_x.py \
        --data ../algo-trading/data/round1_a.csv ../algo-trading/data/sample_v2_a.csv \
        --grid "QUOTE_SIZE=10,20,30;SKEW_STEP=10,20" \
        --out logs/max/x/sweep --workers 6

Each NAME in --grid must be a module-level constant `NAME = value` in the strategy
file. Results are sorted by the worst dataset PnL (robustness first).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import json
import re
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', required=True)
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--grid', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()

    text = Path(a.strategy).read_text(encoding='utf-8')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = [(k.strip(), v.split(',')) for k, v in (p.split('=') for p in a.grid.split(';'))]
    combos = list(itertools.product(*[[(k, v.strip()) for v in vs] for k, vs in grid]))

    files = []
    for i, combo in enumerate(combos):
        t = text
        for k, v in combo:
            t, n = re.subn(rf'^{k} = [^#\n]*', f'{k} = {v}  ', t, flags=re.M)
            if n != 1:
                raise SystemExit(f'constant {k} not found exactly once in {a.strategy}')
        f = out / f's{i}.py'
        f.write_text(t, encoding='utf-8')
        files.append(f)

    def run(job: tuple[int, int]) -> tuple[int, int, dict | None, str]:
        i, d = job
        log = out / f'r{i}_{d}'
        shutil.rmtree(log, ignore_errors=True)
        p = subprocess.run(
            ['uv', 'run', 'exchange-simulator', '--strategy', str(files[i]),
             '--backtest', a.data[d], '--log-dir', str(log), '--no-logs'],
            capture_output=True, text=True,
        )
        res = log / 'result.json'
        if not res.exists():
            return i, d, None, (p.stderr or p.stdout)[-400:]
        return i, d, json.loads(res.read_text()), ''

    jobs = [(i, d) for i in range(len(combos)) for d in range(len(a.data))]
    results: dict[tuple[int, int], dict | None] = {}
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, d, r, err in ex.map(run, jobs):
            results[i, d] = r
            if r is None:
                print(f'FAILED combo {i} on {a.data[d]}: {err}')

    names = [Path(x).stem for x in a.data]
    rows = []
    for i, combo in enumerate(combos):
        rs = [results[i, d] for d in range(len(a.data))]
        if any(r is None for r in rs):
            continue
        rows.append((min(r['pnl'] for r in rs), combo, rs))
    rows.sort(key=lambda x: -x[0])
    for worst, combo, rs in rows:
        params = ' '.join(f'{k}={v}' for k, v in combo)
        cells = ' | '.join(
            f"{n}: pnl={r['pnl']:.0f}+-{r['pnl_se'] or 0:.0f} pen={r['breach_penalty']:.0f} "
            f"fills={r['fills']} q95={r['abs_inventory_q95']} q997={r['abs_inventory_q997']}"
            for n, r in zip(names, rs, strict=True)
        )
        print(f'worst={worst:.0f} | {params} | {cells}')


if __name__ == '__main__':
    main()
