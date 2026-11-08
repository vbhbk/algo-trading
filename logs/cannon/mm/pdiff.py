# paired PnL difference vs baseline with a 60 s block standard error
import sys, numpy as np, pandas as pd
base = {0: 'logs/cannon/mm/full_base_round3_a', 1: 'logs/cannon/mm/full_base_round2_a'}
for d in sys.argv[1:]:
    k = int(d[-1])
    a = pd.read_csv(base[k] + '/pnl.csv').pnl.values; b = pd.read_csv(d + '/pnl.csv').pnl.values
    n = min(len(a), len(b)); x = (b - a)[:n]
    inc = np.diff(x[::60])
    print(f'{d}: diff {x[n-1]:+.0f}  se {inc.std()*np.sqrt(len(inc)):.0f}')
