"""pdiff.py dirA dirB: PnL difference A-B with SE from 60 s blocks of the per-second pnl curves."""
import sys, numpy as np, pandas as pd
a, b = [pd.read_csv(f'{d}/pnl.csv') for d in sys.argv[1:3]]
c = a.columns[-1] if 'pnl' not in a.columns else 'pnl'
n = min(len(a), len(b)); d = a[c].values[:n] - b[c].values[:n]
inc = np.diff(np.r_[0, d[59::60]])
print(f'diff {d[-1]:+.0f}  se {inc.std(ddof=1) * np.sqrt(len(inc)):.0f}  blocks {len(inc)}')
