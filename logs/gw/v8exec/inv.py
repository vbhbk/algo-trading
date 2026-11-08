import sys, numpy as np, pandas as pd
tr = pd.read_csv(sys.argv[1] + '/trades.csv'); u = tr[(tr.taker_owner == 'user') | (tr.maker_owner == 'user')].copy()
u['passive'] = u.maker_owner == 'user'
u['sg'] = np.where(u.passive, np.where(u.taker_side == 'B', -1, 1), np.where(u.taker_side == 'B', 1, -1))
u['d'] = u.sg * u.quantity
pos = u.d.cumsum().values; prev = np.r_[0, pos[:-1]]
t = u.timestamp_us.values; T = 3000e6
dur = np.diff(np.r_[t, T]); dur = np.maximum(dur, 0)
ap = np.abs(pos)
for lo, hi in [(0, 10), (10, 50), (50, 90), (90, 96)]:
    m = (ap >= lo) & (ap < hi); print(f'|pos| in [{lo},{hi}): {dur[m].sum()/dur.sum():.2f} of time')
red = np.sign(u.d.values) == -np.sign(prev)
for p in [True, False]:
    m = u.passive.values == p
    print('PASSIVE' if p else 'TAKER', 'reducing vol', u.quantity.values[m & red].sum(), 'adding vol', u.quantity.values[m & ~red].sum())
