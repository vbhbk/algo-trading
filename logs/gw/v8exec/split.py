import sys, numpy as np, pandas as pd
d, r = sys.argv[1], sys.argv[2]
tr = pd.read_csv(d + '/trades.csv'); u = tr[(tr.taker_owner == 'user') | (tr.maker_owner == 'user')].copy()
u['passive'] = u.maker_owner == 'user'
u['sg'] = np.where(u.passive, np.where(u.taker_side == 'B', -1, 1), np.where(u.taker_side == 'B', 1, -1))
pos = (u.sg * u.quantity).cumsum().values; prev = np.r_[0, pos[:-1]]
u['kind'] = np.where(np.sign(u.sg) == -np.sign(prev), np.where(np.abs(pos) <= np.abs(prev), 'reduce', 'flip'), 'add')
F = np.load(f'logs/gw/feat/{r}.npz'); ok = ~np.isnan(F['bid']); t = F['t'][ok]; mid = (F['bid'][ok] + F['ask'][ok]) / 2
mid_at = lambda ts: mid[np.clip(np.searchsorted(t, ts, side='right') - 1, 0, len(t) - 1)]
for (pa, k), g in u.groupby(['passive', 'kind']):
    out = [f"{np.sum(g.sg*(mid_at(g.timestamp_us.values+int(H*1e6))-g.price)*g.quantity)/g.quantity.sum():+.2f}" for H in [0, 1, 2, 5, 30]]
    print('PASSIVE' if pa else 'TAKER  ', f'{k:7s} vol {g.quantity.sum():7d}  markout/lot 0/1/2/5/30s', out)
