import sys, numpy as np, pandas as pd
d, r = sys.argv[1], sys.argv[2]
tr = pd.read_csv(d + '/trades.csv'); u = tr[(tr.taker_owner == 'user') | (tr.maker_owner == 'user')].copy()
u['passive'] = u.maker_owner == 'user'
u['sg'] = np.where(u.passive, np.where(u.taker_side == 'B', -1, 1), np.where(u.taker_side == 'B', 1, -1))
pos = (u.sg * u.quantity).cumsum().values; tt = u.timestamp_us.values
F = np.load(f'logs/gw/feat/{r}.npz'); ok = ~np.isnan(F['bid']) & ~np.isnan(F['p'])
t = F['t'][ok]; p = F['p'][ok]
mid = (F['bid'][ok] + F['ask'][ok]) / 2
P = np.r_[0, pos][np.searchsorted(tt, t, side='left')]
fut = mid[np.clip(np.searchsorted(t, t + 2_000_000), 0, len(t) - 1)] - mid
for z in [0.6, 0.9, 1.2]:
    m = np.abs(p) >= z; s = np.sign(p[m]) * P[m]
    # count distinct "episodes": first event of a run
    st = m & ~np.r_[False, m[:-1]]
    ss = np.sign(p[st]) * P[st]
    print(f'|p|>={z}: events {m.sum()} episodes {st.sum()}  pos-in-dir at episode start: >=90 {np.mean(ss>=90):.2f}  (0,90) {np.mean((ss>0)&(ss<90)):.2f}  <=0 {np.mean(ss<=0):.2f}  mean 2s move in dir {np.mean(np.sign(p[st])*fut[st]):.2f}')
