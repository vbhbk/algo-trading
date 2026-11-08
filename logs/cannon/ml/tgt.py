"""How much of the ridge R2 is in the first 5 ms (untradable)? Train R4 t<1800 s, test 1800-3000 s and >=3000 s."""
import numpy as np, sys
sys.path.insert(0, 'logs/cannon/ml'); from common import ridge
d = np.load('logs/cannon/r4/feat_round4_public.npz')
t, mid, t_on, mid_on, X = d['t'], d['mid'], d['t_on'], d['mid_on'], d['X'].astype(np.float64)
def m_at(x): return mid[np.searchsorted(t, x, side='right') - 1]
def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
ok = t_on + 10_000_000 <= t[-1]
t_on, mid_on, X = t_on[ok], mid_on[ok], X[ok]
m5 = m_at(t_on + 5000)
tr = t_on < 1.8e9; va = (t_on >= 1.8e9) & (t_on < 3e9); te = t_on >= 3e9
print('frac |mid(t+5ms)-mid(t)|>0:', (m5 != mid_on).mean())
for H in (0.25, 1, 2, 5):
    mh = m_at(t_on + int(H * 1e6))
    for lab, y in (('from t', mh - mid_on), ('from t+5ms', mh - m5)):
        y = np.clip(y, -10, 10)
        b, w = ridge(X[tr], y[tr], 0.01)
        p = b + X @ w
        print(f'H={H:4} {lab:10s} R2 va {r2(y[va], p[va]):.3f} te {r2(y[te], p[te]):.3f}')
