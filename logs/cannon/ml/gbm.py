"""HistGradientBoosting on 33 v8 features (+ impulse score) for R4; same policy screen as pol.py."""
import numpy as np, sys, time
sys.path.insert(0, 'logs/cannon/ml'); sys.path.insert(0, 'logs/cannon/imp'); sys.path.insert(0, 'logs/cannon/tools')
from common import r2
from sim import data, run_targets
from load import takers
from sklearn.ensemble import HistGradientBoostingRegressor
d = np.load('logs/cannon/r4/feat_round4_public.npz')
t, mid, t_on, mid_on, X = d['t'], d['mid'], d['t_on'], d['mid_on'], d['X'].astype(np.float64)
def m_at(x): return mid[np.searchsorted(t, np.minimum(x, t[-1]), side='right') - 1]
E, T, K = data('round4_public')
# impulse score: sum sign*min(q,1000)*exp(-(t-ti)/tau) over aggressive orders up to t (inclusive of same timestamp)
def impulse(tau_us):
    kt = K[:, 0]; v = K[:, 1] * np.minimum(K[:, 2], 1000)
    s = np.zeros(len(kt)); acc = 0.0; last = kt[0]
    for i in range(len(kt)):
        acc = acc * np.exp(-(kt[i] - last) / tau_us) + v[i]; last = kt[i]; s[i] = acc
    j = np.searchsorted(kt, t_on, side='right') - 1
    out = np.where(j >= 0, s[np.maximum(j, 0)] * np.exp(-(t_on - kt[np.maximum(j, 0)]) / tau_us), 0.0)
    return out
imp = np.c_[impulse(200_000), impulse(500_000), impulse(2_000_000)] * 1e-3
XI = np.c_[X, imp]
tr = t_on < 1.8e9; va = (t_on >= 1.8e9) & (t_on < 3e9); te = t_on >= 3e9
rng = np.random.default_rng(0); sub = np.nonzero(tr)[0]; sub = np.sort(rng.choice(sub, 300_000, replace=False))
P = dict(np.load('logs/cannon/ml/preds_r4.npz'))
for H in (0.25, 2.0):
    y = np.clip(m_at(t_on + int(H * 1e6)) - mid_on, -10, 10)
    for lab, F in (('gbm33', X), ('gbm36', XI)):
        t0 = time.time()
        m = HistGradientBoostingRegressor(max_iter=200, max_depth=5, learning_rate=0.05, min_samples_leaf=200, random_state=0).fit(F[sub], y[sub])
        p = m.predict(F); P[f'{lab}_{H}'] = p
        print(f'{lab} H={H} R2 va {r2(y[va], p[va]):.3f} te {r2(y[te], p[te]):.3f}  ({time.time()-t0:.0f}s)', flush=True)
    rr = P[f'R4_{H}']; print(f'ridgeR4 H={H} R2 va {r2(y[va], rr[va]):.3f} te {r2(y[te], rr[te]):.3f}')
np.savez('logs/cannon/ml/preds_r4.npz', **P)
for key in ('gbm33_0.25', 'gbm36_0.25', 'gbm33_2.0', 'gbm36_2.0'):
    p = P[key]; sd = p[tr].std()
    for zk in (0.5, 0.75, 1.0, 1.25):
        tg = np.where(p > zk * sd, 95, np.where(p < -zk * sd, -95, 0)).astype(np.int64)
        idx = np.where(tg != 0, np.arange(len(tg)), 0); np.maximum.accumulate(idx, out=idx); tg = tg[idx]
        a = run_targets(E, t_on, tg, int(1.8e9), int(3e9)); b = run_targets(E, t_on, tg, int(3e9), int(6e9))
        print(f'{key:11s} z={zk:.2f}sd va {a["pnl"]:8d} tr={a["trades"]:5d}  te {b["pnl"]:8d} tr={b["trades"]:5d}', flush=True)
