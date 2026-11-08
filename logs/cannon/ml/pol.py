"""Screen bang-bang +-95 policies with a deadband on 2 s / 0.25 s predictions (IOC-only sim from logs/cannon/imp/sim.py).
Train R4 t<1800 s, select on 1800-3000 s, report >=3000 s."""
import numpy as np, sys, re, ast
sys.path.insert(0, 'logs/cannon/ml'); sys.path.insert(0, 'logs/cannon/imp')
from common import ridge
from sim import data, run_targets
d = np.load('logs/cannon/r4/feat_round4_public.npz')
t, mid, t_on, mid_on, X = d['t'], d['mid'], d['t_on'], d['mid_on'], d['X'].astype(np.float64)
def m_at(x): return mid[np.searchsorted(t, np.minimum(x, t[-1]), side='right') - 1]
src = open('strategies/cannon_ref.py').read()
W3 = {0.25: (float(re.search(r'^    INTERCEPT = (.*)$', src, re.M).group(1)), np.array(ast.literal_eval(re.search(r'^    W = (\(.*\))$', src, re.M).group(1))))}
m2 = ast.literal_eval(re.search(r'^    2: (\(.*\)),$', src, re.M).group(1)); W3[2.0] = (m2[0], np.array(m2[1]))
tr = t_on < 1.8e9
P = {}
for H in (0.25, 2.0):
    y = np.clip(m_at(t_on + int(H * 1e6)) - mid_on, -10, 10)
    b, w = ridge(X[tr], y[tr], 0.01); P['R4', H] = b + X @ w
    P['R3', H] = W3[H][0] + X @ W3[H][1]
np.savez('logs/cannon/ml/preds_r4.npz', t_on=t_on, **{f'{k}_{h}': v for (k, h), v in P.items()})
E = data('round4_public')[0]
for src_ in ('R3', 'R4'):
    for H in (0.25, 2.0):
        p = P[src_, H]; sd = p[tr].std()
        for zk in (0.0, 0.5, 1.0, 1.5, 2.0):
            z = zk * sd
            tg = np.where(p > z, 95, np.where(p < -z, -95, 0)).astype(np.int64)
            # hold previous target inside the deadband
            idx = np.where(tg != 0, np.arange(len(tg)), 0); np.maximum.accumulate(idx, out=idx); tg = tg[idx]
            r_va = run_targets(E, t_on, tg, int(1.8e9), int(3e9))
            r_te = run_targets(E, t_on, tg, int(3e9), int(6e9))
            print(f'{src_} H={H:4} z={zk:.1f}sd  va {r_va["pnl"]:8d} tr={r_va["trades"]:5d}  te {r_te["pnl"]:8d} tr={r_te["trades"]:5d} cost {r_te["cost"]}', flush=True)
