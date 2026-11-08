"""Headroom: bang-bang +-95 on the TRUE future mid move (perfect foresight), same IOC sim, R4 >= 3000 s."""
import numpy as np, sys
sys.path.insert(0, 'logs/cannon/imp'); from sim import data, run_targets
d = np.load('logs/cannon/r4/feat_round4_public.npz'); t, mid, t_on, mid_on = d['t'], d['mid'], d['t_on'], d['mid_on']
def m_at(x): return mid[np.searchsorted(t, np.minimum(x, t[-1]), side='right') - 1]
E = data('round4_public')[0]
P = np.load('logs/cannon/ml/preds_r4.npz')
for H in (0.25, 1.0, 2.0):
    y = m_at(t_on + int(H * 1e6)) - m_at(t_on + 5000)
    for z in (0.5, 1.0, 2.0, 3.0):
        tg = np.where(y > z, 95, np.where(y < -z, -95, 0)).astype(np.int64)
        idx = np.where(tg != 0, np.arange(len(tg)), 0); np.maximum.accumulate(idx, out=idx); tg = tg[idx]
        r = run_targets(E, t_on, tg, int(3e9), int(6e9)); print(f'oracle H={H} z={z}: te {r["pnl"]} trades {r["trades"]}', flush=True)
# noisy oracle: corr between model and truth
p = P['R3_0.25']; y = m_at(t_on + 250000) - mid_on
print('corr R3 ridge 0.25 vs truth', np.corrcoef(p, y)[0, 1], 'gbm', np.corrcoef(P['gbm36_0.25'], y)[0, 1])
