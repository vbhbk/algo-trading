"""export.py template.py out.py train_feat.npz [t_max_s] [n_trees] [depth] [lr]: fit HGB on the 0.25 s target and write
its trees into a gl_gbm-style strategy (GBM_BASE/ROOTS/F/T/L/R/V), ridge weights from cannon_ref, v8 thresholds."""
import numpy as np, sys, re, json
from sklearn.ensemble import HistGradientBoostingRegressor
tpl, out, feat = sys.argv[1:4]
tmax = float(sys.argv[4]) if len(sys.argv) > 4 else 1e18
nt = int(sys.argv[5]) if len(sys.argv) > 5 else 100
dep = int(sys.argv[6]) if len(sys.argv) > 6 else 5
lr = float(sys.argv[7]) if len(sys.argv) > 7 else 0.1
Xs, ys = [], []
for fp in feat.split(','):
    d = np.load(fp); t, mid, t_on, mid_on, X = d['t'], d['mid'], d['t_on'], d['mid_on'], d['X'].astype(np.float64)
    ok = (t_on + 250_000 <= t[-1]) & (t_on < tmax * 1e6)
    ys.append(np.clip(mid[np.searchsorted(t, t_on[ok] + 250_000, side='right') - 1] - mid_on[ok], -10, 10)); Xs.append(X[ok])
Xf = np.vstack(Xs); y = np.concatenate(ys)
rng = np.random.default_rng(0); sub = np.sort(rng.choice(len(y), min(len(y), 1_000_000), replace=False))
m = HistGradientBoostingRegressor(max_iter=nt, max_depth=dep, learning_rate=lr, min_samples_leaf=200, random_state=0).fit(Xf[sub], y[sub])
F, T, L, R, V, roots = [], [], [], [], [], []
for pr in m._predictors:
    nodes = pr[0].nodes; base = len(F); roots.append(base)
    leaf_id = {}
    for i, n in enumerate(nodes):
        if n['is_leaf']:
            leaf_id[i] = len(V); V.append(float(n['value']))
    def ref(j): return ~leaf_id[j] if nodes[j]['is_leaf'] else base + int((nodes['is_leaf'][:j] == 0).sum())
    for i, n in enumerate(nodes):
        if n['is_leaf']: continue
        F.append(int(n['feature_idx'])); T.append(float(n['num_threshold'])); L.append(ref(int(n['left']))); R.append(ref(int(n['right'])))
    if nodes[0]['is_leaf']: raise SystemExit('stump tree')
base_pred = float(np.ravel(m._baseline_prediction)[0])
# check exported traversal == sklearn
def walk(x):
    g = base_pred
    for n in roots:
        while n >= 0: n = L[n] if x[F[n]] <= T[n] else R[n]
        g += V[~n]
    return g
chk = sub[:2000]; err = np.max(np.abs(np.array([walk(x) for x in Xf[chk]]) - m.predict(Xf[chk])))
print('trees', len(roots), 'nodes', len(F), 'leaves', len(V), 'max export err', err)
s = open(tpl).read()
for k, v in (('GBM_BASE', repr(base_pred)), ('GBM_ROOTS', repr(tuple(roots))), ('GBM_F', repr(tuple(F))), ('GBM_T', repr(tuple(T))),
             ('GBM_L', repr(tuple(L))), ('GBM_R', repr(tuple(R))), ('GBM_V', repr(tuple(V)))):
    s, n = re.subn(rf'^{k} = .*$', f'{k} = {v}', s, flags=re.M); assert n == 1, k
open(out, 'w').write(s)
