"""Fit 0.25 s / 2 s direction models on round3_a + round2_a (first half of round2_public); export to json.
Ridge (as v8) and a small HistGradientBoosting exported as flat node arrays for pure-python inference."""
import numpy as np, json, sys, time
from sklearn.ensemble import HistGradientBoostingRegressor
TRAIN = [('round3_a', None), ('round2_public', 3000.0)]   # (file, keep t < limit s)
if len(sys.argv) > 2 and sys.argv[2] == 'full':          # for site upload: all of R3 + all of R2
    TRAIN = [('round3_a', None), ('round3_b', None), ('round2_public', None)]
if len(sys.argv) > 2 and sys.argv[2] == 'r4':            # all of R2 + R3 + round4_a (test on round4_b)
    TRAIN = [('round3_a', None), ('round3_b', None), ('round2_public', None), ('round4_a', None)]
if len(sys.argv) > 2 and sys.argv[2] == 'r4full':        # all of R2 + R3 + R4 (for upload)
    TRAIN = [('round3_a', None), ('round3_b', None), ('round2_public', None), ('round4_a', None), ('round4_b', None)]
OUT = sys.argv[1]
def load(f, lim):
    d = np.load(f'logs/gw/feat/{f}.npz'); ok = ~np.isnan(d['bid'])
    t = d['t'][ok] / 1e6; mid = (d['bid'][ok] + d['ask'][ok]) / 2; X = d['X'][ok].astype(np.float64); on = ~np.isnan(X[:, 0])
    return t, mid, X, on, (t < lim if lim else np.ones(len(t), bool))
def xy(H, step=2):
    Xs, ys = [], []
    for f, lim in TRAIN:
        t, mid, X, on, keep = load(f, lim)
        j = np.searchsorted(t, t + H, side='right') - 1
        y = np.clip(mid[j] - mid, -12, 12); m = np.flatnonzero(on & keep & (t < t[-1] - H))[::step]
        Xs.append(X[m]); ys.append(y[m])
    return Xs, ys
def ridge(X, y, lam):
    mu, sd = X.mean(0), X.std(0); keep = sd > 1e-12; Z = (X[:, keep] - mu[keep]) / sd[keep]; ym = y.mean()
    ws = np.linalg.solve(Z.T @ Z + lam * len(y) * np.eye(Z.shape[1]), Z.T @ (y - ym)); w = np.zeros(X.shape[1]); w[keep] = ws / sd[keep]
    return float(ym - w @ mu), w.tolist()
def export(g):
    F, T, L, R, V = [], [], [], [], []
    roots = []
    for (tree,) in g._predictors:
        n = tree.nodes; base = len(F); roots.append(base)
        for nd in n:
            leaf = bool(nd['is_leaf'])
            F.append(-1 if leaf else int(nd['feature_idx'])); T.append(float(nd['num_threshold']))
            L.append(base + int(nd['left'])); R.append(base + int(nd['right'])); V.append(float(nd['value']))
    return {'base': float(g._baseline_prediction.ravel()[0]), 'roots': roots, 'F': F, 'T': T, 'L': L, 'R': R, 'V': V}
res = {}
for H, key in [(0.25, 'p25'), (2.0, 'p2')]:
    Xs, ys = xy(H)
    X = np.vstack(Xs); y = np.concatenate(ys)
    # inner validation: last 30% of each training part
    tr = np.concatenate([np.arange(len(a)) < 0.7 * len(a) for a in ys])
    best = None
    for lam in [1e-5, 1e-4, 1e-3, 1e-2]:
        b, w = ridge(X[tr], y[tr], lam); p = b + X[~tr] @ np.array(w); r = 1 - np.mean((y[~tr] - p) ** 2) / np.var(y[~tr])
        if best is None or r > best[0]: best = (r, lam)
    res[key + '_ridge'] = ridge(X, y, best[1]); print(f'{key} ridge lam {best[1]} inner R2 {best[0]:.3f}')
    bestg = None
    for it, leaves in ([(100, 15)] if 'small' in sys.argv else [(150, 31)] if 'big' in sys.argv else [(100, 15), (200, 15), (150, 31)]):
        g = HistGradientBoostingRegressor(max_iter=it, learning_rate=0.08, max_leaf_nodes=leaves, min_samples_leaf=300,
                                          l2_regularization=1.0, early_stopping=False, random_state=0).fit(X[tr], y[tr])
        r = 1 - np.mean((y[~tr] - g.predict(X[~tr])) ** 2) / np.var(y[~tr]); print(f'  gbm it {it} leaves {leaves}: inner R2 {r:.3f}')
        if bestg is None or r > bestg[0]: bestg = (r, it, leaves)
    g = HistGradientBoostingRegressor(max_iter=bestg[1], learning_rate=0.08, max_leaf_nodes=bestg[2], min_samples_leaf=300,
                                      l2_regularization=1.0, early_stopping=False, random_state=0).fit(X, y)
    e = export(g)
    # check exported inference matches
    xs = X[:200]; pe = []
    for x in xs:
        p = e['base']
        for i in e['roots']:
            while e['F'][i] >= 0:
                i = e['L'][i] if x[e['F'][i]] <= e['T'][i] else e['R'][i]
            p += e['V'][i]
        pe.append(p)
    print(f'  chosen gbm it {bestg[1]} leaves {bestg[2]}; export max abs err {np.max(np.abs(np.array(pe) - g.predict(xs))):.2e}, nodes {len(e["F"])}')
    res[key + '_gbm'] = e
json.dump(res, open(OUT, 'w'))
