"""fit2.py out.json lam025 lam2 train1.npz [train2.npz ...]: ridge on concatenated tapes (targets per tape), same math as logs/max/r3/fit.py."""
import json, sys
import numpy as np
YCLIP = 10.0

def load(f, h):
    d = np.load(f)
    t, mid, t_on, mid_on, X = d['t'], d['mid'], d['t_on'], d['mid_on'], d['X'].astype(np.float64)
    tgt = t_on + int(h * 1e6); ok = tgt <= t[-1]
    idx = np.searchsorted(t, tgt[ok], side='right') - 1
    return X[ok], np.clip(mid[idx] - mid_on[ok], -YCLIP, YCLIP)

def ridge(X, y, lam):
    mu, sd = X.mean(0), X.std(0); keep = sd > 1e-12
    Z = (X[:, keep] - mu[keep]) / sd[keep]; ym = y.mean()
    ws = np.linalg.solve(Z.T @ Z + lam * len(y) * np.eye(Z.shape[1]), Z.T @ (y - ym))
    w = np.zeros(X.shape[1]); w[keep] = ws / sd[keep]
    return ym - float(w @ mu), w

out, l1, l2, files = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4:]
res = {}
for h, lam in ((0.25, l1), (2.0, l2)):
    XY = [load(f, h) for f in files]
    X = np.vstack([a for a, _ in XY]); y = np.concatenate([b for _, b in XY])
    b, w = ridge(X, y, lam)
    res[str(h)] = {'b': b, 'w': w.tolist(), 'lam': lam}
json.dump(res, open(out, 'w')); print('saved', out)
