"""Fit ridge direction models (same 33 features as strategy_v8) on a train tape, report R2 on a holdout tape."""

import importlib.util
import json
import sys

import numpy as np

spec = importlib.util.spec_from_file_location('v8', 'strategies/strategy_v8.py')
v8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v8)

train_f, test_f, out_f = sys.argv[1], sys.argv[2], sys.argv[3]
YCLIP = 10.0


def load(f, h):
    d = np.load(f)
    t, mid, t_on, mid_on, X = d['t'], d['mid'], d['t_on'], d['mid_on'], d['X'].astype(np.float64)
    tgt = t_on + int(h * 1e6)
    ok = tgt <= t[-1]
    idx = np.searchsorted(t, tgt[ok], side='right') - 1
    y = np.clip(mid[idx] - mid_on[ok], -YCLIP, YCLIP)
    return X[ok], y


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def ridge(X, y, lam):
    mu, sd = X.mean(0), X.std(0)
    keep = sd > 1e-12
    Z = (X[:, keep] - mu[keep]) / sd[keep]
    ym = y.mean()
    A = Z.T @ Z + lam * len(y) * np.eye(Z.shape[1])
    ws = np.linalg.solve(A, Z.T @ (y - ym))
    w = np.zeros(X.shape[1]); w[keep] = ws / sd[keep]
    b = ym - float(w @ mu)
    return b, w


res = {}
for h, ref in ((0.25, (v8.DirectionModel.INTERCEPT, np.array(v8.DirectionModel.W))),
               (2.0, (v8.LONG_MODELS[2][0], np.array(v8.LONG_MODELS[2][1])))):
    Xa, ya = load(train_f, h)
    Xb, yb = load(test_f, h)
    b0, w0 = ref
    print(f'H={h}s rows train {len(ya)}, test {len(yb)}; v8 weights R2 train {r2(ya, b0 + Xa @ w0):.4f}, test {r2(yb, b0 + Xb @ w0):.4f}')
    n = len(ya); k = int(0.7 * n)
    best = None
    for lam in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
        b, w = ridge(Xa[:k], ya[:k], lam)
        s = r2(ya[k:], b + Xa[k:] @ w)
        print(f'   lam {lam:g}: inner R2 {s:.4f}')
        if best is None or s > best[0]:
            best = (s, lam)
    b, w = ridge(Xa, ya, best[1])
    print(f'   chosen lam {best[1]:g}; refit on full train: R2 train {r2(ya, b + Xa @ w):.4f}, test {r2(yb, b + Xb @ w):.4f}')
    res[str(h)] = {'b': b, 'w': w.tolist(), 'lam': best[1]}
json.dump(res, open(out_f, 'w'))
print('saved', out_f)
