import numpy as np

def ridge(X, y, lam):
    mu, sd = X.mean(0), X.std(0); keep = sd > 1e-12
    Z = (X[:, keep] - mu[keep]) / sd[keep]; ym = y.mean()
    ws = np.linalg.solve(Z.T @ Z + lam * len(y) * np.eye(Z.shape[1]), Z.T @ (y - ym))
    w = np.zeros(X.shape[1]); w[keep] = ws / sd[keep]
    return ym - float(w @ mu), w

def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
