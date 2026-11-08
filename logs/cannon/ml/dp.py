"""Exact ceiling: best position path in {-95,0,+95} on a 10 ms grid, mark to mid, cost = half-spread x lots traded."""
import numpy as np, sys
sys.path.insert(0, 'logs/cannon/tools'); from load import load
E, T, ok = load('round4_public'); t = E[:, 0]
for t0, t1 in ((3e9, 6e9),):
    g = np.arange(t0, t1, 10_000); k = np.searchsorted(t, g, side='right') - 1
    mid = (E[k, 1] + E[k, 2]) / 2; hs = (E[k, 2] - E[k, 1]) / 2
    S = np.array([-95, 0, 95]); V = np.zeros(3)
    for i in range(len(g) - 1):
        # value after choosing state at i (pay cost), then earn state*(mid[i+1]-mid[i])
        tr = V[:, None] - hs[i] * np.abs(S[None, :] - S[:, None])   # from row -> to col
        V = tr.max(0) + S * (mid[i + 1] - mid[i])
    print(f'DP ceiling {t0/1e9:.0f}-{t1/1e9:.0f}s: {V.max():.0f}')
