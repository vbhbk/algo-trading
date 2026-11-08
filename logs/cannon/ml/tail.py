"""Trading value of prediction tails on R4 >=3000 s: IOC at touch 5 ms later, mark at mid H later, per signal event."""
import numpy as np, sys, json, re, ast
sys.path.insert(0, 'logs/cannon/ml'); from common import ridge
d = np.load('logs/cannon/r4/feat_round4_public.npz')
t, mid, t_on, mid_on, X = d['t'], d['mid'], d['t_on'], d['mid_on'], d['X'].astype(np.float64)
ok = t_on + 10_000_000 <= t[-1]; t_on, mid_on, X = t_on[ok], mid_on[ok], X[ok]
def m_at(x): return mid[np.searchsorted(t, x, side='right') - 1]
# entry: ask/bid at t+5ms ~ mid5 +- half spread; spread at t+5ms unknown here -> assume 0.5 (spread-1 state)
m5 = m_at(t_on + 5000)
src = open('strategies/cannon_ref.py').read()
W3 = {'0.25': (float(re.search(r'^    INTERCEPT = (.*)$', src, re.M).group(1)), np.array(ast.literal_eval(re.search(r'^    W = (\(.*\))$', src, re.M).group(1)))),
      '2.0': tuple(np.array(v) if i else v for i, v in enumerate(ast.literal_eval(re.search(r'^    2: (\(.*\)),$', src, re.M).group(1))))}
tr = t_on < 1.8e9; te = t_on >= 3e9
for H, key in ((0.25, '0.25'), (2.0, '2.0')):
    mh = m_at(t_on + int(H * 1e6)); y = np.clip(mh - mid_on, -10, 10)
    b4, w4 = ridge(X[tr], y[tr], 0.01)
    for lab, (b, w) in (('R3a', W3[key]), ('R4a1', (b4, w4))):
        p = (b + X @ w)[te]; g = (mh - m5)[te]
        out = []
        for q in (0.9, 0.97, 0.99, 0.997):
            z = np.quantile(np.abs(p), q); s = np.abs(p) >= z
            out.append(f'q{q}: z={z:.2f} n={s.sum()} net/lot={np.mean(np.sign(p[s]) * g[s]) - 0.5:+.2f}')
        print(f'H={H} {lab:5s} ' + ' | '.join(out))
