"""Build an ML variant of gw_v8hyst_l2u45: python logs/gw/gbm/make.py out.py models.json MODE [KEY=VAL ...]
MODE: rmix (ridge refit, both models), gbm (0.25 s = GBM), avg (0.25 s = (ridge + GBM) / 2). 2 s model = ridge refit."""
import json, re, sys
out, mj, mode = sys.argv[1:4]
M = json.load(open(mj))
s = open('strategies/gw_v8hyst_l2u45.py').read()
b, w = M['p25_ridge']
s = re.sub(r'    INTERCEPT = .*\n', f'    INTERCEPT = {b!r}\n', s, count=1)
s = re.sub(r'    W = \(.*\)\n', f'    W = {tuple(w)!r}\n', s, count=1)
b2, w2 = M['p2_ridge']
s = re.sub(r'    2: \(.*\)\),\n', f'    2: ({b2!r}, {tuple(w2)!r}),\n', s, count=1)
for kv in sys.argv[4:]:
    k, v = kv.split('=', 1); s, n = re.subn(rf'^{k} = .*$', f'{k} = {v}', s, flags=re.M); assert n == 1, k
old = """        p = self.INTERCEPT
        for w, v in zip(self.W, x):
            p += w * v
"""
assert old in s
if mode != 'rmix':
    g = M['p25_gbm']
    new = """        p = _gbm(x)
""" if mode == 'gbm' else """        p = self.INTERCEPT
        for w, v in zip(self.W, x):
            p += w * v
        p = 0.5 * (p + _gbm(x))
"""
    s = s.replace(old, new)
    fn = f'''

G_BASE = {g['base']!r}
G_ROOTS = {tuple(g['roots'])!r}
G_F = {tuple(g['F'])!r}
G_T = {tuple(g['T'])!r}
G_L = {tuple(g['L'])!r}
G_R = {tuple(g['R'])!r}
G_V = {tuple(g['V'])!r}


def _gbm(x):
    """Gradient-boosted trees (fit on round3_a + round2_a), flat node arrays; F < 0 marks a leaf."""
    F, T, L, R, V = G_F, G_T, G_L, G_R, G_V
    p = G_BASE
    for i in G_ROOTS:
        f = F[i]
        while f >= 0:
            i = L[i] if x[f] <= T[i] else R[i]
            f = F[i]
        p += V[i]
    return p
'''
    s = s.replace('\n\nclass DirectionModel', fn + '\n\nclass DirectionModel', 1)
s = s.replace(s.split('\n', 1)[0], f'# gw_ml ({mode}): gw_v8hyst_l2u45 with direction models refit on round3_a + round2_a (logs/gw/gbm)', 1)
open(out, 'w').write(s)
