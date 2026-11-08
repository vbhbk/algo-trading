"""Tasks 2+3: predict mid change in spread>1 states; OOS R2/corr; IOC-at-touch markouts.
Train: round3_a + round2_public t<3000 s. Test: round3_b, round2_public t>=3000 s."""
import numpy as np, time
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
np.random.seed(0)
HS = [0.25, 0.5, 'close']
FN = None
def build(f):
    d = dict(np.load(f'logs/gw/spread2/feat_{f}.npz')); t = d['t'] / 1e6; bid, ask = d['bid'], d['ask']
    sp = ask - bid; mid = (bid + ask) / 2; n = len(t)
    X = d['X'].astype(np.float64)
    pre = (d['rb'] + d['ra']) / 2
    E = np.column_stack([sp, d['ga'] - d['gb'], d['gb'], d['ga'], np.minimum(d['age'], 2), np.sign(d['lastq']) * np.log1p(np.abs(d['lastq'])),
                         d['lastdt'], np.sign(d['sv']) * np.log1p(np.abs(d['sv'])), np.log1p(d['d5b']), np.log1p(d['d5a']),
                         np.log1p(d['d3b']), np.log1p(d['d3a']), d['g2b'], d['g2a'], mid - pre,
                         (np.log1p(d['d3b']) - np.log1p(d['d3a'])), (d['bq'] - d['aq']) / (d['bq'] + d['aq'])])
    F = np.column_stack([X, E])
    Y = {}
    for H in [0.25, 0.5]:
        j = np.searchsorted(t, t + H, side='right') - 1; Y[H] = np.clip(mid[j] - mid, -12, 12)
    # until spread closes: mid at the next event with spread<=1
    one = np.flatnonzero(sp <= 1); k = np.searchsorted(one, np.arange(n)); k = np.minimum(k, len(one) - 1)
    Y['close'] = np.clip(mid[one[k]] - mid, -12, 12)
    m = (sp > 1) & (t < t[-1] - 3) & np.isfinite(F).all(1)
    return dict(t=t, bid=bid, ask=ask, mid=mid, sp=sp, F=F, Y=Y, m=m, aq=d['aq'], bq=d['bq'], age=d['age'])
D = {f: build(f) for f in ['round3_a', 'round3_b', 'round2_public']}
def sel(f, lo=-1, hi=1e9):
    d = D[f]; return np.flatnonzero(d['m'] & (d['t'] >= lo) & (d['t'] < hi))
S = {'train': [('round3_a', sel('round3_a')), ('round2_public', sel('round2_public', hi=3000))],
     'r3b': [('round3_b', sel('round3_b'))], 'r2h2': [('round2_public', sel('round2_public', lo=3000))]}
def XY(name, H):
    return (np.vstack([D[f]['F'][i] for f, i in S[name]]), np.concatenate([D[f]['Y'][H][i] for f, i in S[name]]))
r2 = lambda y, p: 1 - np.mean((y - p) ** 2) / np.var(y)
cor = lambda y, p: np.corrcoef(y, p)[0, 1]
print('n spread>1 events: train %d, r3b %d, r2h2 %d' % tuple(len(XY(k, 0.5)[1]) for k in ['train', 'r3b', 'r2h2']))
MODELS = {}
for H in HS:
    Xtr, ytr = XY('train', H)
    rid = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xtr, ytr)
    side = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xtr[:, [33, 34, 47]], ytr)   # spread, gap side, mid-pre
    sub = np.arange(0, len(ytr), 2)
    t0 = time.time()
    g = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300,
                                      l2_regularization=1.0, random_state=0).fit(Xtr[sub], ytr[sub])
    MODELS[H] = (rid, g, side)
    row = []
    for nm in ['train', 'r3b', 'r2h2']:
        X, y = XY(nm, H); pr, pg, ps = rid.predict(X), g.predict(X), side.predict(X[:, [33, 34, 47]])
        row.append(f'{nm}: ridge R2 {r2(y,pr):.3f} c {cor(y,pr):.3f} | gbm R2 {r2(y,pg):.3f} c {cor(y,pg):.3f} | gap-only R2 {r2(y,ps):.3f}')
    print(f'H={H} (gbm fit {time.time()-t0:.0f}s, std y test r3b {XY("r3b",H)[1].std():.2f}):\n   ' + '\n   '.join(row))

# ---- Task 3: IOC at touch, 5 ms latency, in sign of the 0.5 s prediction; markout to mid at 0.5/1/2 s
print('\n### IOC markouts (ticks/lot, entry = touch at t+5ms on the taking side; one take per 100 ms)')
def thin(t, idx, gap=0.1):
    keep = []; last = -1e9
    for i in idx:
        if t[i] - last >= gap: keep.append(i); last = t[i]
    return np.array(keep, dtype=int)
def markout(f, idx, s):
    d = D[f]; t = d['t']; k = np.searchsorted(t, t[idx] + 0.005)
    entry = np.where(s > 0, d['ask'][k], d['bid'][k]); q = np.where(s > 0, d['aq'][k], d['bq'][k])
    out = {}
    for H in [0.5, 1, 2]:
        j = np.searchsorted(t, t[idx] + H, side='right') - 1
        out[H] = s * (d['mid'][j] - entry)
    out['cost'] = s * (entry - d['mid'][idx]); out['q'] = q; out['sp_arr'] = d['sp'][k]
    return out
for mdl in ['ridge', 'gbm']:
    for nm in ['r3b', 'r2h2']:
        f, i = S[nm][0]; X = D[f]['F'][i]
        rid, g, _ = MODELS[0.5]; p = rid.predict(X) if mdl == 'ridge' else g.predict(X)
        T = D[f]['t'][i]
        for thr in [1.0, 1.5, 2.0, 3.0]:
            ii = thin(D[f]['t'], i[np.abs(p) >= thr]); pp = p[np.searchsorted(i, ii)]
            if len(ii) < 20: continue
            o = markout(f, ii, np.sign(pp)); dur = T[-1] - T[0]
            print(f'{mdl:5s} {nm:5s} |p05|>={thr}: n {len(ii):5d} ({len(ii)/dur:.2f}/s) pred {np.abs(pp).mean():.2f} cost(entry-mid) {o["cost"].mean():.2f} '
                  f'markout 0.5s {o[0.5].mean():+.3f} 1s {o[1].mean():+.3f} 2s {o[2].mean():+.3f} (se2s {o[2].std()/np.sqrt(len(ii)):.3f}) touchq med {np.median(o["q"]):.0f} spread@arr>1 {np.mean(o["sp_arr"]>1):.2f}')
# simple rule: at the episode-opening event, take in the gap direction
print('\n### rule: IOC in gap direction at episode open (first spread>1 event)')
for nm in ['train', 'r3b', 'r2h2']:
    for f, i in S[nm]:
        d = D[f]; op = i[d['age'][i] == 0]; s = np.sign(d['F'][op, 34])
        for spk in [2, 3, 4]:
            mm = (d['sp'][op] == spk) if spk < 4 else (d['sp'][op] >= 4)
            o = markout(f, op[mm], s[mm])
            print(f'{nm} {f} spread{"=" if spk<4 else ">="}{spk}: n {mm.sum()} cost {o["cost"].mean():.2f} mk 0.5s {o[0.5].mean():+.3f} 1s {o[1].mean():+.3f} 2s {o[2].mean():+.3f}')
# comparison: take at the close of the episode (spread back to 1) in gap direction
print('\n### rule: IOC in gap direction at the event the spread closes (spread 1 again)')
for nm in ['r3b', 'r2h2']:
    f, i = S[nm][0]; d = D[f]; op = i[d['age'][i] == 0]; s = np.sign(d['F'][op, 34])
    one = np.flatnonzero(d['sp'] <= 1); cl = one[np.searchsorted(one, op)]
    o = markout(f, cl, s); print(f'{nm}: n {len(cl)} cost {o["cost"].mean():.2f} mk 0.5s {o[0.5].mean():+.3f} 1s {o[1].mean():+.3f} 2s {o[2].mean():+.3f}')
import pickle; pickle.dump({H: MODELS[H][:2] for H in HS}, open('logs/gw/spread2/models.pkl', 'wb'))
