"""Robustness: r3a-only models evaluated on r3b / r2 both halves; regularised GBM; per-block markouts."""
import numpy as np
_src = open('logs/gw/spread2/ml.py').read()
exec(_src.split("print('n spread>1")[0])
exec('def thin' + _src.split('def thin')[1].split('for mdl in')[0])
S['r3a'] = [('round3_a', sel('round3_a'))]; S['r2h1'] = [('round2_public', sel('round2_public', hi=3000))]
def run(trainset, H=0.5, leaf=2000):
    Xtr, ytr = XY(trainset, H)
    rid = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xtr, ytr)
    g = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=leaf,
                                      l2_regularization=5.0, random_state=0).fit(Xtr[::2], ytr[::2])
    return rid, g
for tr in ['r3a', 'train']:
    rid, g = run(tr)
    for nm in ['r3a', 'r2h1', 'r3b', 'r2h2']:
        X, y = XY(nm, 0.5); pr, pg = rid.predict(X), g.predict(X)
        f, i = S[nm][0]; t = D[f]['t']
        line = f'train={tr:5s} test {nm}: ridge R2 {r2(y,pr):+.3f} c {cor(y,pr):.3f} | regGBM R2 {r2(y,pg):+.3f} c {cor(y,pg):.3f}'
        for mn, p in [('ridge', pr), ('gbm', pg)]:
            ii = thin(t, i[np.abs(p) >= 1.5]); pp = p[np.searchsorted(i, ii)]; o = markout(f, ii, np.sign(pp))
            # per-1000 s block mean of the 1 s markout
            blk = np.floor((t[ii] - t[ii][0]) / 1000).astype(int)
            bm = [round(float(o[1][blk == b].mean()), 2) for b in np.unique(blk)]
            line += f' | {mn} take|p|>=1.5 n {len(ii)} mk1s {o[1].mean():+.2f} mk2s {o[2].mean():+.2f} blocks1s {bm}'
        print(line)
