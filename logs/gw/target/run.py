# Target / loss study: fit on inner-train, pick threshold on inner-val by offline PnL, refit on full train,
# evaluate on round3_b and round2_public 2nd half. Usage: uv run python logs/gw/target/run.py [spec ...]
# 1 core: OMP_NUM_THREADS=1.
import numpy as np, json, time, sys, os.path as osp
sys.path.insert(0, 'logs/gw/target')
from common import *
from sklearn.ensemble import HistGradientBoostingRegressor as HGR, HistGradientBoostingClassifier as HGC
from sklearn.linear_model import HuberRegressor, LogisticRegression

STEP_H, STEP_R = 3, 1
QGRID = np.geomspace(0.6, 0.002, 32)  # val trigger rates of the threshold grid
D = {f: load(f) for f in ROUNDS}
T = {f: targets(D[f]) for f in ROUNDS}
M0 = {float(k): (v[0], np.array(v[1])) for k, v in json.load(open('logs/gw/models_r3a.json')).items()}


def rows(segs, tname, step):
    Xs, ys = [], []
    for f, a, b in segs:
        y = T[f][tname]; m = np.flatnonzero(seg_mask(D[f], a, b) & D[f]['on'] & np.isfinite(y))[::step]
        Xs.append(D[f]['X'][m]); ys.append(y[m])
    return np.vstack(Xs).astype(np.float64), np.concatenate(ys)


# ---- learners: each returns predict(X)->array
def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0); keep = sd > 1e-12; Z = (X[:, keep] - mu[keep]) / sd[keep]; ym = y.mean()
    ws = np.linalg.solve(Z.T @ Z + lam * len(y) * np.eye(Z.shape[1]), Z.T @ (y - ym)); w = np.zeros(X.shape[1]); w[keep] = ws / sd[keep]
    b = ym - float(w @ mu); return lambda X: b + X @ w


def L_ridge(clip=None, lam=1e-4):
    def fit(segs, tn):
        X, y = rows(segs, tn, STEP_R)
        if clip: y = np.clip(y, *clip)
        return ridge_fit(X, y, lam)
    return fit


def L_huber():
    def fit(segs, tn):
        X, y = rows(segs, tn, 4); mu, sd = X.mean(0), X.std(0) + 1e-12
        h = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=200).fit((X - mu) / sd, y)
        return lambda X: h.predict((X - mu) / sd)
    return fit


def L_logit():
    def fit(segs, tn):
        X, y = rows(segs, tn, 2); mu, sd = X.mean(0), X.std(0) + 1e-12
        c = LogisticRegression(C=1.0, max_iter=300).fit((X - mu) / sd, y > 0.5)
        return lambda X: c.predict_proba((X - mu) / sd)[:, 1]
    return fit


HP = dict(max_iter=150, max_leaf_nodes=15, learning_rate=0.1, min_samples_leaf=1000, l2_regularization=1.0, early_stopping=False)


def L_hgb(loss='squared_error', clip=None, seed=0):
    def fit(segs, tn):
        X, y = rows(segs, tn, STEP_H)
        if clip: y = np.clip(y, *clip)
        g = HGR(loss=loss, random_state=seed, **HP).fit(X, y); return g.predict
    return fit


def L_hgbc(cost=False, seed=0):
    """classifier; cost=True: label = target>0 weighted by |target| (money-weighted 0/1 loss)."""
    def fit(segs, tn):
        X, y = rows(segs, tn, STEP_H)
        lab = y > (0 if cost else 0.5); w = np.abs(y) if cost else None
        g = HGC(random_state=seed, **HP).fit(X, lab, sample_weight=w); return lambda X: g.predict_proba(X)[:, 1]
    return fit


# spec: name -> (learner, kind, targets)  kind: 'signed' (one target, su=p, sd=-p), 'pair' (su,sd from two targets),
#        'cls' (two classifiers, su=Pu-Pd, sd=-su)
SPECS = {
    'a_ridge_d25': (L_ridge(), 'signed', ['d25']),
    'a_hgb_d25': (L_hgb(), 'signed', ['d25']),
    'b_ridge_tk0.5': (L_ridge(), 'pair', ['gl0.5', 'gs0.5']),
    'b_ridge_tk1': (L_ridge(), 'pair', ['gl1', 'gs1']),
    'b_ridge_tk2': (L_ridge(), 'pair', ['gl2', 'gs2']),
    'b_hgb_tk0.5': (L_hgb(), 'pair', ['gl0.5', 'gs0.5']),
    'b_hgb_tk1': (L_hgb(), 'pair', ['gl1', 'gs1']),
    'b_hgb_tk2': (L_hgb(), 'pair', ['gl2', 'gs2']),
    'c_logit_ud0.5': (L_logit(), 'cls', ['up0.5', 'dn0.5']),
    'c_logit_ud1': (L_logit(), 'cls', ['up1', 'dn1']),
    'c_hgbc_ud0.5': (L_hgbc(), 'cls', ['up0.5', 'dn0.5']),
    'c_hgbc_ud1': (L_hgbc(), 'cls', ['up1', 'dn1']),
    'd_huber_d25': (L_huber(), 'signed', ['d25']),
    'd_ridge_d25_clip2': (L_ridge(clip=(-2, 2)), 'signed', ['d25']),
    'd_ridge_d25_clip1': (L_ridge(clip=(-1, 1)), 'signed', ['d25']),
    'd_ridge_d25_clip4': (L_ridge(clip=(-4, 4)), 'signed', ['d25']),
    'd_hgb_d25_clip1': (L_hgb(clip=(-1, 1)), 'signed', ['d25']),
    'd_ridge_tk0.5_clip3': (L_ridge(clip=(-3, 3)), 'pair', ['gl0.5', 'gs0.5']),
    'd_hgb_tk0.5_clip3': (L_hgb(clip=(-3, 3)), 'pair', ['gl0.5', 'gs0.5']),
    'd_hgb_d25_clip2': (L_hgb(clip=(-2, 2)), 'signed', ['d25']),
    'd_ridge_tk1_clip3': (L_ridge(clip=(-3, 3)), 'pair', ['gl1', 'gs1']),
    'd_hgb_tk1_clip3': (L_hgb(clip=(-3, 3)), 'pair', ['gl1', 'gs1']),
    'd_hgb_tk1_l1': (L_hgb(loss='absolute_error'), 'pair', ['gl1', 'gs1']),
    'd_hgbc_cost_tk1': (L_hgbc(cost=True), 'pair', ['gl1', 'gs1']),
    'd_hgbc_cost_tk2': (L_hgbc(cost=True), 'pair', ['gl2', 'gs2']),
}
for sd_ in [1, 2]:  # seed variability of HGB (feature-binning subsample)
    SPECS[f'a_hgb_d25_s{sd_}'] = (L_hgb(seed=sd_), 'signed', ['d25'])
    SPECS[f'd_hgb_d25_clip2_s{sd_}'] = (L_hgb(clip=(-2, 2), seed=sd_), 'signed', ['d25'])
    SPECS[f'b_hgb_tk0.5_s{sd_}'] = (L_hgb(seed=sd_), 'pair', ['gl0.5', 'gs0.5'])
    SPECS[f'c_hgbc_ud0.5_s{sd_}'] = (L_hgbc(seed=sd_), 'cls', ['up0.5', 'dn0.5'])


def scores(preds, kind, f):
    on = D[f]['on']; X = D[f]['X'][on].astype(np.float64); n = len(on)
    ps = [np.zeros(n) for _ in preds]
    for p, fn in zip(ps, preds): p[on] = fn(X)
    if kind == 'signed': su = ps[0]; sd = -su
    elif kind == 'pair': su, sd = ps
    else: su = ps[0] - ps[1]; sd = -su
    lo = -1e9; su = np.where(on, su, lo); sd = np.where(on, sd, lo)
    return su, sd


def evaluate(name):
    learner, kind, tns = SPECS[name]; t0 = time.time(); out = dict(name=name)
    # inner fit + threshold
    preds = [learner(INNER_TR, tn) for tn in tns]
    S = {}
    for f, a, b in INNER_VA:
        if f not in S: S[f] = scores(preds, kind, f)
    allv = np.concatenate([np.maximum(*S[f])[seg_mask(D[f], a, b) & D[f]['on']] for f, a, b in INNER_VA])
    grid = np.quantile(allv, 1 - QGRID)
    curve = []
    for thr in grid:
        curve.append(sum(sim(D[f], seg_mask(D[f], a, b), *S[f], thr)[0] for f, a, b in INNER_VA))
    curve = np.array(curve); sm = np.convolve(np.pad(curve, 1, mode='edge'), np.ones(3) / 3, 'valid')
    i = int(np.argmax(sm)); thr = float(grid[i])
    out.update(thr=thr, val_pnl=float(curve[i]), val_rate=float(np.mean(allv >= thr)),
               curve=[(float(g), float(c)) for g, c in zip(grid, curve)])
    # full refit, test
    preds = [learner(FULL_TR, tn) for tn in tns]
    for f, a, b in TEST:
        su, sd = scores(preds, kind, f); m = seg_mask(D[f], a, b)
        pnl, n, mo = sim(D[f], m, su, sd, thr)
        # robustness: test PnL over the whole threshold grid (NOT used for selection)
        tc = [float(sim(D[f], m, su, sd, g)[0]) for g in grid]
        rate = float(np.mean(np.maximum(su, sd)[m & D[f]['on']] >= thr))
        key = 'r3b' if f == 'round3_b' else 'r2h2'
        out[key] = dict(pnl=float(pnl), takes=n, mo1=float(mo[1]), mo2=float(mo[2]), rate=rate, curve=tc)
        if name.startswith('a_'):  # R2 of the d25 regression for reference
            y = T[f]['d25']; mm = m & D[f]['on'] & np.isfinite(y); p = su[mm]
            out[key]['R2'] = float(1 - np.mean((y[mm] - p) ** 2) / np.var(y[mm]))
    out['sec'] = time.time() - t0
    return out


def ref():
    """production reference: r3a ridge p25 (fit on all r3a), |p|>=0.6, no selection."""
    b, w = M0[0.25]; out = dict(name='ref_ridge_r3a_p25_0.6', thr=0.6)
    for f, a, c in TEST:
        p = np.where(D[f]['on'], b + np.nan_to_num(D[f]['X'].astype(np.float64)) @ w, 0); m = seg_mask(D[f], a, c)
        pnl, n, mo = sim(D[f], m, p, -p, 0.6); key = 'r3b' if f == 'round3_b' else 'r2h2'
        y = T[f]['d25']; mm = m & D[f]['on'] & np.isfinite(y)
        out[key] = dict(pnl=float(pnl), takes=n, mo1=float(mo[1]), mo2=float(mo[2]),
                        R2=float(1 - np.mean((y[mm] - p[mm]) ** 2) / np.var(y[mm])))
    return out


if __name__ == '__main__':
    RES = 'logs/gw/target/results.json'
    res = json.load(open(RES)) if osp.exists(RES) else {}
    names = sys.argv[1:] or (['ref'] + list(SPECS))
    for nm in names:
        if nm in res and '--force' not in sys.argv: continue
        if nm == '--force': continue
        r = ref() if nm == 'ref' else evaluate(nm); res[nm] = r
        json.dump(res, open(RES, 'w'), indent=1)
        print(f"{nm:20s} thr {r['thr']:.3f} val {r.get('val_pnl', 0)/1e3:6.0f}k | r3b {r['r3b']['pnl']/1e3:6.0f}k "
              f"n {r['r3b']['takes']} mo {r['r3b']['mo1']:+.2f}/{r['r3b']['mo2']:+.2f} | r2h2 {r['r2h2']['pnl']/1e3:6.0f}k "
              f"n {r['r2h2']['takes']} mo {r['r2h2']['mo1']:+.2f}/{r['r2h2']['mo2']:+.2f} | {r.get('sec', 0):.0f}s", flush=True)
