# Shared data / target / offline-sim helpers for the target study (run from repo root, 1 core).
import numpy as np

ROUNDS = ['round3_a', 'round3_b', 'round2_public']
LAT = 0.005


def load(f):
    d = np.load(f'logs/gw/feat/{f}.npz'); ok = ~np.isnan(d['bid'])
    t = d['t'][ok] / 1e6; bid = d['bid'][ok]; ask = d['ask'][ok]
    X = d['X'][ok].astype(np.float32); on = ~np.isnan(X[:, 0])
    return dict(t=t, bid=bid, ask=ask, X=X, on=on, mid=(bid + ask) / 2)


# segments: (round, t0, t1, role)
INNER_TR = [('round3_a', 0, 2100), ('round2_public', 0, 2100)]
INNER_VA = [('round3_a', 2100, 3000), ('round2_public', 2100, 3000)]
FULL_TR = [('round3_a', 0, 3000), ('round2_public', 0, 3000)]
TEST = [('round3_b', 0, 3000.1), ('round2_public', 3000, 6000.1)]


def fwd_idx(t, H):
    return np.searchsorted(t, t + H, side='right') - 1


def range_max(a, lo, hi):
    """max(a[lo..hi]) inclusive for each pair (lo<=hi), sparse table."""
    n = len(a); K = int(np.log2(n)) + 1; st = [a.astype(np.float32)]
    for k in range(1, K):
        p = st[-1]; s = 1 << (k - 1); q = p.copy(); q[:n - s] = np.maximum(p[:n - s], p[s:]); st.append(q)
    ln = np.maximum(hi - lo + 1, 1); k = np.floor(np.log2(ln)).astype(int)
    out = np.empty(len(lo), np.float32)
    for kk in np.unique(k):
        m = k == kk; out[m] = np.maximum(st[kk][lo[m]], st[kk][hi[m] - (1 << kk) + 1])
    return out


def targets(D):
    """per-round dict of targets (NaN where horizon runs past end)."""
    t, bid, ask, mid = D['t'], D['bid'], D['ask'], D['mid']; n = len(t); T = {}
    j = fwd_idx(t, 0.25); T['d25'] = np.clip(mid[j] - mid, -12, 12)
    k = np.minimum(np.searchsorted(t, t + LAT), n - 1)  # execution event of an IOC sent at t
    for H in [0.5, 1, 2]:
        j = fwd_idx(t, H)
        T[f'gl{H}'] = np.clip(mid[j] - ask[k], -12, 12)   # long taker gain per lot
        T[f'gs{H}'] = np.clip(bid[k] - mid[j], -12, 12)   # short taker gain per lot
    for H in [0.5, 1]:
        j = fwd_idx(t, H); lo = np.minimum(np.arange(n) + 1, n - 1); hi = np.maximum(j, lo)
        mx = range_max(mid, lo, hi); mn = -range_max(-mid, lo, hi)
        T[f'up{H}'] = (mx >= mid + 1).astype(np.float32); T[f'dn{H}'] = (mn <= mid - 1).astype(np.float32)
    bad = t > t[-1] - 2.0
    for v in T.values(): v[bad] = np.nan
    return T


def seg_mask(D, t0, t1):
    return (D['t'] >= t0) & (D['t'] < t1)


# ---------------- offline trading sim (same logic as an3.hyst / an4.run2) ----------------
def hyst2(up, dn):
    trig = np.where(up, 1, np.where(dn, -1, 0)); idx = np.where(trig != 0, np.arange(len(trig)), 0)
    idx = np.maximum.accumulate(idx); out = trig[idx]
    if (trig != 0).any(): out[:np.argmax(trig != 0)] = 0
    else: out[:] = 0
    return out


def run2(t, bid, ask, tg, lat=LAT):
    n = len(t); j = np.minimum(np.searchsorted(t, t + lat), n - 1)
    ch = np.flatnonzero(np.diff(np.concatenate([[0], tg])) != 0)
    pos = 0; cash = 0.0; ks = []; ds = []; ps = []
    for i in ch:
        k = j[i]; d = tg[i] - pos
        if d == 0: continue
        px = ask[k] if d > 0 else bid[k]; cash -= d * px; pos = tg[i]; ks.append(k); ds.append(d); ps.append(px)
    mid = (bid + ask) / 2; pnl = cash + pos * mid[-1]
    ks = np.array(ks, int); ds = np.array(ds, float); ps = np.array(ps, float); mo = {}
    for H in [1, 2]:
        if len(ks) == 0: mo[H] = np.nan; continue
        mh = mid[np.minimum(np.searchsorted(t, t[ks] + H, side='right') - 1, n - 1)]
        mo[H] = np.sum(np.sign(ds) * (mh - ps) * np.abs(ds)) / np.abs(ds).sum()
    return pnl, len(ks), mo


def sim(D, m, su, sd, thr, cap=95):
    """long trigger when su>=thr and su>sd, short when sd>=thr and sd>su; hold +-cap by last trigger."""
    t, bid, ask = D['t'][m], D['bid'][m], D['ask'][m]; a, b = su[m], sd[m]
    up = (a >= thr) & (a > b); dn = (b >= thr) & (b > a)
    return run2(t, bid, ask, cap * hyst2(up, dn))
