"""Fast screening simulator for impulse-follower ideas on recorded tapes (IOC at touch, 5 ms latency)."""
import numpy as np, sys
sys.path.insert(0, 'logs/cannon/tools')
from load import load, takers
_C = {}
def data(r):
    if r not in _C:
        E, T, ok = load(r); K = takers(T)
        _C[r] = (E, T, K)
    return _C[r]

def run_targets(E, dec_t, dec_tg, t0, t1, lat=5000, maxq=10**9, defer=0, maxsp=1):
    """dec_t sorted decision times (us), dec_tg target at/after that time. Orders: IOC at the touch arriving
    at first event >= send+5 ms, filled at the touch of the book just before it, limited by L1 size.
    One IOC in flight at a time; after it lands the residual gap to the current target is re-sent."""
    t = E[:, 0]; bid = E[:, 1]; ask = E[:, 2]; bq = E[:, 3]; aq = E[:, 4]
    pos = 0; cash = 0.0; vol = 0; ntr = 0; abst = 0.0; last = t0
    A = None; cur = 0; cost = [0.0]; spc = [0, 0, 0, 0]
    def send(d):
        nonlocal pos, cash, vol, ntr, abst, last
        delta = cur - pos
        if delta == 0: return None
        j = np.searchsorted(t, d + lat, 'left')
        if j >= len(t) or t[j] >= t1: return None
        k = j - 1
        if defer and ask[k] - bid[k] > maxsp:
            jj = j; tl = t[j] + defer
            while jj < len(t) and t[jj] <= tl and ask[jj] - bid[jj] > maxsp: jj += 1
            j = np.searchsorted(t, min(t[min(jj, len(t)-1)], tl) + lat, 'left')
            if j >= len(t) or t[j] >= t1: return None
            k = j - 1
        if delta > 0: f = min(delta, int(aq[k]), maxq); px = ask[k]
        else: f = -min(-delta, int(bq[k]), maxq); px = bid[k]
        abst += abs(pos) * (t[j] - last); last = t[j]
        pos += f; cash -= f * px; vol += abs(f); ntr += 1; cost[0] += f * (px - (bid[k] + ask[k]) / 2); spc[min(int(ask[k]-bid[k]),3)] += abs(f)
        return t[j]
    m = (dec_t >= t0) & (dec_t < t1)
    for d, tg in zip(dec_t[m], dec_tg[m]):
        while A is not None and A <= d:
            A = send(A)
        cur = tg
        if A is None:
            A = send(d)
    while A is not None:
        A = send(A)
    k = np.searchsorted(t, t1, 'left') - 1
    abst += abs(pos) * (t1 - last)
    mk = pos * (bid[k] if pos > 0 else ask[k])
    return dict(pnl=round(cash + mk), trades=ntr, vol=vol, avgabs=round(abst / (t1 - t0), 1), endpos=pos, cost=round(cost[0]), spc=spc)
