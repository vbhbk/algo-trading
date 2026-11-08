import numpy as np
for d in ('logs/cannon/mm/full_base_round3_a', 'logs/cannon/mm/full_base_round2_a'):
    R = np.load(d + '_fills.npy')
    ts, sg, q, p, k = R[:, 0], R[:, 1], R[:, 2], R[:, 3], R[:, 4]
    print('==', d)
    # next fill of any kind after each passive fill
    for kind, nm in ((1, 'passive'), (0, 'IOC')):
        idx = np.where(k == kind)[0]
        nxt = idx + 1; nxt = nxt[nxt < len(R)]; idx = idx[:len(nxt)]
        dt = (ts[nxt] - ts[idx]) / 1e6
        opp = sg[nxt] != sg[idx]
        for lo, hi in ((0, 0.2), (0.2, 1), (1, 5), (5, 1e9)):
            m = (dt >= lo) & (dt < hi)
            for nk in (0, 1):
                mm = m & opp & (k[nxt] == nk)
                rt = -sg[idx[mm]] * (p[idx[mm]] - p[nxt[mm]])  # round trip ticks per lot (sell p0, buy p1 -> p0-p1)
                print(f'  {nm} -> opposite {"IOC" if nk==0 else "pas"} within {lo}-{hi}s: n {mm.sum():5d} vol {q[idx[mm]].sum():7.0f} rt/lot {rt.mean() if mm.sum() else 0:+.3f}')
