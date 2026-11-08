# maker markout of market aggressive orders at the touch, split by whether the order cleared the level
import sys, numpy as np
sys.path.insert(0, 'logs/cannon/tools')
from load import load, takers
for rnd in ('round2_public', 'round3_public'):
    E, T, ok = load(rnd)
    t = E[:, 0]; mid = (E[:, 1] + E[:, 2]) / 2
    K = takers(T)
    ei = K[:, 4].astype(int) - 1  # book before the taker's event
    ei = np.clip(ei, 0, len(E) - 1)
    bid, ask = E[ei, 1], E[ei, 2]; sp = ask - bid
    s = K[:, 1]; q = K[:, 2]; lv = K[:, 3]
    l1 = np.where(s > 0, E[ei, 4], E[ei, 3])
    touch = np.where(s > 0, ask, bid)
    def mid_at(x):
        return mid[np.clip(np.searchsorted(t, x, side='right') - 1, 0, len(t) - 1)]
    for half, hm in (('a', K[:, 0] < 3000e6), ('b', K[:, 0] >= 3000e6)):
        print(f'== {rnd} {half}  takers {hm.sum()}  aggr vol {q[hm].sum():.0f}')
        for h in (0.1, 1, 5):
            m1 = mid_at(K[:, 0] + h * 1e6)
            mo = -s * (m1 - touch)  # per lot markout for a maker filled at the touch
            for name, m in (('sp1 partial (q<L1)', (sp == 1) & (q < l1)), ('sp1 clears L1', (sp == 1) & (q >= l1)),
                            ('sp2 partial', (sp == 2) & (q < l1)), ('sp2 clears', (sp == 2) & (q >= l1))):
                m = m & hm
                fq = np.minimum(q, l1)[m]
                print(f'  h{h:>4} {name:20s} n {m.sum():6d} touch-vol {fq.sum():8.0f} mo/lot {np.sum(mo[m]*fq)/fq.sum():+.3f}')
