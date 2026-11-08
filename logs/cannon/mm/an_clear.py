import sys, numpy as np
sys.path.insert(0, 'logs/cannon/tools')
from load import load
for rnd, d in (('round3_public', 'logs/cannon/mm/full_base_round3_a'), ('round2_public', 'logs/cannon/mm/full_base_round2_a')):
    E, T, ok = load(rnd); t = E[:, 0]; mid = (E[:, 1] + E[:, 2]) / 2
    R = np.load(d + '_fills.npy'); P = R[R[:, 4] == 1]
    i = np.clip(np.searchsorted(t, P[:, 0], side='right') - 1, 0, len(t) - 1)  # book after the event (market-only)
    bid, ask = E[i, 1], E[i, 2]
    sg, q, p = P[:, 1], P[:, 2], P[:, 3]
    clear = np.where(sg < 0, ask > p, bid < p)
    def at(x): return mid[np.clip(np.searchsorted(t, x, side='right') - 1, 0, len(t) - 1)]
    print('==', rnd)
    for nm, m in (('cleared', clear), ('partial', ~clear)):
        r = [f'{nm:8s} n {m.sum():5d} vol {q[m].sum():7.0f}']
        for h in (0.1, 1, 5):
            r.append(f'mo{h} {np.sum(sg[m]*q[m]*(at(P[m,0]+h*1e6)-p[m]))/q[m].sum():+.3f}')
        print('  '.join(r))
