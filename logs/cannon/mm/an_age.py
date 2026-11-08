import sys, numpy as np
sys.path.insert(0, 'logs/cannon/tools')
from load import load
for rnd, d, off in (('round3_public', 'logs/cannon/mm/full_base_round3_a', 0), ('round2_public', 'logs/cannon/mm/full_base_round2_a', 0)):
    E, T, ok = load(rnd); t = E[:, 0] - off; mid = (E[:, 1] + E[:, 2]) / 2; sp = E[:, 2] - E[:, 1]
    R = np.load(d + '_fills.npy')
    def at(a, x): return a[np.clip(np.searchsorted(t, x, side='right') - 1, 0, len(t) - 1)]
    P = R[R[:, 4] == 1]
    s0 = at(sp, P[:, 0] - 1)
    print('==', rnd)
    for lo, hi in ((0, 0.05), (0.05, 0.2), (0.2, 1), (1, 5), (5, 1e9)):
        m = (P[:, 6] / 1e6 >= lo) & (P[:, 6] / 1e6 < hi) & (s0 == 1)
        q = P[m, 2]; sg = P[m, 1]; p = P[m, 3]
        r = [f'age {lo}-{hi}s n {m.sum():5d} vol {q.sum():7.0f}']
        for h in (0.1, 1, 5, 30):
            r.append(f'mo{h} {np.sum(sg*q*(at(mid, P[m,0]+h*1e6)-p))/max(q.sum(),1):+.3f}')
        print('  '.join(r))
    # IOC by size
    I = R[R[:, 4] == 0]
    print(' IOC per-lot mo1 %.3f mo5 %.3f' % (np.sum(I[:,1]*I[:,2]*(at(mid,I[:,0]+1e6)-I[:,3]))/I[:,2].sum(), np.sum(I[:,1]*I[:,2]*(at(mid,I[:,0]+5e6)-I[:,3]))/I[:,2].sum()))
