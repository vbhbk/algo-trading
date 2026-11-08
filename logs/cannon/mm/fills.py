# fill markout decomposition from a full-log run: python fills.py LOGDIR ROUND OFFSET_US
import json, sys, numpy as np
sys.path.insert(0, 'logs/cannon/tools')
from load import load
logdir, rnd, off = sys.argv[1], sys.argv[2], int(sys.argv[3])
E, T, ok = load(rnd)
t = E[:, 0] - off
mid = (E[:, 1] + E[:, 2]) / 2
sp = E[:, 2] - E[:, 1]
orders = {}
rows = []
for line in open(f'{logdir}/run.jsonl'):
    d = json.loads(line)
    k = d['kind']
    if k == 'strategy_order_submitted':
        r = d['payload']['request']
        orders[d['payload']['order_id']] = (r['order_type'], r['quantity'], d['timestamp_us'])
    elif k == 'trade':
        p = d['payload']
        if p['taker_owner'] == 'user':
            oid = p['taker_order_id']; sgn = 1 if p['book_side'] == 'ask' else -1; kind = 0
        elif p['maker_owner'] == 'user':
            oid = p['maker_order_id']; sgn = -1 if p['book_side'] == 'ask' else 1; kind = 1
        else:
            continue
        o = orders[oid]
        rows.append((p['timestamp_us'], sgn, p['quantity'], p['price'], kind, o[1], p['timestamp_us'] - o[2]))
R = np.array(rows, dtype=float)
ts = R[:, 0]
def mid_at(x):
    i = np.searchsorted(t, x, side='right') - 1
    return mid[np.clip(i, 0, len(mid) - 1)]
m0 = mid_at(ts); s0 = sp[np.clip(np.searchsorted(t, ts, side='right') - 1, 0, len(t) - 1)]
fin = mid[np.searchsorted(t, t[-1]) - 1]
print('n fills', len(R), 'vol', R[:, 2].sum())
for kind, name in ((0, 'IOC'), (1, 'passive')):
    m = R[:, 4] == kind
    q = R[m, 2]; sg = R[m, 1]; p = R[m, 3]
    out = [f'{name:8s} n {m.sum():6d} vol {q.sum():8.0f}  edge_vs_mid0 {np.sum(sg*q*(m0[m]-p)):9.0f}']
    for h in (0.1, 1, 5, 30):
        mh = mid_at(ts[m] + h * 1e6)
        out.append(f'mo{h}s {np.sum(sg*q*(mh-p)):9.0f}')
    print('  '.join(out))
    for s in (1, 2):
        mm = m & (s0 == s) if s == 1 else m & (s0 >= 2)
        q = R[mm, 2]; sg = R[mm, 1]; p = R[mm, 3]
        print(f'    spread{s}{"+" if s>1 else ""} n {mm.sum()} vol {q.sum():.0f}  mo0 {np.sum(sg*q*(m0[mm]-p)):.0f}  mo1 {np.sum(sg*q*(mid_at(ts[mm]+1e6)-p)):.0f}  mo5 {np.sum(sg*q*(mid_at(ts[mm]+5e6)-p)):.0f}')
np.save(f'{logdir}_fills.npy', R)
