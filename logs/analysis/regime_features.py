"""Rebuild L1 book from tape, compute regime features per dataset. Usage: python regime_features.py OUT.json file1.csv ..."""
import csv, json, sys, heapq
from collections import deque
import numpy as np

def replay(path):
    bids, asks = {}, {}          # price -> deque[[oid, vol]]
    bvol, avol = {}, {}          # price -> total vol
    orders = {}                  # oid -> [side, price, entry]
    bh, ah = [], []              # heaps (lazy)
    snaps = []                   # t, bb, ba, bq, aq
    trades = []                  # t, sign, size_filled, levels, mid_before
    n_ins = n_can = n_ioc = 0
    def best_bid():
        while bh and bvol.get(-bh[0], 0) <= 0: heapq.heappop(bh)
        return -bh[0] if bh else None
    def best_ask():
        while ah and avol.get(ah[0], 0) <= 0: heapq.heappop(ah)
        return ah[0] if ah else None
    with open(path) as f:
        r = csv.reader(f); next(r)
        last_t = -1
        for row in r:
            t = float(row[0]); op = row[1]
            if op == 'Cancel':
                n_can += 1
                o = orders.pop(row[2], None)
                if o is not None:
                    side, p, ent = o
                    vm = bvol if side == 'B' else avol
                    vm[p] -= ent[1]; ent[1] = 0
            else:
                n_ins += 1
                side = row[3]; v = int(row[4]); p = int(row[5]); ioc = row[6] == 'F'
                if ioc: n_ioc += 1
                bb, ba = best_bid(), best_ask()
                mid0 = (bb + ba) / 2 if bb is not None and ba is not None else None
                filled = 0; lv = 0
                if side == 'B':
                    while v > 0:
                        ba = best_ask()
                        if ba is None or ba > p: break
                        lv += 1; q = asks[ba]
                        while v > 0 and avol[ba] > 0:
                            ent = q[0]
                            if ent[1] == 0: q.popleft(); continue
                            x = min(v, ent[1]); ent[1] -= x; avol[ba] -= x; v -= x; filled += x
                            if ent[1] == 0: q.popleft(); orders.pop(ent[0], None)
                else:
                    while v > 0:
                        bb = best_bid()
                        if bb is None or bb < p: break
                        lv += 1; q = bids[bb]
                        while v > 0 and bvol[bb] > 0:
                            ent = q[0]
                            if ent[1] == 0: q.popleft(); continue
                            x = min(v, ent[1]); ent[1] -= x; bvol[bb] -= x; v -= x; filled += x
                            if ent[1] == 0: q.popleft(); orders.pop(ent[0], None)
                if filled > 0 and mid0 is not None:
                    trades.append((t, 1 if side == 'B' else -1, filled, lv, mid0, int(ioc)))
                if v > 0 and not ioc:
                    ent = [row[2], v]
                    if side == 'B':
                        if p not in bids: bids[p] = deque(); bvol[p] = 0
                        if bvol[p] <= 0: heapq.heappush(bh, -p)
                        bids[p].append(ent); bvol[p] += v
                    else:
                        if p not in asks: asks[p] = deque(); avol[p] = 0
                        if avol[p] <= 0: heapq.heappush(ah, p)
                        asks[p].append(ent); avol[p] += v
                    orders[row[2]] = [side, p, ent]
            bb, ba = best_bid(), best_ask()
            if bb is not None and ba is not None:
                snaps.append((t, bb, ba, bvol[bb], avol[ba]))
    return np.array(snaps, float), np.array(trades, float), dict(ins=n_ins, can=n_can, ioc=n_ioc)

def grid(snaps, dt):
    T = snaps[-1, 0]; g = np.arange(dt, T, dt)
    idx = np.searchsorted(snaps[:, 0], g, side='right') - 1
    ok = idx >= 0
    return g[ok], snaps[idx[ok]]

def ac1(x):
    x = x - x.mean(); d = (x * x).sum()
    return float((x[1:] * x[:-1]).sum() / d) if d > 0 else float('nan')

def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or a[m].std() == 0 or b[m].std() == 0: return float('nan')
    return float(np.corrcoef(a[m], b[m])[0, 1])

def features(path):
    snaps, tr, cnt = replay(path)
    out = {'file': path.split('/')[-1].replace('.csv', ''), 'T': float(snaps[-1, 0])}
    g, s = grid(snaps, 0.1)
    mid = (s[:, 1] + s[:, 2]) / 2; spr = s[:, 2] - s[:, 1]
    out['n_events'] = cnt['ins'] + cnt['can']
    out['cancel_share'] = cnt['can'] / (cnt['ins'] + cnt['can'])
    out['ioc_share'] = cnt['ioc'] / cnt['ins']
    out['range_total'] = float(mid.max() - mid.min())
    out['spread_mean'] = float(spr.mean())
    out['spr1'] = float((spr <= 1).mean()); out['spr2'] = float((spr == 2).mean()); out['spr3p'] = float((spr >= 3).mean())
    out['l1_vol_med'] = float(np.median(s[:, 3] + s[:, 4]) / 2)
    out['l1_vol_mean'] = float((s[:, 3] + s[:, 4]).mean() / 2)
    # moves at 10s / 60s / 600s
    for h, name in ((100, '10s'), (600, '60s'), (6000, '10m')):
        d = mid[h:] - mid[:-h]
        out[f'absmove_{name}_mean'] = float(np.abs(d).mean())
        out[f'absmove_{name}_q95'] = float(np.quantile(np.abs(d), 0.95))
    # 1s grid
    m1 = mid[9::10]; d1 = np.diff(m1)
    out['rv_1s_per_sqrt_s'] = float(d1.std())
    out['rv_1h'] = float(d1.std() * 60)
    out['jumps5_per_1000s'] = float((np.abs(d1) >= 5).sum() / out['T'] * 1000)
    # 10-min windows
    W = 6000; nw = len(mid) // W
    wr = [float(mid[i*W:(i+1)*W].max() - mid[i*W:(i+1)*W].min()) for i in range(nw)]
    wn = [float(mid[(i+1)*W-1] - mid[i*W]) for i in range(nw)]
    wv = [float(np.diff(mid[i*W:(i+1)*W][9::10]).std()) for i in range(nw)]
    out['win10m_range_med'] = float(np.median(wr)); out['win10m_range_max'] = float(max(wr))
    out['win10m_net_absmax'] = float(max(abs(x) for x in wn)); out['win10m_rv_min'] = float(min(wv)); out['win10m_rv_max'] = float(max(wv))
    out['win10m_rows'] = [dict(range=a, net=b, rv1s=c) for a, b, c in zip(wr, wn, wv)]
    # autocorr of increments at 1s, 5s, 30s
    for k, name in ((10, '1s'), (50, '5s'), (300, '30s')):
        mk = mid[k-1::k]; out[f'ac_{name}'] = ac1(np.diff(mk))
    # variance ratio 30s vs 1s
    out['var_ratio_30_1'] = float(np.var(np.diff(mid[299::300])) / (30 * np.var(d1)))
    # trades
    out['trades_per_min'] = float(len(tr) / out['T'] * 60)
    out['ioc_trades_per_min'] = float(tr[:, 5].sum() / out['T'] * 60)
    out['trade_size_med'] = float(np.median(tr[:, 2])); out['trade_size_mean'] = float(tr[:, 2].mean())
    out['multi_level_share'] = float((tr[:, 3] >= 2).mean())
    out['flow_lots_per_s'] = float(tr[:, 2].sum() / out['T'])
    # markouts after trades, by size bucket
    tt = tr[:, 0]
    def mid_at(ts):
        i = np.searchsorted(snaps[:, 0], ts, side='right') - 1
        i = np.clip(i, 0, len(snaps) - 1)
        return (snaps[i, 1] + snaps[i, 2]) / 2
    after0 = mid_at(tt + 1e-6)
    out['markout'] = {}
    buckets = [('lt30', tr[:, 2] < 30), ('30_100', (tr[:, 2] >= 30) & (tr[:, 2] < 100)), ('100_300', (tr[:, 2] >= 100) & (tr[:, 2] < 300)), ('ge300', tr[:, 2] >= 300), ('all', np.ones(len(tr), bool))]
    for h in (0, 1, 5, 30):
        mh = mid_at(tt + h) if h > 0 else after0
        sm = tr[:, 1] * (mh - tr[:, 4])
        sm[tt + h > out['T']] = np.nan
        for bn, mask in buckets:
            out['markout'][f'{bn}_{h}s'] = float(np.nanmean(sm[mask])) if mask.sum() else float('nan')
    for bn, mask in buckets: out['markout'][f'{bn}_n'] = int(mask.sum())
    # predictability: imb_L1 and signed flow (last 1s) vs future mid change 1s,5s on 0.1s grid sampled every 1s
    imb = (s[:, 3] - s[:, 4]) / (s[:, 3] + s[:, 4])
    cs = np.concatenate([[0], np.cumsum(tr[:, 1] * tr[:, 2])])
    def flow_win(ts, w):
        a = np.searchsorted(tt, ts - w, side='right'); b = np.searchsorted(tt, ts, side='right'); return cs[b] - cs[a]
    sel = np.arange(0, len(g), 10)
    for h, name in ((10, '1s'), (50, '5s')):
        ok = sel[sel + h < len(g)]
        fut = mid[ok + h] - mid[ok]
        out[f'corr_imb_{name}'] = corr(imb[ok], fut)
        out[f'corr_imb_spr1_{name}'] = corr(np.where(spr[ok] <= 1, imb[ok], np.nan), fut)
        out[f'corr_flow1s_{name}'] = corr(flow_win(g[ok], 1.0), fut)
        out[f'corr_flow5s_{name}'] = corr(flow_win(g[ok], 5.0), fut)
    return out

if __name__ == '__main__':
    from multiprocessing import Pool
    with Pool(int(sys.argv[2])) as p:
        res = p.map(features, sys.argv[3:])
    json.dump(res, open(sys.argv[1], 'w'), indent=1)
