"""Task 1: characterise spread>1 episodes."""
import numpy as np
R = ['round3_a', 'round3_b', 'round2_public']
def load(f):
    return dict(np.load(f'logs/gw/spread2/feat_{f}.npz'))
for f in R:
    d = load(f); t = d['t'] / 1e6; bid, ask = d['bid'], d['ask']; sp = ask - bid; mid = (bid + ask) / 2
    w = sp > 1
    st = np.flatnonzero(w & ~np.r_[False, w[:-1]]); en = np.flatnonzero(w & ~np.r_[w[1:], False]) + 1  # en = first event after
    if en[-1] >= len(t): st, en = st[:-1], en[:-1]
    st = st[st > 0]; en = en[-len(st):]
    dur = t[en] - t[st]
    pre = (d['rb'][st] + d['ra'][st]) / 2          # mid before the gap opened
    gb, ga = d['gb'][st], d['ga'][st]
    side = np.sign(ga - gb)                           # +1 ask emptied (gap up), -1 bid emptied (gap down), 0 both/neither
    m_open = mid[st]; m_close = mid[en]
    mv_open = m_open - pre; mv_close = m_close - pre; mv_in = m_close - m_open
    print(f'=== {f}: n episodes {len(st)}, rate {len(st)/t[-1]:.2f}/s, time share {np.sum(np.diff(t)[w[:-1]])/t[-1]:.3f}')
    print('  duration s quantiles 10/25/50/75/90/99:', np.round(np.quantile(dur, [.1, .25, .5, .75, .9, .99]), 4), 'mean', round(dur.mean(), 4))
    print('  spread at open: ', {k: round(float(np.mean(sp[st] == k)), 3) for k in [2, 3, 4]}, '>=5', round(float(np.mean(sp[st] >= 5)), 3))
    print('  side at open: ask emptied %.3f bid emptied %.3f both/none %.3f' % (np.mean(side > 0), np.mean(side < 0), np.mean(side == 0)))
    s = side != 0
    # continuation = close mid relative to pre mid in direction of the emptied side
    cont = side[s] * mv_close[s]
    print('  close-mid vs pre-mid, in gap direction: mean %.3f | >0 %.3f =0 %.3f <0 %.3f' % (cont.mean(), np.mean(cont > 0), np.mean(cont == 0), np.mean(cont < 0)))
    ci = side[s] * mv_in[s]
    print('  close-mid vs open-mid (move DURING episode) in gap dir: mean %.3f | >0 %.3f =0 %.3f <0 %.3f; |move| mean %.3f' % (ci.mean(), np.mean(ci > 0), np.mean(ci == 0), np.mean(ci < 0), np.abs(mv_in).mean()))
    # who refilled: compare close touch to open touch
    bfill = bid[en] - bid[st]; afill = ask[st] - ask[en]
    # refill of the emptied side = the emptied side's touch comes back toward pre level
    emp_refill = np.where(side[s] > 0, afill[s], bfill[s]) > 0
    oth_moves = np.where(side[s] > 0, bfill[s], afill[s]) > 0
    print('  emptied side refilled (touch moved back) %.3f | other side followed (moved into gap) %.3f' % (emp_refill.mean(), oth_moves.mean()))
    # future mid after close, in gap direction, relative to close
    for H in [0.5, 2.0]:
        j = np.searchsorted(t, t[en] + H, side='right') - 1
        fm = side[s] * (mid[j[s]] - m_close[s])
        print(f'  mid {H}s after close vs close, in gap dir: mean {fm.mean():+.3f}')
    for H in [0.25, 0.5, 1, 2]:
        j = np.searchsorted(t, t[st] + H, side='right') - 1
        fm = side[s] * (mid[j[s]] - m_open[s]); fp = side[s] * (mid[j[s]] - pre[s])
        print(f'  from open +{H}s: mid vs open-mid {fm.mean():+.3f} (>0 {np.mean(fm>0):.2f} <0 {np.mean(fm<0):.2f}) | vs pre-mid {fp.mean():+.3f}')
    # move sizes: per-episode open->close |move| by spread
    for k in [2, 3]:
        mk = sp[st] == k
        print(f'  spread {k} at open: n {mk.sum()} dur med {np.median(dur[mk]):.4f} |close-open| mean {np.abs(mv_in[mk]).mean():.3f}, |close-pre| mean {np.abs(mv_close[mk]).mean():.3f}')
    mk = sp[st] >= 4
    print(f'  spread>=4 at open: n {mk.sum()} dur med {np.median(dur[mk]):.4f} |close-open| mean {np.abs(mv_in[mk]).mean():.3f}')
    # with aggressor: did a market trade trigger it?
    trig = d['sv'][st] != 0
    print('  opened by aggressive trade %.3f; in gap dir sign agree %.3f' % (trig.mean(), np.mean(np.sign(d['sv'][st][trig & s]) == side[trig & s])))
