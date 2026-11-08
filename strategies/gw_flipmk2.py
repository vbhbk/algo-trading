# gw_flip: hold +-CAP in the predicted direction (hysteresis on ridge 0.25 s / 0.5 s, fit on round3_a), flip by IOC
from bisect import bisect_left, bisect_right
from math import exp, log1p

from simulator.models import OrderRequest, OrderStatus, OrderType, Side
from simulator.strategy import BaseStrategy

CAP = 95
ENTER = 0.8
KEEP = 0.0
IOC_COOLDOWN_US = 20000
SLIP = 0
A25 = 1.0
A50 = 0.0
PASSIVE = True
PASS_EN = 0.15
PASS_EDGE = 0.15
PASS_LOTS = 190
ADOPT = 5
PULL_EDGE = 0.1
M50 = (-0.0024703699138938287, (0.02358688813985808, -0.02820099392440792, -0.0056743795851622175, 0.01563764217463773, -0.0019193354626532912, 1.2393639507820633, -0.45709321624820626, 0.6546831649102781, -0.20507605576526258, -0.02656970844379486, 0.23091596977866846, -0.03089731520715777, -0.0030027934248067017, -4.010471265524332e-05, 0.03151429648605883, 0.017366046043304315, -0.018119345657766423, -0.07845721552591961, 0.05253869558662455, -2.259117130238863e-05, 0.0, -0.1569144309187313, 0.04606821251074546, 0.8522781088448615, -0.5778579642657877, 0.6172753014788009, 0.016132999726516373, -0.027149529803769154, -0.11019504910251605, 0.0791820809896692, 0.03890771054026226, 0.195703813975891, -0.07845721552593576))


class DirectionModel:
    # ridge H=0.25s, spread 1, alpha 10, fit on the whole round2_public.csv (analysis/direction/export_model.py, fit_frac 1.0)
    HORIZON_S = 0.25
    INTERCEPT = -0.05911016745658127
    W = (0.021281234646839908, -0.03437082918481743, 0.002744184634598198, 0.012422648741952763, -0.0032112321787145124, 0.864825134933366, -0.22899211210268636, 0.3163631335462487, -0.11329188510933987, -0.005295475358277676, 0.1567401874502311, -0.029886497024441714, 0.0012316434934282227, -7.539908477111756e-05, 0.02225727147959171, 0.015601410178357332, -0.00610087097154535, -0.0666357323099817, 0.07708217186044419, 0.0009014558182803367, 0.0, -0.13327141179900093, 0.037039587988097644, 0.5655051012090865, -0.13286432555574676, 0.20337721126587852, 0.01685781285534642, -0.019804264146002233, -0.061393496576373704, 0.048453603085032895, 0.021039852465954627, 0.14621569415973848, -0.06663573227087996)
    CLIP = 3.0
    SPREAD_MAX = 1
    PLACE_EDGE = 0.25    # chosen on first 60% of train (worst-of-3-blocks)
    CANCEL_EDGE = 0.0

    def __init__(self):
        self.us = None
        self.bid = self.ask = 0
        self.bq = self.aq = 0
        self.mid = 0.0
        self.o1 = self.o5 = self.o10 = self.o30 = self.o100 = 0.0   # order-flow imbalance, tau 0.1/0.5/1/3/10 s
        self.f1 = self.f5 = self.f10 = self.f30 = self.f100 = 0.0   # signed aggressive volume, same taus
        self.e05 = self.e2 = self.e10 = self.e60 = 0.0              # EMAs of mid (returns = mid - EMA)
        self.started = False
        self.t_f = self.t_o = self.t_ema = self.t_rv = self.t_rb = self.t_ra = 0
        self.sweeps = 0.0
        self.vol = 0.0
        self.rvol = 0.0
        self.rate_b = self.rate_a = 0.0
        self.lb = self.la = 0.0
        self.b_change_us = self.a_change_us = 0
        self.last_aggr = 0.0
        self.x = None
        self.valid = False
        self.pred = 0.0
        self.stale = False
        self.bn = self.an = 0
        self.views = None
        self.on = False   # True when a prediction is live (valid book, spread <= SPREAD_MAX)

    def update(self, snap, trades):
        """O(1) state update; every decayed quantity is decayed lazily (only when it changes or is read)."""
        now = snap.timestamp_us
        prev = self.us
        self.us = now
        if trades:
            signed = 0
            takers = {}
            vol = 0
            for t in trades:
                if t.taker_owner != 'market':
                    continue
                s = 1 if t.book_side == 'ask' else -1
                q = t.quantity
                signed += s * q
                vol += q
                lv = takers.get(t.taker_order_id)
                if lv is None:
                    takers[t.taker_order_id] = [s, t.price, 1, q]
                else:
                    if t.price != lv[1]:
                        lv[1] = t.price
                        lv[2] += 1
                    lv[3] += q
            if takers:
                self._decay_flow(now)
                self.vol += vol * 1e-3
                if signed:
                    f = signed * 1e-3
                    self.f1 += f
                    self.f5 += f
                    self.f10 += f
                    self.f30 += f
                    self.f100 += f
                for lv in takers.values():
                    if lv[2] >= 2:
                        self.sweeps += lv[0]
                    self.last_aggr = lv[0] * log1p(lv[3])
        bids, asks = snap.bids, snap.asks
        bid, bq, bn = _touch(bids, _NEG)
        ask, aq, an = _touch(asks, _POS)
        if not bid or not ask or ask <= bid:
            self.valid = self.on = False
            self.pred = 0.0
            return False
        obid, oask = self.bid, self.ask
        if obid:
            e = 0
            if bid >= obid:
                e += bq
            if bid <= obid:
                e -= self.bq
            if ask <= oask:
                e -= aq
            if ask >= oask:
                e += self.aq
            if e:
                self._decay_ofi(now)
                e *= 1e-3
                self.o1 += e
                self.o5 += e
                self.o10 += e
                self.o30 += e
                self.o100 += e
        mid = 0.5 * (bid + ask)
        if not self.started:
            self.e05 = self.e2 = self.e10 = self.e60 = mid
            self.t_ema = self.t_rv = now
            self.started = True
        elif mid != self.mid:
            self._decay_ema(now)
            self._decay_rvol(now)
            self.rvol += abs(mid - self.mid)
        lb, la = log1p(bq), log1p(aq)
        dt = 0.0 if prev is None else (now - prev) * 1e-6
        if bid != obid:
            self.b_change_us = now
            self.rate_b = 0.0
            self.t_rb = now
        elif dt > 0 and lb != self.lb:
            r = self.rate_b
            if r and self.t_rb != prev:
                r *= exp(-(prev - self.t_rb) * 5e-6)
            dec = exp(-dt / 0.2)
            self.rate_b = dec * r + (1 - dec) * (lb - self.lb) / max(dt, 0.001)
            self.t_rb = now
        if ask != oask:
            self.a_change_us = now
            self.rate_a = 0.0
            self.t_ra = now
        elif dt > 0 and la != self.la:
            r = self.rate_a
            if r and self.t_ra != prev:
                r *= exp(-(prev - self.t_ra) * 5e-6)
            dec = exp(-dt / 0.2)
            self.rate_a = dec * r + (1 - dec) * (la - self.la) / max(dt, 0.001)
            self.t_ra = now
        self.bid, self.ask, self.bq, self.aq, self.mid, self.lb, self.la = bid, ask, bq, aq, mid, lb, la
        self.valid = True
        self.stale = True
        if ask - bid > self.SPREAD_MAX:
            self.pred = 0.0
            self.on = False
            return True
        self.on = True
        self.bn, self.an, self.views = bn, an, (bids, asks)
        return True

    def _decay_flow(self, now):
        if now != self.t_f:
            dt = (now - self.t_f) * 1e-6
            d1, d5, d10, d30, d100 = exp(-dt / 0.1), exp(-dt / 0.5), exp(-dt), exp(-dt / 3.0), exp(-dt / 10.0)
            self.f1 *= d1
            self.f5 *= d5
            self.f10 *= d10
            self.f30 *= d30
            self.f100 *= d100
            self.vol *= d10
            self.sweeps *= exp(-dt / 5.0)
            self.t_f = now

    def _decay_ofi(self, now):
        if now != self.t_o:
            dt = (now - self.t_o) * 1e-6
            self.o1 *= exp(-dt / 0.1)
            self.o5 *= exp(-dt / 0.5)
            self.o10 *= exp(-dt)
            self.o30 *= exp(-dt / 3.0)
            self.o100 *= exp(-dt / 10.0)
            self.t_o = now

    def _decay_ema(self, now):
        if now != self.t_ema:
            dt = (now - self.t_ema) * 1e-6
            m = self.mid
            self.e05 = m + (self.e05 - m) * exp(-dt / 0.5)
            self.e2 = m + (self.e2 - m) * exp(-dt / 2.0)
            self.e10 = m + (self.e10 - m) * exp(-dt / 10.0)
            self.e60 = m + (self.e60 - m) * exp(-dt / 60.0)
            self.t_ema = now

    def _decay_rvol(self, now):
        if now != self.t_rv:
            self.rvol *= exp(-(now - self.t_rv) * 1e-7)
            self.t_rv = now

    def predict(self):
        """Expected mid move over HORIZON_S in ticks; computed lazily once per event, only when asked."""
        if not self.stale or not self.on:
            return self.pred
        self.stale = False
        now = self.us
        self._decay_flow(now)
        self._decay_ofi(now)
        self._decay_ema(now)
        self._decay_rvol(now)
        rb, ra = self.rate_b, self.rate_a
        if rb and self.t_rb != now:
            rb *= exp(-(now - self.t_rb) * 5e-6)
        if ra and self.t_ra != now:
            ra *= exp(-(now - self.t_ra) * 5e-6)
        bid, ask, bq, aq, mid, lb, la = self.bid, self.ask, self.bq, self.aq, self.mid, self.lb, self.la
        bn, an = self.bn, self.an
        bq2, bd = _depth(self.views[0], _NEG)
        aq2, ad = _depth(self.views[1], _POS)
        spread = ask - bid
        imb = (bq - aq) / (bq + aq)
        x = (self.o1, self.o5, self.o10, self.o30, self.o100, self.f1, self.f5, self.f10, self.f30, self.f100,
             mid - self.e05, mid - self.e2, mid - self.e10, mid - self.e60,
             self.sweeps, self.vol, self.rvol * 0.1, imb, (bd - ad) / (bd + ad), (bd + ad) * 1e-3, spread,
             (bid * aq + ask * bq) / (bq + aq) - mid, (bq2 - aq2) / (bq2 + aq2 + 1), (bn - an) / (bn + an),
             lb * 0.1, la * 0.1, rb, ra,
             min(2.0, (now - self.b_change_us) * 1e-6), min(2.0, (now - self.a_change_us) * 1e-6),
             self.last_aggr, imb * abs(imb), imb * spread)
        self.x = x
        p = self.INTERCEPT
        for w, v in zip(self.W, x):
            p += w * v
        c = self.CLIP
        self.pred = c if p > c else -c if p < -c else p
        return self.pred

    def edge(self, sign, price):
        return sign * (self.mid + self.predict() - price)

    def allow(self, sign, price):
        return not self.on or sign * (self.mid + self.predict() - price) > self.PLACE_EDGE

    def keep(self, sign, price):
        return not self.on or sign * (self.mid + self.predict() - price) >= PULL_EDGE


def _POS(v):
    return v.price


def _NEG(v):
    return -v.price


def _touch(views, key):
    """best price, size and number of orders at the best level (market orders only).

    Views are sorted best first, so the level ends where key(view) > key(best) (binary search).
    """
    if not views:
        return 0, 0, 0
    v0 = views[0]
    if v0.owner != 'market':
        b, q1, _, n1, _ = _side_scan(views)
        return b, q1, n1
    j = bisect_right(views, key(v0), key=key)
    q = [v.quantity for v in views[:j] if v.owner == 'market']
    return v0.price, sum(q), len(q)


def _depth(views, key):
    """(size at L2, size within 5 ticks of best), market orders only; equal to _side_scan's values."""
    v0 = views[0]
    if v0.owner != 'market':
        _, _, q2, _, d5 = _side_scan(views)
        return q2, d5
    k0 = key(v0)
    k = bisect_left(views, k0 + 5, key=key)
    j = bisect_right(views, k0, lo=0, hi=k, key=key)
    if j >= k:
        return 0, sum([v.quantity for v in views[:k] if v.owner == 'market'])
    if views[j].owner != 'market':
        _, _, q2, _, d5 = _side_scan(views)
        return q2, d5
    j2 = bisect_right(views, key(views[j]), lo=j, hi=k, key=key)
    return (sum([v.quantity for v in views[j:j2] if v.owner == 'market']),
            sum([v.quantity for v in views[:k] if v.owner == 'market']))


def _side_scan(views):
    """best price, size at L1, size at L2, orders at L1, size within 5 ticks of best (market orders only)."""
    best = p2 = 0
    q1 = q2 = n1 = d5 = 0
    for v in views:
        if v.owner != 'market':
            continue
        p = v.price
        if best == 0:
            best = p
        elif (p - best if p > best else best - p) >= 5:
            break
        d5 += v.quantity
        if p == best:
            q1 += v.quantity
            n1 += 1
        elif p2 == 0 or p == p2:
            p2 = p
            q2 += v.quantity
    return best, q1, q2, n1, d5


class _O:
    __slots__ = ('side', 'sign', 'price', 'rem', 'ioc', 'cxl')

    def __init__(self, side, price, rem, ioc):
        self.side, self.price, self.rem, self.ioc, self.cxl = side, price, rem, ioc, False
        self.sign = 1 if side == Side.BUY else -1


class GwFlip(BaseStrategy):
    """Hold +-CAP in the direction of the short-horizon prediction (hysteresis), flip by IOC; optional passive legs."""

    def __init__(self) -> None:
        super().__init__()
        self.dm = DirectionModel()
        self.pos = 0
        self.cur = 0
        self.orders = {}
        self.ioc_us = -10 ** 15
        self.sig = 0.0

    def exposure(self, sign):
        e = sign * self.pos
        for o in self.orders.values():
            if o.sign == sign:
                e += o.rem
        return e

    def signal(self):
        dm = self.dm
        s = A25 * dm.predict()
        if A50:
            b, w = M50
            p = b
            for wi, v in zip(w, dm.x):
                p += wi * v
            s += A50 * p
        return s

    def on_order_book_change(self, snapshot, trades) -> None:
        dm = self.dm
        dm.update(snapshot, trades)
        if not dm.valid:
            return
        now = snapshot.timestamp_us
        if dm.on:
            s = self.sig = self.signal()
            if s > ENTER:
                self.cur = 1
            elif s < -ENTER:
                self.cur = -1
        else:
            s = self.sig = 0.0
        tgt = self.cur * CAP
        gap = tgt - self.pos
        has_ioc = any(o.ioc for o in self.orders.values())
        # IOC towards target while the signal still supports the move
        if gap and dm.on and not has_ioc and now - self.ioc_us >= IOC_COOLDOWN_US:
            sg = 1 if gap > 0 else -1
            if sg * s >= KEEP:
                side = Side.BUY if sg > 0 else Side.SELL
                # cancel resting orders on the same side first if they block room
                q = min(abs(gap), CAP - self.exposure(sg))
                if q < abs(gap):
                    for oid, o in self.orders.items():
                        if o.sign == sg and not o.ioc and not o.cxl:
                            o.cxl = True
                            self.cancel_order(oid)
                if q > 0:
                    px = dm.ask + SLIP if sg > 0 else dm.bid - SLIP
                    oid = self.send_order(OrderRequest(side=side, price=px, quantity=q, order_type=OrderType.IOC))
                    self.orders[oid] = _O(side, px, q, True)
                    self.ioc_us = now
        if PASSIVE:
            self.passive(s, now)

    def passive(self, s, now):
        dm = self.dm
        # desired passive quote per side: (price, qty) or None
        want = {1: None, -1: None}
        if dm.on and dm.ask - dm.bid == 1:
            for sg in (1, -1):
                px = dm.bid if sg > 0 else dm.ask
                edge = sg * (dm.mid + dm.pred - px)
                lean = sg * s
                if sg * self.cur < 0 or self.cur == 0:
                    ok = lean > PASS_EN            # pre-position for a flip in direction sg
                    lim = CAP
                else:
                    ok = edge > PASS_EDGE          # top up towards target
                    lim = CAP
                if ok:
                    q = lim - sg * self.pos
                    if q > 0:
                        want[sg] = (px, q)
        for oid, o in list(self.orders.items()):
            if o.ioc or o.cxl:
                continue
            w = want[o.sign]
            if w is None or w[0] != o.price:
                o.cxl = True
                self.cancel_order(oid)
            else:
                want[o.sign] = None if o.rem >= min(w[1], PASS_LOTS) * 0 + 1 else w
        for sg in (1, -1):
            w = want[sg]
            if w is None:
                continue
            if any(o.sign == sg and not o.ioc for o in self.orders.values()):
                continue
            q = min(w[1], PASS_LOTS, CAP - self.exposure(sg))
            if q <= 0:
                continue
            side = Side.BUY if sg > 0 else Side.SELL
            oid = self.send_order(OrderRequest(side=side, price=w[0], quantity=q, order_type=OrderType.LIMIT))
            self.orders[oid] = _O(side, w[0], q, False)

    def on_private_order_update(self, event) -> None:
        o = self.orders.get(event.order_id)
        if o is None:
            return
        st = event.status
        if st == OrderStatus.PARTIALLY_FILLED or st == OrderStatus.FILLED:
            self.pos += o.sign * event.filled_quantity
            o.rem = event.remaining_quantity
            if not o.ioc and abs(self.pos) >= CAP - ADOPT:
                self.cur = 1 if self.pos > 0 else -1
        if st in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            if st == OrderStatus.REJECTED and o.cxl and not o.ioc:
                return  # cancel rejected: order already gone or filled; wait for its final event
            del self.orders[event.order_id]
