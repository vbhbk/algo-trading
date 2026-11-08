# max_r2_ml with both ridge models refit on round3_a (logs/max/r3/fit.py)
"""max_r2_fast + short-horizon direction model (ridge, from strategy_v5) for gating, taking and unwinding."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from math import exp, log1p

from collections import deque

from simulator.models import (
    OrderBookSnapshot,
    OrderRequest,
    OrderStatus,
    OrderType,
    PrivateOrderEvent,
    Side,
    Trade,
)
from simulator.strategy import BaseStrategy

QUOTE_SIZE = 200
MAX_POS = 95  # inventory + resting same-side size never exceeds this
SKEW_STEP = 1000  # 1000 = no skew
IMPROVE_SPREAD = 3
UNWIND_TH = 1
UNWIND_FULL = 10.0
TOPUP_MIN = 10
L2_OFFSET = 0  # 0 = off
L2_SIZE = 20
TGT_MK = 0.0  # 0 = off
TGT_MAX = 30
SZ_K = 0.0  # 0 = off
VOL_WIN_US = 5_000_000  # look-back for the mid move
VOL_TH = 10  # |mid move| in ticks over the window that switches to the guarded cap (1e9 = off)
VOL_HOLD_US = 10_000_000  # stay guarded this long after the last trigger
VOL_POS = 0  # position cap while guarded
# queue (e5bba41: KEEP_TICKS=1, SPLIT_LEAD=30, KEEP_SHAVE=-30)
KEEP_TICKS = 0  # keep resting orders up to this many ticks behind the quote price (0 = off)
KEEP_SHAVE = -30  # <0: keep one order <= -KEEP_SHAVE lots per deeper price; >1: shrink deeper orders to it
SPLIT_LEAD = 0  # >0: a new quote goes out as SPLIT_LEAD lots + the rest (0 = off)
# signal (600c106: MOD_IMB_TH=-0.92)
FAST_WIN_US = 1_000_000
FAST_TH = 6  # |mid move| in ticks over 1 s that also triggers the guard (1000 = off)
DIR_ON = 1  # 1: compute the direction model on every event
DIR_GATE_EDGE = 0.25  # new passive order only if model edge > this (ticks; -1e9 = off)
DIR_TAKE_Z = 0.5  # |prediction| >= this -> IOC at the touch in the predicted direction (1e9 = off)
DIR_TAKE_LOTS = 60
DIR_TAKE_CD_US = 250_000
DIR_TAKE_GUARD = 1  # 1: taking allowed while the volatility guard is on
DIR_UNWIND_Z = 0.8  # sign(inv) * prediction < -this -> IOC on the reducing side (1e9 = off)
DIR_UNWIND_LOTS = 95
DIR_UNWIND_CD_US = 100_000
PULL_EDGE = 0.1
BG_ON = 0  # book guard: spread >= BG_SPREAD, depth within 5 ticks < BG_DEPTH or 1 s aggressive volume > BG_FLOW -> cap 0
BG_SPREAD = 3
BG_DEPTH = 1700
BG_FLOW = 2000
BG_PAUSE_US = 2_000_000
DIR_PULL_EDGE = 0.1  # cancel a resting quote when model edge < this (ticks; -1e9 = off)
LONG_MODELS = {
    2: (0.3905465144019048, (0.008222996599788496, -0.011939458097696696, -0.043527465674766654, -0.009954262010378107, 0.0379228608724918, 1.6101108667996815, -1.467530062254439, 2.267543678967845, -0.7760433205237406, -0.10663502419219674, 0.360617766504242, -0.04335646055549179, -0.007380828697768852, -0.004565764757071, -0.025592980510277037, -0.06588188223895934, -0.0879866443956433, -0.22139152690237018, 0.009060210916717158, -0.022444136876037662, 0.0, -0.4427830535369407, 0.019574393269930566, 1.1163618633627233, -1.8024573012105534, 1.8348010353891973, 0.026044356702801683, -0.029985546098892417, -0.1999355937232568, 0.10831243869589456, 0.05310084692789269, 1.0351065294496722, -0.22139152690235744)),
}
LONG_H = 2  # long-horizon ridge (s), weights from strategy_v8, fit on the first 60% of round2_public_train
LONG_K = 200.0  # target inventory = clip(LONG_K * long prediction, +-MAX_POS) (0 = off)
LONG_UNWIND = 1  # 1: the unwind IOC steers inventory to the target instead of 0
LONG_LIMIT = 0  # 1: the side moving inventory away from the target quotes at most DIR_CARRY_SLACK lots
JUMP_TH = 1e9  # |mid move| over 1 s that starts riding the jump (ticks; 1e9 = off)
JUMP_HOLD_US = 20_000_000  # max ride time
JUMP_STOP = 10  # exit when the price retraces this many ticks from the ride's best
JUMP_SLIP = 3  # IOC price this many ticks through the touch
DIR_CARRY = 0.0  # target inventory = DIR_CARRY * prediction (lots per tick, 0 = off)
DIR_CARRY_SLACK = 0  # extra lots the side moving away from the target may still quote
MOD_IMB_TH = -0.92  # side_sign * imb_L1 <= this -> side capped at flattening size via modify (<= -1 = off)
MOD_CAP_INC = 0  # increasing exposure still allowed when triggered


class DirectionModel:
    # ridge H=0.25s, spread 1, alpha 10, fit on round2_public_train first 60% (analysis/direction/export_model.py)
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



class MaxR3Ml(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.inventory = 0
        self.orders: dict[str, list] = {}  # id -> [side, price, remaining]
        self.cancelling: set[str] = set()
        self.modifying: dict[str, int] = {}
        self.cap = MAX_POS
        self.mids: deque = deque()
        self.guard_until = -1
        self.fast_mids: deque = deque()
        self.dm = DirectionModel()
        self.iocs: dict[str, list] = {}  # id -> [sign, remaining]
        self.take_us = -(10**18)
        self.target = 0
        self.jmids: deque = deque()
        self.jdir = 0
        self.jend = 0
        self.jbest = 0.0
        self.bg_flow: deque = deque()
        self.bg_sum = 0
        self.bg_until = -1
        self.unwind_us = -(10**18)

    def _cancel(self, oid: str) -> None:
        if oid not in self.cancelling:
            self.cancelling.add(oid)
            self.cancel_order(oid)

    def _modify(self, oid: str, new_qty: int) -> None:
        if new_qty <= 0:
            self._cancel(oid)
        elif oid not in self.cancelling and self.modifying.get(oid) != new_qty:
            self.modifying[oid] = new_qty
            self.modify_order(oid, new_qty)

    def _ioc_pending(self, sign: int) -> int:
        return sum(o[1] for o in self.iocs.values() if o[0] == sign)

    def _resting(self, side: Side) -> int:
        return sum(o[2] for o in self.orders.values() if o[0] == side)

    def _ioc(self, sign: int, price: int, qty: int) -> None:
        side = Side.BUY if sign > 0 else Side.SELL
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=qty, order_type=OrderType.IOC))
        self.iocs[oid] = [sign, qty]

    def _quote_side(self, side: Side, px: int | None, size: int, l2_size: int, max_keep: int = 1000) -> None:
        live = [oid for oid, o in self.orders.items() if o[0] == side and oid not in self.cancelling]
        want: list[tuple[int, int]] = []
        if px is not None:
            if size > 0:
                want.append((px, size))
            if L2_OFFSET and l2_size > 0:
                want.append((px - L2_OFFSET if side == Side.BUY else px + L2_OFFSET, l2_size))
        prices = {p for p, _ in want}
        kept: dict[int, list[str]] = {}
        deeper: list[str] = []
        pull = DIR_PULL_EDGE > -1e9 and self.dm.on
        for oid in live:
            p = self.orders[oid][1]
            if pull and (1 if side == Side.BUY else -1) * (self.dm.mid + self.dm.predict() - p) < DIR_PULL_EDGE:
                self._cancel(oid)
                continue
            if p in prices:
                kept.setdefault(p, []).append(oid)
            elif KEEP_TICKS and want and 0 < (px - p if side == Side.BUY else p - px) <= KEEP_TICKS:
                deeper.append(oid)
            else:
                self._cancel(oid)
        if not want:
            return
        if deeper:
            if KEEP_SHAVE < 0:
                seen: set[int] = set()
                for oid in deeper:
                    dp = self.orders[oid][1]
                    if dp not in seen and self.orders[oid][2] <= -KEEP_SHAVE:
                        seen.add(dp)
                    else:
                        self._cancel(oid)
            elif KEEP_SHAVE > 1:
                for oid in deeper:
                    if self.orders[oid][2] > KEEP_SHAVE:
                        self._modify(oid, KEEP_SHAVE)
        if max_keep < 1000 and px in kept:
            oids = kept[px]
            rem = sum(self.orders[oid][2] for oid in oids) - max_keep
            for oid in reversed(list(oids)):
                if rem <= 0:
                    break
                q = self.orders[oid][2]
                if q <= rem:
                    self._cancel(oid)
                    oids.remove(oid)
                    rem -= q
                elif oid not in self.modifying:
                    self._modify(oid, q - rem)
                    rem = 0
        sign = 1 if side == Side.BUY else -1
        room = self.cap - sign * self.inventory - sum(o[2] for o in self.orders.values() if o[0] == side) - self._ioc_pending(sign)
        for p, sz in want:
            oids = kept.get(p, [])
            have = sum(self.orders[oid][2] for oid in oids)
            add = sz - have if not oids or have <= sz - TOPUP_MIN else 0
            add = min(add, room)
            if add <= 0:
                continue
            if SPLIT_LEAD and not oids and add > SPLIT_LEAD:
                parts = (SPLIT_LEAD, add - SPLIT_LEAD)
            else:
                parts = (add,)
            if DIR_GATE_EDGE > -1e9 and self.dm.on and sign * (self.dm.mid + self.dm.predict() - p) <= DIR_GATE_EDGE:
                continue
            for q in parts:
                oid = self.send_order(OrderRequest(side=side, price=p, quantity=q, order_type=OrderType.LIMIT))
                self.orders[oid] = [side, p, q]
            room -= add

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        if DIR_ON:
            self.dm.update(snapshot, trades)
            if LONG_K and self.dm.on:
                self.dm.predict()
                b, w = LONG_MODELS[LONG_H]
                pl = b
                for wi, v in zip(w, self.dm.x):
                    pl += wi * v
                t = int(round(LONG_K * pl))
                self.target = MAX_POS if t > MAX_POS else -MAX_POS if t < -MAX_POS else t
            elif LONG_K:
                self.target = 0
        bid = next((o.price for o in snapshot.bids if o.owner == 'market'), None)
        ask = next((o.price for o in snapshot.asks if o.owner == 'market'), None)
        if bid is None or ask is None:
            self._quote_side(Side.BUY, None, 0, 0)
            self._quote_side(Side.SELL, None, 0, 0)
            return

        spread = ask - bid
        inv = self.inventory
        if JUMP_TH < 1e9:
            now_j = snapshot.timestamp_us
            mj = 0.5 * (bid + ask)
            jm = self.jmids
            jm.append((now_j, mj))
            while now_j - jm[0][0] > 1_000_000:
                jm.popleft()
            if self.jdir == 0:
                mv = mj - jm[0][1]
                if abs(mv) >= JUMP_TH:
                    self.jdir = 1 if mv > 0 else -1
                    self.jend = now_j + JUMP_HOLD_US
                    self.jbest = mj
            elif abs(self.jdir) == 1:
                if self.jdir * (mj - self.jbest) > 0:
                    self.jbest = mj
                if now_j > self.jend or self.jdir * (self.jbest - mj) >= JUMP_STOP:
                    self.jdir *= 2
        if VOL_TH < 1e9:
            now = snapshot.timestamp_us
            m = 0.5 * (bid + ask)
            mids = self.mids
            mids.append((now, m))
            while now - mids[0][0] > VOL_WIN_US:
                mids.popleft()
            if abs(m - mids[0][1]) >= VOL_TH:
                self.guard_until = now + VOL_HOLD_US
            if FAST_TH < 1000:
                fm = self.fast_mids
                fm.append((now, m))
                while now - fm[0][0] > FAST_WIN_US:
                    fm.popleft()
                if abs(m - fm[0][1]) >= FAST_TH:
                    self.guard_until = max(self.guard_until, now + VOL_HOLD_US)
            self.cap = VOL_POS if now < self.guard_until else MAX_POS
        if BG_ON:
            now_b = snapshot.timestamp_us
            for t in trades:
                if t.taker_owner == 'market':
                    self.bg_flow.append((t.timestamp_us, t.quantity))
                    self.bg_sum += t.quantity
            while self.bg_flow and self.bg_flow[0][0] < now_b - 1_000_000:
                self.bg_sum -= self.bg_flow.popleft()[1]
            danger = spread >= BG_SPREAD or self.bg_sum > BG_FLOW
            if not danger:
                depth = 0
                for o in snapshot.bids:
                    if o.owner == 'market':
                        if bid - o.price >= 5:
                            break
                        depth += o.quantity
                for o in snapshot.asks:
                    if o.owner == 'market':
                        if o.price - ask >= 5:
                            break
                        depth += o.quantity
                danger = depth < BG_DEPTH
            if danger:
                self.bg_until = now_b + BG_PAUSE_US
            if now_b < self.bg_until:
                self.cap = VOL_POS
        cap = self.cap
        tgt = 0.0
        x = 0.0
        if TGT_MK or SZ_K:
            bv = sum(o.quantity for o in snapshot.bids if o.owner == 'market' and o.price == bid)
            av = sum(o.quantity for o in snapshot.asks if o.owner == 'market' and o.price == ask)
            if bv + av > 0:
                x = (bv - av) / (bv + av)
                micro = (bid * av + ask * bv) / (bv + av)
                tgt = max(-TGT_MAX, min(TGT_MAX, TGT_MK * (micro - 0.5 * (bid + ask)) / spread))

        skew = int((inv - tgt) / SKEW_STEP)
        imp = 1 if spread >= IMPROVE_SPREAD else 0
        bid_px = min(bid + imp - max(skew, 0), ask - 1)
        ask_px = max(ask - imp - min(skew, 0), bid + 1)

        bq = max(0, round(QUOTE_SIZE * (1 + SZ_K * x)))
        aq = max(0, round(QUOTE_SIZE * (1 - SZ_K * x)))
        buy_size = min(bq, cap - inv)
        sell_size = min(aq, cap + inv)
        if UNWIND_FULL and inv >= UNWIND_TH:
            sell_size = min(max(sell_size, round(UNWIND_FULL * inv)), cap + inv)
        elif UNWIND_FULL and -inv >= UNWIND_TH:
            buy_size = min(max(buy_size, round(UNWIND_FULL * -inv)), cap - inv)

        keep_buy = keep_sell = 1000
        if MOD_IMB_TH > -1.0:
            bv = sum(o.quantity for o in snapshot.bids if o.owner == 'market' and o.price == bid)
            av = sum(o.quantity for o in snapshot.asks if o.owner == 'market' and o.price == ask)
            imb = (bv - av) / (bv + av) if bv + av > 0 else 0.0
            if imb <= MOD_IMB_TH:
                keep_buy = max(0, -inv) + MOD_CAP_INC
                buy_size = min(buy_size, keep_buy)
            if -imb <= MOD_IMB_TH:
                keep_sell = max(0, inv) + MOD_CAP_INC
                sell_size = min(sell_size, keep_sell)

        if DIR_ON and ((DIR_CARRY and self.dm.on) or (LONG_K and LONG_LIMIT)):
            tgt = self.target if LONG_K and LONG_LIMIT else max(-MAX_POS, min(MAX_POS, DIR_CARRY * self.dm.predict()))
            if tgt > 0:
                lim = max(0, int(inv - tgt)) + DIR_CARRY_SLACK
                keep_sell = min(keep_sell, lim)
                sell_size = min(sell_size, lim)
            elif tgt < 0:
                lim = max(0, int(tgt - inv)) + DIR_CARRY_SLACK
                keep_buy = min(keep_buy, lim)
                buy_size = min(buy_size, lim)
        if JUMP_TH < 1e9 and self.jdir:
            jd = self.jdir
            if abs(jd) == 1:
                if jd > 0:
                    keep_sell = 0
                    sell_size = 0
                else:
                    keep_buy = 0
                    buy_size = 0
                if not self.iocs:
                    side_j = Side.BUY if jd > 0 else Side.SELL
                    free = MAX_POS - jd * inv - self._resting(side_j) - self._ioc_pending(jd)
                    if free > 0:
                        self._ioc(jd, ask + JUMP_SLIP if jd > 0 else bid - JUMP_SLIP, free)
            else:
                if inv == 0:
                    self.jdir = 0
                elif not self.iocs:
                    sg = 1 if inv > 0 else -1
                    side_r = Side.SELL if inv > 0 else Side.BUY
                    q = min(abs(inv), MAX_POS + abs(inv) - self._resting(side_r) - self._ioc_pending(-sg))
                    if q > 0:
                        self._ioc(-sg, bid - JUMP_SLIP if inv > 0 else ask + JUMP_SLIP, q)
            self._quote_side(Side.BUY, bid_px, buy_size, 0, keep_buy)
            self._quote_side(Side.SELL, ask_px, sell_size, 0, keep_sell)
            return
        take_sign = 0
        if DIR_ON and DIR_TAKE_Z < 1e9 and self.dm.on and abs(self.dm.predict()) >= DIR_TAKE_Z:
            take_sign = 1 if self.dm.pred > 0 else -1
            if DIR_TAKE_GUARD or cap == MAX_POS:
                lim = max(0, MAX_POS - take_sign * inv - DIR_TAKE_LOTS - self._ioc_pending(take_sign))
                if take_sign > 0:
                    keep_buy = min(keep_buy, lim)
                    buy_size = min(buy_size, lim)
                else:
                    keep_sell = min(keep_sell, lim)
                    sell_size = min(sell_size, lim)
            else:
                take_sign = 0

        self._quote_side(Side.BUY, bid_px, buy_size, min(L2_SIZE, cap - inv - buy_size), keep_buy)
        self._quote_side(Side.SELL, ask_px, sell_size, min(L2_SIZE, cap + inv - sell_size), keep_sell)

        if DIR_ON and self.dm.on and not self.iocs:
            now = snapshot.timestamp_us
            dev = inv - (self.target if LONG_K and LONG_UNWIND else 0)
            if dev and DIR_UNWIND_Z < 1e9 and now - self.unwind_us >= DIR_UNWIND_CD_US:
                sg = 1 if dev > 0 else -1
                if sg * self.dm.predict() < -DIR_UNWIND_Z:
                    side_r = Side.SELL if dev > 0 else Side.BUY
                    q = min(abs(dev), DIR_UNWIND_LOTS, MAX_POS + sg * inv - self._resting(side_r) - self._ioc_pending(-sg))
                    if q > 0:
                        self._ioc(-sg, self.dm.bid if dev > 0 else self.dm.ask, q)
                        self.unwind_us = now
            if take_sign and not self.iocs and now - self.take_us >= DIR_TAKE_CD_US:
                side_t = Side.BUY if take_sign > 0 else Side.SELL
                free = MAX_POS - take_sign * inv - self._resting(side_t) - self._ioc_pending(take_sign)
                if free >= DIR_TAKE_LOTS:
                    self._ioc(take_sign, self.dm.ask if take_sign > 0 else self.dm.bid, DIR_TAKE_LOTS)
                    self.take_us = now

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        st = event.status
        ioc_o = self.iocs.get(event.order_id)
        if ioc_o is not None:
            if st in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
                self.inventory += ioc_o[0] * event.filled_quantity
                ioc_o[1] = event.remaining_quantity
            if st in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
                self.iocs.pop(event.order_id, None)
            return
        o = self.orders.get(event.order_id)
        if o is None:
            return
        if st in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
            q = event.filled_quantity
            self.inventory += q if o[0] == Side.BUY else -q
            o[2] = event.remaining_quantity
        elif st == OrderStatus.MODIFIED:
            o[2] = event.remaining_quantity
            self.modifying.pop(event.order_id, None)
        elif st == OrderStatus.REJECTED and event.remaining_quantity > 0:  # modify rejected, order still resting
            o[2] = event.remaining_quantity
            self.modifying.pop(event.order_id, None)
            return
        if st in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            self.orders.pop(event.order_id, None)
            self.cancelling.discard(event.order_id)
            self.modifying.pop(event.order_id, None)
