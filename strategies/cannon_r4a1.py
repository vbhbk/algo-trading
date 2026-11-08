# cannon_ref with both ridge models refit on round4_a1 (thresholds tuned on round4_a2)
# v8 with both ridge models refit on round3_a (logs/max/r3/fit.py)
"""Round 2 v8_all = v5 + target (session 3) + taking 0.6 / 45 / 100 ms + guard DEPTH 1100, QUIET 0.7 s; was: v5 + target (session 3): inventory steered to clip(200 x ridge 2 s prediction, +-95) via quote skew and
the unwind IOC (target - inventory); cap 95, all v5 guards kept.

Based on v5: v4 (cap fix + direction + unwind + taking Z 0.5 / 30) + guard recovery (S2) + directional pull (S3).

- Guard recovery: GUARD_MODE 1, HL 0.5 s, QUIET 1 s, one-side pull while only the flow keeps the guard on.
- Directional pull: a resting quote is cancelled when the model edge drops below PULL_EDGE 0.1 (v4: 0.0).

- Base: unified queue maker quoting 200 lots at the touch, position cap 95 (IOC trim above it).
- Guard: while spread >= 3, market size within 5 ticks of the touch < 1700, or market aggressive volume in the last
  1 s > 2000, all our orders are cancelled and nothing new is placed; stays on 2 s after the last trigger.
- Direction model (session 3, fit on the first 60% of round 2 train): place a quote only if the predicted fair price
  beats it by > 0.25 tick, cancel a resting quote when the edge drops below 0.0. Updated on every event.
- Unwind (session 3): IOC on the reducing side when sign(inventory) * prediction < -0.8, up to 95 lots, at most
  every 100 ms, also while the guard is on.
- Taking (session 3): IOC of 20 lots at the touch in the predicted direction when |prediction| >= 0.5 tick, at most
  every 250 ms, also while the guard is on; room is reserved first so resting orders + IOC never pass the cap.
- Cap: sizes are limited by inventory + every order of ours that could still fill on that side (free_room).
"""
from bisect import bisect_left, bisect_right
from collections import deque
from math import exp, log1p

from simulator.models import OrderBookSnapshot, OrderRequest, OrderStatus, OrderType, PrivateOrderEvent, Side, Trade
from simulator.strategy import BaseStrategy

POSITION_CAP = 95
QUOTE_LOTS = 200
SKEW_AT_CAP = 1.36
FLOW_LEAN = 0.25
FLOW_TAU_US = 80000
FLOW_NORM = 100.0
MIN_EDGE = 0.15
QUEUE_W = 500
QUEUE_CAP = 1000000000
QUOTE_TTL_US = 10000000000000
ADVERSE_FLOW = 9.0
QPOS_KEEP = -1
REQUOTE_US = 0
TWO_SIDED = True
FAST_REPLACE = True
TOPUP = True
TOPUP_MIN = 1
TOPUP_EDGE = False
SHRINK_EXCESS = True


COLLECT = False
FADE_WEIGHT = 0.25
REGIME_THRESHOLD = 1e+30
REGIME_SCALE = 1.0
GATE = False
GATE_REDUCING_BYPASS = True
GATE_TABLE = {}
GATE_DEFAULT = 0.0
GATE_THRESHOLD = 0.0

GUARD_SPREAD = 3
GUARD_DEPTH = 1100
GUARD_FLOW = 2000
GUARD_PAUSE_US = 2_000_000
GUARD_ON = True
# guard recovery (session 2, guard2): GUARD_MODE 0 = blind 2 s pause; 1 = keep paused while a trigger or the decayed
# market flow (half-life GUARD_HL_US, rate > GUARD_RESUME_FRAC x GUARD_FLOW) stays elevated, resume after GUARD_QUIET_US
# of quiet; GUARD_ONE_SIDE: when only the flow keeps it on, pull just the side the flow hits.
GUARD_MODE = 1
GUARD_HL_US = 500_000
GUARD_QUIET_US = 700_000
GUARD_RESUME_FRAC = 0.5
GUARD_ONE_SIDE = True
UNWIND_Z = 0.8
UNWIND_LOTS = 95
UNWIND_COOLDOWN_US = 100_000
TAKE_Z = 0.6
TAKE_LOTS = 95
TAKE_COOLDOWN_US = 100_000
TARGET_K = 200        # lots per tick of the long (2 s) prediction; 0 = v5 (target 0)
TARGET_MAX = 95
LONG_H = 2            # horizon of the long ridge (s), fit on round2 train first 60% (analysis/target_v5/fit_long.py)
LONG_MODELS = {
    2: (-0.6122049432703986, (-0.08355683652858811, 0.024484469569039456, 0.1125840188541784, 0.03009468883211256, -0.02819392304052017, 1.2410308595714754, 0.20572058457399045, 0.16175112867504876, -0.09104458691652506, -0.16051190625768005, 0.14990036167621515, 0.08049356926934371, -0.006331714970350411, -0.009578049481344615, 0.14836861319495187, -0.034814065966420366, 0.012131313332244193, -0.07458027744958194, -0.2966872814449031, 0.011852869370227687, 0.0, -0.1491605549007585, 0.21772405874840267, 1.297536513823359, -0.317154962883196, 0.7199725026103854, 0.03918033405170079, -0.055074750131731874, -0.08886372788274115, 0.07340708833513696, 0.05415626162114357, -0.13122506345193208, -0.07458027744958197)),
}
PULL_EDGE = 0.1      # session 3: cancel a resting quote when model edge < PULL_EDGE (v4: 0.0)
DONE = (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED)


class DirectionModel:
    # ridge H=0.25s, spread 1, alpha 10, fit on the whole round2_public.csv (analysis/direction/export_model.py, fit_frac 1.0)
    HORIZON_S = 0.25
    INTERCEPT = -0.14095877885036462
    W = (-0.01132818888767065, -0.0442755440761233, 0.04904918735805192, 0.017018987282288674, -0.013164371357034087, 0.6473101476059241, 0.4043733250152936, -0.010628371072301274, -0.10782581049975291, -0.03670065567773626, 0.06157151615405971, 0.018644980926032114, -0.006298978594489987, -0.0011355192610591602, 0.09894025811186175, 0.06812695221164132, -0.005310619201246497, -0.0580381005242613, 0.03655668439777515, 0.004995423868248979, 0.0, -0.11607620105066507, 0.12097196029906576, 0.7121586207047063, -0.1576970999946875, 0.17222820240653167, 0.019182982815608162, -0.022527773518312342, -0.058038860290400125, 0.06052389892859796, 0.023587371612278762, 0.04439795226568309, -0.05803810052424453)
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


class _Ord:
    def __init__(self, side: Side, price: int, qty: int) -> None:
        self.side, self.price, self.rem = side, price, qty
        self.accepted_us: int | None = None
        self.cancel_pending = False
        self.modify_pending = False


class Round2MLMaker(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.research_init()
        self.dm = DirectionModel()
        self.model_on = False
        self.fv = 0.0
        self.n_model_cancel = 0
        self.position = 0
        self.orders: dict[str, _Ord] = {}
        self.next_quote_us = 0
        self.flow_us = 0
        self.signed_flow = 0.0
        self.total_flow = 0.0
        self.bid = self.ask = None
        self.trim_id: str | None = None
        self.trim_side: Side | None = None
        self.trim_qty = 0
        self.unwind_id = None
        self.unwind_side = None
        self.target = 0
        self.n_tgt_unwind = 0
        self.unwind_us = -10 ** 15
        self.unwind_q = 0
        self.take_id = None
        self.take_sign = 0
        self.take_q = 0
        self.take_us = -10 ** 15
        self.reserve_sign = 0
        self.guard_live = set()
        self.guard_flow = deque()
        self.guard_flow_sum = 0
        self.guard_until = -1
        self.g2_abs = 0.0
        self.g2_sgn = 0.0
        self.g2_us = None
        self.g2_hard_until = -1
        self.g2_flow_until = -1
        self.g2_block = None  # Side whose LIMIT quotes are pulled (one-side mode)

    def research_init(self) -> None:
        self.research_now = 0
        self.research_last = 0
        self.research_rate = 0.0
        self.research_fast = None
        self.research_slow = None
        self.research_mid = None
        self.research_spread = 1
        self.research_queues = {Side.BUY: 0, Side.SELL: 0}
        self.research_records: dict[str, dict] = {}
        self.research_samples: list[list[float]] = []
        self.research_sample_us = -100_000
        self.research_blocked = 0

    def research_update(self, snapshot: OrderBookSnapshot) -> None:
        now = snapshot.timestamp_us
        dt = max(0, now - self.research_last)
        self.research_rate = self.research_rate * exp(-dt / 1_000_000) + 1
        self.research_last = now
        self.research_now = now
        bid = next((o.price for o in snapshot.bids if o.owner == 'market'), None)
        ask = next((o.price for o in snapshot.asks if o.owner == 'market'), None)
        if bid is None or ask is None or ask <= bid:
            return
        mid = (bid + ask) / 2
        self.research_mid = mid
        self.research_spread = ask - bid
        if self.research_fast is None:
            self.research_fast = self.research_slow = mid
        else:
            self.research_fast += (mid - self.research_fast) * (1 - exp(-dt / 2_000_000))
            self.research_slow += (mid - self.research_slow) * (1 - exp(-dt / 10_000_000))
        if COLLECT or GATE:
            self.research_queues = {
                Side.BUY: sum(o.quantity for o in snapshot.bids if o.owner == 'market' and o.price == bid),
                Side.SELL: sum(o.quantity for o in snapshot.asks if o.owner == 'market' and o.price == ask),
            }
        if COLLECT and now - self.research_sample_us >= 100_000:
            self.research_samples.append([now, mid, self.research_rate])
            self.research_sample_us = now

    def research_tilt(self) -> float:
        if self.research_fast is None:
            return 0.0
        return FADE_WEIGHT * max(-1.0, min(1.0, self.research_slow - self.research_fast))

    def research_lots(self) -> int:
        if self.research_rate > REGIME_THRESHOLD:
            return max(1, int(QUOTE_LOTS * REGIME_SCALE))
        return QUOTE_LOTS

    def research_key(self, side: Side) -> str:
        sign = 1 if side == Side.BUY else -1
        queue = self.research_queues[side]
        flow = sign * self.signed_flow / (self.total_flow + FLOW_NORM)
        qb = 0 if queue < 250 else 1 if queue < 1000 else 2
        fb = 0 if flow < -0.2 else 1 if flow <= 0.2 else 2
        reducing = int(sign * self.position < 0)
        return f'{qb}:{fb}:{reducing}'

    def research_allow(self, side: Side) -> bool:
        if not GATE:
            return True
        sign = 1 if side == Side.BUY else -1
        if GATE_REDUCING_BYPASS and sign * self.position < 0:
            return True
        score = GATE_TABLE.get(self.research_key(side), GATE_DEFAULT)
        if score <= GATE_THRESHOLD:
            self.research_blocked += 1
            return False
        return True

    def research_quote(self, oid: str, side: Side, price: int, qty: int) -> None:
        if COLLECT:
            self.research_records[oid] = {
                'submitted_us': self.research_now, 'key': self.research_key(side),
                'sign': 1 if side == Side.BUY else -1, 'price': price, 'quantity': qty,
                'fill_us': None, 'fill_price': None,
            }

    def research_fill(self, event: PrivateOrderEvent) -> None:
        if not COLLECT:
            return
        record = self.research_records.get(event.order_id)
        if record is not None and record['fill_us'] is None and event.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            if event.timestamp_us <= record['submitted_us'] + 1_000_000:
                record['fill_us'] = event.timestamp_us
                record['fill_price'] = event.fill_price


    def update_flow(self, now: int, trades: list[Trade]) -> float:
        weight = exp(-(now - self.flow_us) / FLOW_TAU_US)
        self.signed_flow *= weight
        self.total_flow *= weight
        self.flow_us = now
        for t in trades:
            if t.maker_owner == 'market' and t.taker_owner == 'market':
                self.signed_flow += (1 if t.book_side == 'ask' else -1) * t.quantity
                self.total_flow += t.quantity
        return self.signed_flow / (self.total_flow + FLOW_NORM)

    def _send(self, side: Side, price: int, qty: int) -> None:
        if side == self.g2_block or not self.research_allow(side) or not self.dm.allow(1 if side == Side.BUY else -1, price):
            return
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=qty, order_type=OrderType.LIMIT))
        self.orders[oid] = _Ord(side, price, qty)
        self.research_quote(oid, side, price, qty)

    def _room(self, sign: int) -> int:
        """Position room on a side; an in-flight unwind IOC on the same side counts as already filled."""
        room = POSITION_CAP - sign * self.position
        if self.unwind_id is not None and (1 if self.unwind_side == Side.BUY else -1) == sign:
            room -= self.unwind_q
        if self.take_id is not None and self.take_sign == sign:
            room -= self.take_q
        elif self.reserve_sign == sign:
            room -= TAKE_LOTS
        return room

    def maybe_take(self, now: int) -> None:
        """IOC in the predicted direction; room is reserved first so resting orders + IOC never pass the cap."""
        dm = self.dm
        if not (dm.on and abs(dm.predict()) >= TAKE_Z):
            self.reserve_sign = 0
            return
        if (self.take_id is not None or self.unwind_id is not None or self.trim_id is not None
                or now - self.take_us < TAKE_COOLDOWN_US):
            return
        sign = 1 if dm.pred > 0 else -1
        side = Side.BUY if sign > 0 else Side.SELL
        free = POSITION_CAP - sign * self.position - sum(o.rem for o in self.orders.values() if o.side == side)
        if free < TAKE_LOTS:
            if POSITION_CAP - sign * self.position >= TAKE_LOTS:
                self.reserve_sign = sign     # shrink pass frees the room; IOC follows on a later event
            return
        self.reserve_sign = 0
        self.take_sign, self.take_q, self.take_us = sign, TAKE_LOTS, now
        self.take_id = self.send_order(OrderRequest(side=side, price=dm.ask if sign > 0 else dm.bid,
                                                    quantity=TAKE_LOTS, order_type=OrderType.IOC))

    def free_room(self, side: Side, sign: int) -> int:
        """Lots we may still add on `side`: cap minus inventory minus every order of ours that could still fill
        on that side (resting, in flight, cancel-pending, modify-pending at its old size, IOC trim)."""
        pending = 0
        for o in self.orders.values():
            if o.side == side:
                pending += o.rem
        if self.trim_id is not None and self.trim_side == side:
            pending += self.trim_qty
        if self.unwind_id is not None and self.unwind_side == side:
            pending += self.unwind_q
        if self.take_id is not None and (1 if side == Side.BUY else -1) == self.take_sign:
            pending += self.take_q
        return POSITION_CAP - sign * self.position - pending

    def _cancel(self, oid: str, o: _Ord) -> None:
        o.cancel_pending = True
        self.cancel_order(oid)

    def maybe_trim(self) -> None:
        excess = abs(self.position) - POSITION_CAP
        if excess <= 0 or self.trim_id is not None or self.bid is None or self.ask is None:
            return
        if self.position > 0:
            side, price = Side.SELL, self.bid
        else:
            side, price = Side.BUY, self.ask
        self.trim_side = side
        self.trim_qty = excess
        self.trim_id = self.send_order(OrderRequest(side=side, price=price, quantity=excess, order_type=OrderType.IOC))

    @staticmethod
    def queue_ahead(levels, oid: str, price: int) -> int:
        ahead = 0
        for o in levels:
            if o.price != price:
                if ahead:
                    break
                continue
            if o.order_id == oid:
                return ahead
            if o.owner == 'market':
                ahead += o.quantity
        return ahead

    def dm_observe(self, snapshot, trades) -> None:
        dm = self.dm
        dm.update(snapshot, trades)
        self.model_on = dm.on
        pos = self.position
        now = snapshot.timestamp_us
        if TARGET_K and dm.on:
            dm.predict()
            b, w = LONG_MODELS[LONG_H]
            pl = b
            for wi, v in zip(w, dm.x):
                pl += wi * v
            t = int(round(TARGET_K * pl))
            self.target = TARGET_MAX if t > TARGET_MAX else -TARGET_MAX if t < -TARGET_MAX else t
        dev = pos - self.target
        if dev and dm.on and self.unwind_id is None and self.trim_id is None and now - self.unwind_us >= UNWIND_COOLDOWN_US:
            z = (1 if dev > 0 else -1) * dm.predict()
            side = Side.SELL if dev > 0 else Side.BUY
            s = -1 if dev > 0 else 1
            if self.target:  # may add inventory: count every order that could still fill, no overlap with a take IOC
                q = 0 if self.take_id is not None else min(abs(dev), UNWIND_LOTS, self.free_room(side, s))
            else:
                q = min(abs(dev), UNWIND_LOTS, POSITION_CAP - s * pos - sum(o.rem for o in self.orders.values() if o.side == side)
                        - (self.take_q if self.take_id is not None and self.take_sign == s else 0))
            if z < -UNWIND_Z and q > 0:
                self.unwind_side = side
                self.unwind_us = now
                self.unwind_id = self.send_order(OrderRequest(side=side, price=dm.bid if dev > 0 else dm.ask,
                                                              quantity=q, order_type=OrderType.IOC))
                self.unwind_q = q
                if self.target:
                    self.n_tgt_unwind += 1
        self.maybe_take(now)
        self.fv = dm.mid + dm.pred

    def send_order(self, order: OrderRequest) -> str:
        oid = super().send_order(order)
        self.guard_live.add(oid)
        return oid

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        now = snapshot.timestamp_us
        for t in trades:
            if t.taker_owner == 'market':
                self.guard_flow.append((t.timestamp_us, t.quantity))
                self.guard_flow_sum += t.quantity
        while self.guard_flow and self.guard_flow[0][0] < now - 1_000_000:
            self.guard_flow_sum -= self.guard_flow.popleft()[1]
        bid = next((o.price for o in snapshot.bids if o.owner == 'market'), None)
        ask = next((o.price for o in snapshot.asks if o.owner == 'market'), None)
        danger = bid is None or ask is None or ask - bid >= GUARD_SPREAD or self.guard_flow_sum > GUARD_FLOW
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
            danger = depth < GUARD_DEPTH
        if GUARD_MODE == 1 and GUARD_ON:
            self.g2_step(now, trades, danger, snapshot)
            return
        if danger and GUARD_ON:
            self.guard_until = now + GUARD_PAUSE_US
        if now < self.guard_until:
            for oid in list(self.guard_live):
                self.cancel_order(oid)
                self.guard_live.discard(oid)  # the CANCELED event still reaches on_private_order_update
            self.dm_observe(snapshot, trades)
            return
        self.maker_step(snapshot, trades)

    def g2_step(self, now, trades, danger, snapshot) -> None:
        if self.g2_us is not None:
            k = 0.5 ** ((now - self.g2_us) / GUARD_HL_US)
            self.g2_abs *= k
            self.g2_sgn *= k
        self.g2_us = now
        for t in trades:
            if t.taker_owner == 'market':
                self.g2_abs += t.quantity
                self.g2_sgn += t.quantity if t.book_side == 'ask' else -t.quantity
        rate = self.g2_abs * 0.6931471805599453 / (GUARD_HL_US / 1e6)  # lots per second
        flow_trig = self.guard_flow_sum > GUARD_FLOW
        hard = danger and not flow_trig  # spread / depth / empty side
        paused = now < self.g2_hard_until or now < self.g2_flow_until
        if hard or (paused and now < self.g2_hard_until and rate > GUARD_RESUME_FRAC * GUARD_FLOW):
            self.g2_hard_until = now + GUARD_QUIET_US
        if flow_trig or (paused and rate > GUARD_RESUME_FRAC * GUARD_FLOW):
            self.g2_flow_until = now + GUARD_QUIET_US
        both = now < self.g2_hard_until or (not GUARD_ONE_SIDE and now < self.g2_flow_until)
        self.guard_until = max(self.g2_hard_until, self.g2_flow_until)
        if both:
            self.g2_block = None
            for oid in list(self.guard_live):
                self.cancel_order(oid)
                self.guard_live.discard(oid)  # the CANCELED event still reaches on_private_order_update
            self.dm_observe(snapshot, trades)
            return
        self.g2_block = None
        if now < self.g2_flow_until:  # flow run only: pull the side the aggressors hit
            side = Side.SELL if self.g2_sgn > 0 else Side.BUY
            self.g2_block = side
            for oid, o in self.orders.items():
                if o.side == side and not o.cancel_pending:
                    self._cancel(oid, o)
        self.maker_step(snapshot, trades)

    def maker_step(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        self.dm_observe(snapshot, trades)
        self.research_update(snapshot)
        now = snapshot.timestamp_us
        if trades or now - self.flow_us >= 10_000:
            flow = self.update_flow(now, trades)
        else:
            flow = self.signed_flow / (self.total_flow + FLOW_NORM)
        bid = next((o.price for o in snapshot.bids if o.owner == 'market'), None)
        ask = next((o.price for o in snapshot.asks if o.owner == 'market'), None)
        self.bid, self.ask = bid, ask
        self.maybe_trim()

        # cancel pass: stale price, adverse flow, lifetime
        for oid, o in self.orders.items():
            if o.accepted_us is None or o.cancel_pending:
                continue
            if bid is None or ask is None:
                self._cancel(oid, o)
                continue
            buy = o.side == Side.BUY
            if buy:
                stale = o.price < bid or o.price >= ask
                adverse = flow < -ADVERSE_FLOW
            else:
                stale = o.price > ask or o.price <= bid
                adverse = flow > ADVERSE_FLOW
            if adverse and QPOS_KEEP >= 0 and self.queue_ahead(snapshot.bids if buy else snapshot.asks, oid, o.price) <= QPOS_KEEP:
                adverse = False
            if not stale and not self.dm.keep(1 if buy else -1, o.price):
                stale = True
                self.n_model_cancel += 1
            if stale or adverse or now - o.accepted_us >= QUOTE_TTL_US:
                self._cancel(oid, o)

        if SHRINK_EXCESS and bid is not None and ask is not None:
            for side, sign in ((Side.BUY, 1), (Side.SELL, -1)):
                room = self._room(sign)
                live = [(oid, o) for oid, o in self.orders.items()
                        if o.side == side and o.accepted_us is not None and not o.cancel_pending and not o.modify_pending]
                excess = sum(o.rem for _, o in live) - max(room, 0)
                for oid, o in reversed(live):
                    if excess <= 0:
                        break
                    cut = min(excess, o.rem)
                    if cut >= o.rem:
                        self._cancel(oid, o)
                    else:
                        o.modify_pending = True
                        self.modify_order(oid, o.rem - cut)
                    excess -= cut

        if now < self.next_quote_us or bid is None or ask is None or ask <= bid:
            return
        active = [o for o in self.orders.values() if not (FAST_REPLACE and o.cancel_pending)]
        if active and not TWO_SIDED and not TOPUP:
            return
        if active and not TWO_SIDED and TOPUP:
            pass  # only the side already quoted may top up (handled below)
        cancels_in_flight = FAST_REPLACE and any(o.cancel_pending for o in self.orders.values())
        if not cancels_in_flight:
            self.next_quote_us = now + REQUOTE_US
        skew = SKEW_AT_CAP / POSITION_CAP
        fair = (bid + ask) / 2 + FLOW_LEAN * flow - skew * (self.position - self.target) + self.research_tilt()
        active_sides = {o.side for o in active}
        cands = []
        for side, sign, price, levels in ((Side.BUY, 1, bid, snapshot.bids), (Side.SELL, -1, ask, snapshot.asks)):
            mine = [o for o in active if o.side == side]
            room = self._room(sign)
            edge = sign * (fair - price)
            if mine:
                if not TOPUP or any(o.cancel_pending or o.accepted_us is None or o.price != price for o in mine):
                    continue
                add = min(min(self.research_lots(), room) - sum(o.rem for o in mine), self.free_room(side, sign))
                if add >= TOPUP_MIN and (not TOPUP_EDGE or edge > MIN_EDGE):
                    self._send(side, price, add)
                    if cancels_in_flight:
                        self.next_quote_us = now + REQUOTE_US
                continue
            if not TWO_SIDED and active_sides:
                continue
            queue = sum(o.quantity for o in levels if o.owner == 'market' and o.price == price)
            qty = min(self.research_lots(), room, self.free_room(side, sign))
            if qty <= 0 or edge <= MIN_EDGE or queue > QUEUE_CAP:
                continue
            cands.append((edge / (1 + queue / QUEUE_W), side, price, qty))
        if not cands:
            return
        cands.sort(reverse=True)
        if not TWO_SIDED:
            cands = cands[:1]
        if cancels_in_flight:
            self.next_quote_us = now + REQUOTE_US
        for _, side, price, qty in cands:
            self._send(side, price, qty)

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        if event.status in DONE:
            self.guard_live.discard(event.order_id)
        self.research_fill(event)
        if event.order_id == self.trim_id:
            if event.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
                self.position += (1 if self.trim_side == Side.BUY else -1) * event.filled_quantity
            if event.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
                self.trim_id = None
                self.maybe_trim()
            return
        if event.order_id == self.take_id:
            if event.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
                self.position += self.take_sign * event.filled_quantity
                self.take_q -= event.filled_quantity
            if event.status in DONE:
                self.take_id = None
                self.take_q = 0
            return
        if event.order_id == self.unwind_id:
            if event.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
                self.position += (1 if self.unwind_side == Side.BUY else -1) * event.filled_quantity
                self.unwind_q -= event.filled_quantity
            if event.status in DONE:
                self.unwind_id = None
            return
        o = self.orders.get(event.order_id)
        if o is None:
            return
        st = event.status
        if st == OrderStatus.ACCEPTED:
            o.accepted_us = event.timestamp_us
        elif st in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
            self.position += (1 if o.side == Side.BUY else -1) * event.filled_quantity
            o.rem = event.remaining_quantity
            self.maybe_trim()
        elif st == OrderStatus.MODIFIED:
            o.modify_pending = False
            o.rem = event.remaining_quantity
        elif st == OrderStatus.REJECTED and o.accepted_us is not None:
            if o.modify_pending:
                o.modify_pending = False
            else:
                o.cancel_pending = False
            return
        if st in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            del self.orders[event.order_id]
