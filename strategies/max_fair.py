"""Market maker with a microprice inventory target and size tilt."""

from __future__ import annotations

import math

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

QUOTE_SIZE = 20
MAX_POS = 50
SKEW_STEP = 20
FLOW_DECAY = 0.97
FLOW_BLOCK = 1e9
TAKE_FLOW = 1e9
TAKE_SIZE = 10
REQUOTE_TOL = 0
L2_OFFSET = 0
L2_SIZE = 20
IMB_TICKS = 0.0

FAIR_SRC = 0
FAIR_DEPTH = 3
FAIR_W = 1.0
EMA_W = 0.0
EMA_HL = 60.0
FAIR_EDGE = -9.0
KEEP_EDGE = -9.0
IMP_EDGE = 9.0
INV_FAIR = 0.0
IMPROVE_SPREAD = 3
TGT_K = 0.0
TGT_MK = 30.0
TGT_MAX = 30
TGT_DK = 0.0
TGT_HL = 0.0
SZ_K = -0.7
L1_MODE = 0
TGT_NORM = 1


class MaxFair(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.inventory = 0
        self.orders: dict[str, list] = {}
        self.cancelling: set[str] = set()
        self.flow = 0.0
        self.ioc_in_flight = False
        self.ioc: dict[str, Side] = {}
        self.ema = None
        self.last_t = 0
        self.tgt_s = None
        self.tgt_t = 0

    def _resting(self, side: Side) -> list[str]:
        return [oid for oid, o in self.orders.items() if o[0] == side and oid not in self.cancelling]

    def _cancel(self, oid: str) -> None:
        if oid not in self.cancelling:
            self.cancelling.add(oid)
            self.cancel_order(oid)

    def _quote(self, side: Side, price: int | None, size: int, level: int = 0, keep_px: int | None = None) -> None:
        live = [oid for oid in self._resting(side) if self.orders[oid][3] == level]
        if price is None or size <= 0:
            for oid in live:
                self._cancel(oid)
            return
        keep = None
        for oid in live:
            p = self.orders[oid][1]
            behind = (price - p) if side == Side.BUY else (p - price)
            ok = 0 <= behind <= REQUOTE_TOL
            if not ok and keep_px is not None:
                ok = (price <= p <= keep_px) if side == Side.BUY else (keep_px <= p <= price)
            if keep is None and ok:
                keep = oid
            else:
                self._cancel(oid)
        if keep is None:
            oid = self.send_order(OrderRequest(side=side, price=price, quantity=size, order_type=OrderType.LIMIT))
            self.orders[oid] = [side, price, size, level]

    def _fair(self, snapshot: OrderBookSnapshot, bid: int, ask: int) -> float:
        mid = 0.5 * (bid + ask)
        f = mid
        if FAIR_SRC == 1:
            bv = 0
            for o in snapshot.bids:
                if o.price != bid:
                    break
                if o.owner == 'market':
                    bv += o.quantity
            av = 0
            for o in snapshot.asks:
                if o.price != ask:
                    break
                if o.owner == 'market':
                    av += o.quantity
            if bv + av > 0:
                f = mid + FAIR_W * ((bid * av + ask * bv) / (bv + av) - mid)
        elif FAIR_SRC == 2:
            lo = bid - FAIR_DEPTH
            bq = bpq = 0
            for o in snapshot.bids:
                if o.price <= lo:
                    break
                if o.owner == 'market':
                    bq += o.quantity
                    bpq += o.price * o.quantity
            hi = ask + FAIR_DEPTH
            aq = apq = 0
            for o in snapshot.asks:
                if o.price >= hi:
                    break
                if o.owner == 'market':
                    aq += o.quantity
                    apq += o.price * o.quantity
            if bq > 0 and aq > 0:
                f = mid + FAIR_W * (0.5 * (bpq / bq + apq / aq) - mid)
        if EMA_W:
            if not TGT_K:
                self._ema_update(snapshot.timestamp_us, mid)
            f += EMA_W * (self.ema - mid)
        return f

    def _ema_update(self, t: int, mid: float) -> None:
        if self.ema is None:
            self.ema = mid
        elif t != self.last_t:
            a = 1.0 - math.exp(-(t - self.last_t) * 1e-6 * 0.6931471805599453 / EMA_HL)
            self.ema += a * (mid - self.ema)
        self.last_t = t

    @staticmethod
    def _l1(snapshot: OrderBookSnapshot, bid: int, ask: int) -> tuple[int, int]:
        bv = 0
        for o in snapshot.bids:
            p = o.price
            if p < bid or (p > bid and not L1_MODE):
                break
            if p == bid and o.owner == 'market':
                bv += o.quantity
        av = 0
        for o in snapshot.asks:
            p = o.price
            if p > ask or (p < ask and not L1_MODE):
                break
            if p == ask and o.owner == 'market':
                av += o.quantity
        return bv, av

    def _target(self, snapshot: OrderBookSnapshot, bid: int, ask: int, bv: int, av: int) -> float:
        mid = 0.5 * (bid + ask)
        tgt = 0.0
        if TGT_K:
            self._ema_update(snapshot.timestamp_us, mid)
            tgt += TGT_K * (self.ema - mid)
        if TGT_MK and bv + av > 0:
            d = (bid * av + ask * bv) / (bv + av) - mid
            tgt += TGT_MK * (d / (ask - bid) if TGT_NORM else d)
        if TGT_DK:
            lo = bid - FAIR_DEPTH
            bq = bpq = 0
            for o in snapshot.bids:
                if o.price <= lo:
                    break
                if o.owner == 'market':
                    bq += o.quantity
                    bpq += o.price * o.quantity
            hi = ask + FAIR_DEPTH
            aq = apq = 0
            for o in snapshot.asks:
                if o.price >= hi:
                    break
                if o.owner == 'market':
                    aq += o.quantity
                    apq += o.price * o.quantity
            if bq > 0 and aq > 0:
                d = 0.5 * (bpq / bq + apq / aq) - mid
                tgt += TGT_DK * (d / (ask - bid) if TGT_NORM else d)
        if TGT_HL:
            t = snapshot.timestamp_us
            if self.tgt_s is None:
                self.tgt_s = tgt
            elif t != self.tgt_t:
                a = 1.0 - math.exp(-(t - self.tgt_t) * 1e-6 * 0.6931471805599453 / TGT_HL)
                self.tgt_s += a * (tgt - self.tgt_s)
            self.tgt_t = t
            tgt = self.tgt_s
        return max(-TGT_MAX, min(TGT_MAX, tgt))

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        self.flow *= FLOW_DECAY
        for t in trades:
            if t.taker_owner == 'market':
                self.flow += t.quantity if t.book_side == 'ask' else -t.quantity

        bid = next((o.price for o in snapshot.bids if o.owner == 'market'), None)
        ask = next((o.price for o in snapshot.asks if o.owner == 'market'), None)
        if bid is None or ask is None:
            self._quote(Side.BUY, None, 0)
            self._quote(Side.SELL, None, 0)
            return

        spread = ask - bid
        inv = self.inventory
        bv = av = 0
        if TGT_MK or SZ_K:
            bv, av = self._l1(snapshot, bid, ask)
        tgt = 0.0
        if TGT_K or TGT_MK or TGT_DK:
            tgt = self._target(snapshot, bid, ask, bv, av)
        skew = int((inv - tgt) / SKEW_STEP)

        shift = 0
        if IMB_TICKS:
            bv = sum(o.quantity for o in snapshot.bids if o.owner == 'market' and o.price == bid)
            av = sum(o.quantity for o in snapshot.asks if o.owner == 'market' and o.price == ask)
            shift = round(IMB_TICKS * (bv - av) / (bv + av))
        imp = 1 if spread >= IMPROVE_SPREAD else 0
        bid_px = bid + imp - max(skew, 0) + min(shift, 0)
        ask_px = ask - imp - min(skew, 0) + max(shift, 0)

        bid_keep = ask_keep = None
        use_fair = FAIR_SRC or EMA_W or INV_FAIR
        if use_fair:
            f = self._fair(snapshot, bid, ask) - INV_FAIR * inv / SKEW_STEP
            if spread == 2:
                if bid + 1 <= f - IMP_EDGE and skew <= 0:
                    bid_px = max(bid_px, bid + 1)
                if ask - 1 >= f + IMP_EDGE and skew >= 0:
                    ask_px = min(ask_px, ask - 1)
            base_bid, base_ask = bid_px, ask_px
            bid_px = min(bid_px, math.floor(f - FAIR_EDGE))
            ask_px = max(ask_px, math.ceil(f + FAIR_EDGE))
            if bid_px < base_bid:
                bid_keep = min(base_bid, math.floor(f - KEEP_EDGE), ask - 1)
            if ask_px > base_ask:
                ask_keep = max(base_ask, math.ceil(f + KEEP_EDGE), bid + 1)
        bid_px = min(bid_px, ask - 1)
        ask_px = max(ask_px, bid + 1)

        bq_size = aq_size = QUOTE_SIZE
        if SZ_K and bv + av > 0:
            x = (bv - av) / (bv + av)
            bq_size = max(0, round(QUOTE_SIZE * (1 + SZ_K * x)))
            aq_size = max(0, round(QUOTE_SIZE * (1 - SZ_K * x)))
        buy_size = min(bq_size, MAX_POS - inv)
        sell_size = min(aq_size, MAX_POS + inv)
        if self.flow > FLOW_BLOCK:
            sell_size = 0 if inv <= 0 else sell_size
        if self.flow < -FLOW_BLOCK:
            buy_size = 0 if inv >= 0 else buy_size

        self._quote(Side.BUY, bid_px, buy_size, 0, bid_keep)
        self._quote(Side.SELL, ask_px, sell_size, 0, ask_keep)
        if L2_OFFSET:
            self._quote(Side.BUY, bid_px - L2_OFFSET, min(L2_SIZE, MAX_POS - inv - buy_size), 1)
            self._quote(Side.SELL, ask_px + L2_OFFSET, min(L2_SIZE, MAX_POS + inv - sell_size), 1)

        if not self.ioc_in_flight and abs(self.flow) > TAKE_FLOW:
            if self.flow > 0 and inv + TAKE_SIZE <= MAX_POS:
                self.ioc[self.send_order(OrderRequest(side=Side.BUY, price=ask, quantity=TAKE_SIZE, order_type=OrderType.IOC))] = Side.BUY
                self.ioc_in_flight = True
            elif self.flow < 0 and inv - TAKE_SIZE >= -MAX_POS:
                self.ioc[self.send_order(OrderRequest(side=Side.SELL, price=bid, quantity=TAKE_SIZE, order_type=OrderType.IOC))] = Side.SELL
                self.ioc_in_flight = True

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        st = event.status
        o = self.orders.get(event.order_id)
        if st in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
            side = o[0] if o is not None else self.ioc.get(event.order_id)
            if side is None:
                return
            self.inventory += event.filled_quantity if side == Side.BUY else -event.filled_quantity
            if o is not None:
                o[2] = event.remaining_quantity
        if st in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            if self.ioc.pop(event.order_id, None) is not None:
                self.ioc_in_flight = False
            self.orders.pop(event.order_id, None)
            self.cancelling.discard(event.order_id)
