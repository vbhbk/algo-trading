"""Market maker with large quotes and a hard exposure cap."""

from __future__ import annotations

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

QUOTE_SIZE = 100
MAX_POS = 70
SKEW_STEP = 45
FLOW_DECAY = 0.97
FLOW_BLOCK = 1e9
TAKE_FLOW = 1e9
TAKE_SIZE = 10
REQUOTE_TOL = 0
L2_OFFSET = 0
L2_SIZE = 20
IMB_TICKS = 0.0
ASYM = 0.0
ASYM_MIN = 1
REDUCE_BOOST = 0.0
SKEW_QUAD = 0.0
SAFE = 0
SAFE_CAP = 100
RESIZE = 0
SKEW_REDUCE = 0.0


class MaxSizeBest(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.inventory = 0
        self.orders: dict[str, list] = {}
        self.cancelling: set[str] = set()
        self.modifying: set[str] = set()
        self.flow = 0.0
        self.ioc_in_flight = False
        self.ioc: dict[str, Side] = {}

    def _resting(self, side: Side) -> list[str]:
        return [oid for oid, o in self.orders.items() if o[0] == side and oid not in self.cancelling]

    def _cancel(self, oid: str) -> None:
        if oid not in self.cancelling:
            self.cancelling.add(oid)
            self.cancel_order(oid)

    def _quote(self, side: Side, price: int | None, size: int, level: int = 0, room: int | None = None) -> None:
        live = [oid for oid in self._resting(side) if self.orders[oid][3] == level]
        if price is None or size <= 0:
            for oid in live:
                self._cancel(oid)
            return
        keep = None
        for oid in live:
            behind = (price - self.orders[oid][1]) if side == Side.BUY else (self.orders[oid][1] - price)
            if keep is None and 0 <= behind <= REQUOTE_TOL:
                keep = oid
            else:
                self._cancel(oid)
        if keep is not None:
            if RESIZE and self.orders[keep][2] > size and keep not in self.modifying:
                self.modifying.add(keep)
                self.modify_order(keep, size)
        else:
            if SAFE and room is not None:
                pend = sum(o[2] for i, o in self.orders.items() if o[0] == side and (i in self.cancelling or o[3] != level))
                size = min(size, room - pend)
                if size <= 0:
                    return
            if SAFE_CAP:
                held = sum(o[2] for o in self.orders.values() if o[0] == side)
                pos = self.inventory if side == Side.BUY else -self.inventory
                size = min(size, SAFE_CAP - pos - held)
                if size <= 0:
                    return
            oid = self.send_order(OrderRequest(side=side, price=price, quantity=size, order_type=OrderType.LIMIT))
            self.orders[oid] = [side, price, size, level]

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
        x = abs(inv) / SKEW_STEP
        mag = int(x + SKEW_QUAD * x * x)
        skew = mag if inv > 0 else -mag
        red = int(SKEW_REDUCE * mag)

        shift = 0
        if IMB_TICKS:
            bv = sum(o.quantity for o in snapshot.bids if o.owner == 'market' and o.price == bid)
            av = sum(o.quantity for o in snapshot.asks if o.owner == 'market' and o.price == ask)
            shift = round(IMB_TICKS * (bv - av) / (bv + av))
        bid_px = bid + (1 if spread >= 3 else 0) - max(skew, 0) + min(shift, 0)
        ask_px = ask - (1 if spread >= 3 else 0) - min(skew, 0) + max(shift, 0)
        if red:
            if inv > 0:
                ask_px -= red
            else:
                bid_px += red
        bid_px = min(bid_px, ask - 1)
        ask_px = max(ask_px, bid + 1)

        buy_q = sell_q = QUOTE_SIZE
        if inv and (ASYM or REDUCE_BOOST):
            frac = abs(inv) / MAX_POS
            grow = max(ASYM_MIN, int(QUOTE_SIZE * max(0.0, 1.0 - ASYM * frac)))
            shrink = int(QUOTE_SIZE * (1.0 + REDUCE_BOOST * frac))
            if inv > 0:
                buy_q, sell_q = grow, shrink
            else:
                buy_q, sell_q = shrink, grow
        buy_size = min(buy_q, MAX_POS - inv)
        sell_size = min(sell_q, MAX_POS + inv)
        if self.flow > FLOW_BLOCK:
            sell_size = 0 if inv <= 0 else sell_size
        if self.flow < -FLOW_BLOCK:
            buy_size = 0 if inv >= 0 else buy_size

        self._quote(Side.BUY, bid_px, buy_size, 0, MAX_POS - inv)
        self._quote(Side.SELL, ask_px, sell_size, 0, MAX_POS + inv)
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
        if event.order_id in self.modifying and st in (OrderStatus.MODIFIED, OrderStatus.REJECTED):
            self.modifying.discard(event.order_id)
            if o is not None:
                o[2] = event.remaining_quantity
            return
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
            self.modifying.discard(event.order_id)
