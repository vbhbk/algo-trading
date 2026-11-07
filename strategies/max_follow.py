"""Follow informed aggressive flow: take the side of heavy market aggression with IOC orders."""

from __future__ import annotations

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

MAX_POS = 95
FLOW_WIN_US = 200_000  # window for signed aggressive market volume
FLOW_TH = 200  # |signed volume| in the window that sets the target to +-MAX_POS
HOLD_US = 30_000_000  # no new signal for this long -> target 0
SLIP = 1  # IOC price this many ticks through the touch
CHUNK = 95  # max lots per IOC


class MaxFollow(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.inventory = 0
        self.flow: deque = deque()  # (t, signed qty)
        self.flow_sum = 0
        self.target = 0
        self.last_signal = -(10**18)
        self.live: dict[str, list] = {}  # id -> [sign, remaining]

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        now = snapshot.timestamp_us
        for t in trades:
            if t.taker_owner == 'market':
                v = t.quantity if t.book_side == 'ask' else -t.quantity
                self.flow.append((t.timestamp_us, v))
                self.flow_sum += v
        while self.flow and now - self.flow[0][0] > FLOW_WIN_US:
            self.flow_sum -= self.flow.popleft()[1]
        if self.flow_sum >= FLOW_TH:
            self.target, self.last_signal = MAX_POS, now
        elif self.flow_sum <= -FLOW_TH:
            self.target, self.last_signal = -MAX_POS, now
        elif now - self.last_signal > HOLD_US:
            self.target = 0

        if self.live:
            return
        need = self.target - self.inventory
        if need == 0:
            return
        bid = next((o.price for o in snapshot.bids if o.owner == 'market'), None)
        ask = next((o.price for o in snapshot.asks if o.owner == 'market'), None)
        q = min(abs(need), CHUNK)
        if need > 0 and ask is not None:
            oid = self.send_order(OrderRequest(side=Side.BUY, price=ask + SLIP, quantity=q, order_type=OrderType.IOC))
            self.live[oid] = [1, q]
        elif need < 0 and bid is not None:
            oid = self.send_order(OrderRequest(side=Side.SELL, price=bid - SLIP, quantity=q, order_type=OrderType.IOC))
            self.live[oid] = [-1, q]

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        o = self.live.get(event.order_id)
        if o is None:
            return
        if event.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
            self.inventory += o[0] * event.filled_quantity
            o[1] = event.remaining_quantity
        if event.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            self.live.pop(event.order_id, None)
