"""Impulse follower: decayed score of market aggressive prints (size-weighted, boosted when the spread was wide,
damped when the print just cleared a level of 1-2x its size); hold +-CAP in the score direction with hysteresis,
trade by IOC at the touch. One IOC in flight at a time, sized to the gap to target, so |position| <= CAP always."""
from math import exp

from simulator.models import OrderBookSnapshot, OrderRequest, OrderStatus, OrderType, PrivateOrderEvent, Side, Trade
from simulator.strategy import BaseStrategy

CAP = 95
Q0 = 0            # lots subtracted from each print size
TAU_US = 500_000  # score decay time constant
TH = 300          # |score| above which the target flips to +-CAP
TH0 = -1          # |score| below which the target goes flat (-1 = never)
SPW = 2.0         # weight of a print that hit a book with spread > 1
RELBAD = 0.3      # weight of a print of 1-2x the touch size it hit
R0 = 1.25         # weight of a print below 1/4 of the touch size it hit
QCAP = 1000       # print size cap
DONE = (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED)


class ImpulseFollower(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.pos = 0
        self.score = 0.0
        self.score_us = 0
        self.target = 0
        self.ioc_id = None
        self.ioc_sign = 0
        self.bid = self.ask = None
        self.bq = self.aq = 0

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        now = snapshot.timestamp_us
        pb, pa, pbq, paq = self.bid, self.ask, self.bq, self.aq
        bid = ask = None
        bq = aq = 0
        for o in snapshot.bids:
            if o.owner == 'market':
                if bid is None:
                    bid = o.price
                elif o.price != bid:
                    break
                bq += o.quantity
        for o in snapshot.asks:
            if o.owner == 'market':
                if ask is None:
                    ask = o.price
                elif o.price != ask:
                    break
                aq += o.quantity
        self.bid, self.ask, self.bq, self.aq = bid, ask, bq, aq
        if trades:
            agg = {}
            for t in trades:
                if t.taker_owner != 'market':
                    continue
                a = agg.get(t.taker_order_id)
                if a is None:
                    agg[t.taker_order_id] = [1 if t.book_side == 'ask' else -1, t.quantity]
                else:
                    a[1] += t.quantity
            if agg:
                self.score *= exp(-(now - self.score_us) / TAU_US)
                self.score_us = now
                wide = pb is not None and pa is not None and pa - pb > 1
                for s, q in agg.values():
                    w = min(q, QCAP) - Q0
                    if w <= 0:
                        continue
                    if wide:
                        w *= SPW
                    tq = paq if s > 0 else pbq
                    if tq > 0:
                        if q < 0.25 * tq:
                            w *= R0
                        elif tq <= q < 2 * tq:
                            w *= RELBAD
                    self.score += s * w
                x = self.score
                if x > TH:
                    self.target = CAP
                elif x < -TH:
                    self.target = -CAP
                elif TH0 > 0 and abs(x) < TH0:
                    self.target = 0
        if TH0 > 0 and self.target and now - self.score_us > 0:
            x = self.score * exp(-(now - self.score_us) / TAU_US)
            if abs(x) < TH0:
                self.target = 0
        self.execute()

    def execute(self) -> None:
        if self.ioc_id is not None or self.bid is None or self.ask is None:
            return
        d = self.target - self.pos
        if d == 0:
            return
        if d > 0:
            self.ioc_sign = 1
            self.ioc_id = self.send_order(OrderRequest(side=Side.BUY, price=self.ask, quantity=d, order_type=OrderType.IOC))
        else:
            self.ioc_sign = -1
            self.ioc_id = self.send_order(OrderRequest(side=Side.SELL, price=self.bid, quantity=-d, order_type=OrderType.IOC))

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        if event.order_id != self.ioc_id:
            return
        if event.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
            self.pos += self.ioc_sign * event.filled_quantity
        if event.status in DONE:
            self.ioc_id = None
