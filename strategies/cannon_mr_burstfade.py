# Burst fade: after 1 s of market aggressive flow beyond THR, hold -sign(flow) x LOTS from DELAY to DELAY + HOLD.
from collections import deque

from simulator.models import OrderBookSnapshot, OrderRequest, OrderStatus, OrderType, PrivateOrderEvent, Side, Trade
from simulator.strategy import BaseStrategy

THR = 2500            # |signed market aggressive volume in the last 1 s|
DELAY_US = 3_000_000
HOLD_US = 60_000_000
LOTS = 95
CAP = 95
PASSIVE = False       # True: work the delta as a limit at our touch first, cross after PASSIVE_TTL_US
PASSIVE_TTL_US = 2_000_000
MAX_SPREAD = 3        # do not cross wider spreads
COOLDOWN_US = 100_000
DONE = (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED)


class BurstFade(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.flow = deque()
        self.flow_sum = 0
        self.events = deque()      # (start_us, dir)
        self.last_ev_us = -10**12
        self.last_ev_dir = 0
        self.dir = 0
        self.end_us = 0
        self.position = 0
        self.oid = None            # one working order at a time
        self.o_side = None
        self.o_rem = 0
        self.o_ioc = False
        self.o_us = 0
        self.cancel_sent = False
        self.last_send = -10**12

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        now = snapshot.timestamp_us
        for t in trades:
            if t.taker_owner == 'market':
                s = t.quantity if t.book_side == 'ask' else -t.quantity
                self.flow.append((t.timestamp_us, s))
                self.flow_sum += s
        while self.flow and self.flow[0][0] < now - 1_000_000:
            self.flow_sum -= self.flow.popleft()[1]
        f = self.flow_sum
        if f > THR or f < -THR:
            d = -1 if f > 0 else 1
            if d != self.last_ev_dir or now - self.last_ev_us >= 100_000:
                self.events.append((now + DELAY_US, d))
                self.last_ev_us, self.last_ev_dir = now, d
        while self.events and self.events[0][0] <= now:
            st, d = self.events.popleft()
            self.dir, self.end_us = d, st + HOLD_US
        if self.dir and now >= self.end_us:
            self.dir = 0
        if not snapshot.bids or not snapshot.asks:
            return
        bid = snapshot.bids[0].price
        ask = snapshot.asks[0].price
        target = self.dir * LOTS
        if target != self.work_target:
            self.work_target, self.work_us = target, now
        dev = target - self.position
        if self.oid is not None:
            # cancel a working order that no longer points toward the target, or a passive one past its TTL
            want = 1 if dev > 0 else -1 if dev < 0 else 0
            osign = 1 if self.o_side == Side.BUY else -1
            if not self.o_ioc and not self.cancel_sent and (want != osign or now - self.work_us >= PASSIVE_TTL_US
                                                            or (osign > 0 and self.o_px != bid) or (osign < 0 and self.o_px != ask)):
                self.cancel_order(self.oid)
                self.cancel_sent = True
            return
        if dev == 0 or now - self.last_send < COOLDOWN_US:
            return
        sign = 1 if dev > 0 else -1
        q = abs(dev)
        if sign * self.position + q > CAP:
            q = CAP - sign * self.position
        if q <= 0:
            return
        side = Side.BUY if sign > 0 else Side.SELL
        passive = PASSIVE and now - self.work_us < PASSIVE_TTL_US
        if passive:
            px = bid if sign > 0 else ask
            ioc = False
        else:
            if ask - bid > MAX_SPREAD:
                return
            px = ask if sign > 0 else bid
            ioc = True
        self.oid = self.send_order(OrderRequest(side=side, price=px, quantity=q,
                                                order_type=OrderType.IOC if ioc else OrderType.LIMIT))
        self.o_side, self.o_rem, self.o_ioc, self.o_us, self.o_px = side, q, ioc, now, px
        self.o_target = target
        self.cancel_sent = False
        self.last_send = now

    o_px = 0
    o_target = 0
    work_target = 0
    work_us = 0

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        if event.order_id != self.oid:
            return
        if event.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
            q = event.filled_quantity
            self.position += q if self.o_side == Side.BUY else -q
            self.o_rem = event.remaining_quantity
        if event.status in DONE:
            self.oid = None
