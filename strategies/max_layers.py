"""Market maker with a two-level quote ladder that keeps queue position."""

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

QUOTE_SIZE = 20
MAX_POS = 50
SKEW_STEP = 20
L2_OFFSET = 1
L2_SIZE = 20
L3_OFFSET = 0
L3_SIZE = 20
QUEUE_KEEP = 1
MODIFY_SHRINK = 0
CAP_RESTING = 0
TOPUP_MIN = 0


class MaxLayers(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.inventory = 0
        self.orders: dict[str, list] = {}
        self.cancelling: set[str] = set()
        self.modifying: dict[str, int] = {}
        self.levels: list[tuple[int, int]] = [(0, QUOTE_SIZE)]
        if L2_OFFSET:
            self.levels.append((L2_OFFSET, L2_SIZE))
        if L3_OFFSET:
            self.levels.append((L3_OFFSET, L3_SIZE))

    def _cancel(self, oid: str) -> None:
        if oid not in self.cancelling:
            self.cancelling.add(oid)
            self.cancel_order(oid)

    def _send(self, side: Side, price: int, size: int, level: int) -> None:
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=size, order_type=OrderType.LIMIT))
        self.orders[oid] = [side, price, size, level, False]

    def _quote_side(self, side: Side, best: int | None, cap: int) -> None:
        live = [oid for oid, o in self.orders.items() if o[0] == side and oid not in self.cancelling]
        if best is None:
            for oid in live:
                self._cancel(oid)
            return
        sgn = -1 if side == Side.BUY else 1
        want: dict[int, tuple[int, int]] = {}
        for lvl, (off, size) in enumerate(self.levels):
            want.setdefault(best + sgn * off, (lvl, size))

        have: dict[int, list[str]] = {}
        for oid in live:
            o = self.orders[oid]
            w = want.get(o[1])
            if w is None or (not QUEUE_KEEP and w[0] != o[3]):
                self._cancel(oid)
            else:
                have.setdefault(o[1], []).append(oid)

        left = cap
        for lvl, (off, size) in enumerate(self.levels):
            px = best + sgn * off
            if want[px][0] != lvl:
                continue
            alloc = max(0, min(size, left))
            oids = have.get(px, [])
            used = 0
            for oid in oids:
                o = self.orders[oid]
                rem = min(self.modifying.get(oid, o[2]), o[2])
                if used >= alloc:
                    if MODIFY_SHRINK or CAP_RESTING:
                        self._cancel(oid)
                    else:
                        used += rem
                    continue
                if used + rem > alloc and MODIFY_SHRINK:
                    new = alloc - used
                    if o[4] and oid not in self.modifying and 0 < new < o[2]:
                        self.modifying[oid] = new
                        self.modify_order(oid, new)
                        rem = new
                    elif not o[4]:
                        pass
                used += rem
            if not oids:
                if alloc > 0:
                    self._send(side, px, alloc, lvl)
                    used = alloc
            elif TOPUP_MIN and used < TOPUP_MIN and alloc - used > 0 and len(oids) == 1:
                self._send(side, px, alloc - used, lvl)
                used = alloc
            left -= used if CAP_RESTING else alloc

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        bid = next((o.price for o in snapshot.bids if o.owner == 'market'), None)
        ask = next((o.price for o in snapshot.asks if o.owner == 'market'), None)
        if bid is None or ask is None:
            self._quote_side(Side.BUY, None, 0)
            self._quote_side(Side.SELL, None, 0)
            return

        spread = ask - bid
        inv = self.inventory
        skew = int(inv / SKEW_STEP)
        bid_px = bid + (1 if spread >= 3 else 0) - max(skew, 0)
        ask_px = ask - (1 if spread >= 3 else 0) - min(skew, 0)
        bid_px = min(bid_px, ask - 1)
        ask_px = max(ask_px, bid + 1)

        self._quote_side(Side.BUY, bid_px, MAX_POS - inv)
        self._quote_side(Side.SELL, ask_px, MAX_POS + inv)

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        st = event.status
        oid = event.order_id
        o = self.orders.get(oid)
        if o is None:
            return
        if st == OrderStatus.ACCEPTED:
            o[4] = True
            return
        if st in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
            q = event.filled_quantity
            self.inventory += q if o[0] == Side.BUY else -q
            o[2] = event.remaining_quantity
        elif st == OrderStatus.MODIFIED:
            o[2] = event.remaining_quantity
            self.modifying.pop(oid, None)
            return
        elif st == OrderStatus.REJECTED and oid in self.modifying and o[4]:
            self.modifying.pop(oid)
            return
        if st in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            self.orders.pop(oid, None)
            self.cancelling.discard(oid)
            self.modifying.pop(oid, None)
