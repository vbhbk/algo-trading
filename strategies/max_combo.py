"""Market maker: quote the touch with the full free capacity, hard position cap."""

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


class MaxCombo(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.inventory = 0
        self.orders: dict[str, list] = {}  # id -> [side, price, remaining]
        self.cancelling: set[str] = set()

    def _cancel(self, oid: str) -> None:
        if oid not in self.cancelling:
            self.cancelling.add(oid)
            self.cancel_order(oid)

    def _quote_side(self, side: Side, px: int | None, size: int, l2_size: int) -> None:
        live = [oid for oid, o in self.orders.items() if o[0] == side and oid not in self.cancelling]
        want: list[tuple[int, int]] = []
        if px is not None:
            if size > 0:
                want.append((px, size))
            if L2_OFFSET and l2_size > 0:
                want.append((px - L2_OFFSET if side == Side.BUY else px + L2_OFFSET, l2_size))
        prices = {p for p, _ in want}
        kept: dict[int, list[str]] = {}
        for oid in live:
            p = self.orders[oid][1]
            if p in prices:
                kept.setdefault(p, []).append(oid)
            else:
                self._cancel(oid)
        if not want:
            return
        sign = 1 if side == Side.BUY else -1
        room = MAX_POS - sign * self.inventory - sum(o[2] for o in self.orders.values() if o[0] == side)
        for p, sz in want:
            oids = kept.get(p, [])
            have = sum(self.orders[oid][2] for oid in oids)
            add = sz - have if not oids or have <= sz - TOPUP_MIN else 0
            add = min(add, room)
            if add <= 0:
                continue
            oid = self.send_order(OrderRequest(side=side, price=p, quantity=add, order_type=OrderType.LIMIT))
            self.orders[oid] = [side, p, add]
            room -= add

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        bid = next((o.price for o in snapshot.bids if o.owner == 'market'), None)
        ask = next((o.price for o in snapshot.asks if o.owner == 'market'), None)
        if bid is None or ask is None:
            self._quote_side(Side.BUY, None, 0, 0)
            self._quote_side(Side.SELL, None, 0, 0)
            return

        spread = ask - bid
        inv = self.inventory
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
        buy_size = min(bq, MAX_POS - inv)
        sell_size = min(aq, MAX_POS + inv)
        if UNWIND_FULL and inv >= UNWIND_TH:
            sell_size = min(max(sell_size, round(UNWIND_FULL * inv)), MAX_POS + inv)
        elif UNWIND_FULL and -inv >= UNWIND_TH:
            buy_size = min(max(buy_size, round(UNWIND_FULL * -inv)), MAX_POS - inv)

        self._quote_side(Side.BUY, bid_px, buy_size, min(L2_SIZE, MAX_POS - inv - buy_size))
        self._quote_side(Side.SELL, ask_px, sell_size, min(L2_SIZE, MAX_POS + inv - sell_size))

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        st = event.status
        o = self.orders.get(event.order_id)
        if o is None:
            return
        if st in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
            q = event.filled_quantity
            self.inventory += q if o[0] == Side.BUY else -q
            o[2] = event.remaining_quantity
        if st in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            self.orders.pop(event.order_id, None)
            self.cancelling.discard(event.order_id)
