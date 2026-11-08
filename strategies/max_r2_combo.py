"""max_vol + imb_L1 modify cap (max/r2-port) + short-term momentum cap for round 2."""

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
MOD_IMB_TH = -0.92  # side_sign * imb_L1 <= this -> side capped at flattening size via modify (<= -1 = off)
MOD_CAP_INC = 0  # increasing exposure still allowed when triggered
MOD_MAX_SPREAD = 100  # imb_L1 cap only when spread <= this
MOM_WIN_US = 5_000_000  # momentum look-back
MOM_TH = 8  # |mid move| over MOM_WIN that caps the side against the move at flattening size (1e9 = off)
MOM_HOLD_US = 2_000_000
ADAPT_CAP = 0.0  # >0: cap = min(ADAPT_CAP * taker-size EMA, ADAPT_L1 * L1-volume EMA) (max/robust 2d0379c; 0 = off)
ADAPT_L1 = 0.30
ADAPT_ALPHA = 0.002
ADAPT_MIN = 10


class MaxR2Combo(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.inventory = 0
        self.orders: dict[str, list] = {}  # id -> [side, price, remaining]
        self.cancelling: set[str] = set()
        self.modifying: dict[str, int] = {}
        self.cap = MAX_POS
        self.mids: deque = deque()
        self.guard_until = -1
        self.tmids: deque = deque()
        self.mom_until = -1
        self.mom_dir = 0
        self.tr_ema = 100.0
        self.l1_ema = 800.0
        self.n_trades = 0
        self.n_snaps = 0
        self.base_cap = MAX_POS
        self.topup = TOPUP_MIN

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
        for oid in live:
            p = self.orders[oid][1]
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
        room = self.cap - sign * self.inventory - sum(o[2] for o in self.orders.values() if o[0] == side)
        for p, sz in want:
            oids = kept.get(p, [])
            have = sum(self.orders[oid][2] for oid in oids)
            add = sz - have if not oids or have <= sz - self.topup else 0
            add = min(add, room)
            if add <= 0:
                continue
            if SPLIT_LEAD and not oids and add > SPLIT_LEAD:
                parts = (SPLIT_LEAD, add - SPLIT_LEAD)
            else:
                parts = (add,)
            for q in parts:
                oid = self.send_order(OrderRequest(side=side, price=p, quantity=q, order_type=OrderType.LIMIT))
                self.orders[oid] = [side, p, q]
            room -= add

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        if ADAPT_CAP and trades:
            by_taker: dict[str, int] = {}
            for t in trades:
                if t.taker_owner == 'market':
                    by_taker[t.taker_order_id] = by_taker.get(t.taker_order_id, 0) + t.quantity
            for q in by_taker.values():
                self.n_trades += 1
                self.tr_ema += max(ADAPT_ALPHA, 1.0 / self.n_trades) * (q - self.tr_ema)
        bid = next((o.price for o in snapshot.bids if o.owner == 'market'), None)
        ask = next((o.price for o in snapshot.asks if o.owner == 'market'), None)
        if bid is None or ask is None:
            self._quote_side(Side.BUY, None, 0, 0)
            self._quote_side(Side.SELL, None, 0, 0)
            return

        spread = ask - bid
        inv = self.inventory
        if ADAPT_CAP:
            bv0 = sum(o.quantity for o in snapshot.bids if o.owner == 'market' and o.price == bid)
            av0 = sum(o.quantity for o in snapshot.asks if o.owner == 'market' and o.price == ask)
            self.n_snaps += 1
            self.l1_ema += max(ADAPT_ALPHA * 0.1, 1.0 / self.n_snaps) * (0.5 * (bv0 + av0) - self.l1_ema)
            self.base_cap = max(ADAPT_MIN, min(MAX_POS, round(min(ADAPT_CAP * self.tr_ema, ADAPT_L1 * self.l1_ema))))
            self.topup = max(2, round(TOPUP_MIN * self.base_cap / MAX_POS))
            if VOL_TH >= 1e9:
                self.cap = self.base_cap
        if VOL_TH < 1e9:
            now = snapshot.timestamp_us
            m = 0.5 * (bid + ask)
            mids = self.mids
            mids.append((now, m))
            while now - mids[0][0] > VOL_WIN_US:
                mids.popleft()
            if abs(m - mids[0][1]) >= VOL_TH:
                self.guard_until = now + VOL_HOLD_US
            self.cap = VOL_POS if now < self.guard_until else self.base_cap
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
        if MOD_IMB_TH > -1.0 and spread <= MOD_MAX_SPREAD:
            bv = sum(o.quantity for o in snapshot.bids if o.owner == 'market' and o.price == bid)
            av = sum(o.quantity for o in snapshot.asks if o.owner == 'market' and o.price == ask)
            imb = (bv - av) / (bv + av) if bv + av > 0 else 0.0
            if imb <= MOD_IMB_TH:
                keep_buy = max(0, -inv) + MOD_CAP_INC
                buy_size = min(buy_size, keep_buy)
            if -imb <= MOD_IMB_TH:
                keep_sell = max(0, inv) + MOD_CAP_INC
                sell_size = min(sell_size, keep_sell)

        if MOM_TH < 1e9:
            now = snapshot.timestamp_us
            m = 0.5 * (bid + ask)
            tm = self.tmids
            if not tm or tm[-1][1] != m:
                tm.append((now, m))
            while len(tm) > 1 and tm[1][0] <= now - MOM_WIN_US:
                tm.popleft()
            d = m - tm[0][1]
            if abs(d) >= MOM_TH:
                self.mom_until = now + MOM_HOLD_US
                self.mom_dir = 1 if d > 0 else -1
            if now < self.mom_until:
                if self.mom_dir < 0:
                    keep_buy = min(keep_buy, max(0, -inv))
                    buy_size = min(buy_size, keep_buy)
                else:
                    keep_sell = min(keep_sell, max(0, inv))
                    sell_size = min(sell_size, keep_sell)

        self._quote_side(Side.BUY, bid_px, buy_size, min(L2_SIZE, cap - inv - buy_size), keep_buy)
        self._quote_side(Side.SELL, ask_px, sell_size, min(L2_SIZE, cap + inv - sell_size), keep_sell)

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        st = event.status
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
