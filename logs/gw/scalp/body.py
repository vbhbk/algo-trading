

class _O:
    __slots__ = ('side', 'sign', 'price', 'rem', 'ioc', 'cxl')

    def __init__(self, side, price, rem, ioc):
        self.side, self.price, self.rem, self.ioc, self.cxl = side, price, rem, ioc, False
        self.sign = 1 if side == Side.BUY else -1


class GwScalp(BaseStrategy):
    """Take by IOC on a strong 0.25 s prediction, then recycle capacity with a passive exit at the touch.

    Cap: on each side, position + own resting orders on that side + IOC in flight never exceeds CAP.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dm = DirectionModel()
        self.pos = 0
        self.orders = {}
        self.ioc_us = -10 ** 15
        self.unw_us = -10 ** 15
        self.entry_us = 0

    def exposure(self, sign):
        e = sign * self.pos
        for o in self.orders.values():
            if o.sign == sign:
                e += o.rem
        return e

    def cancel_side(self, sign):
        for oid, o in self.orders.items():
            if o.sign == sign and not o.ioc and not o.cxl:
                o.cxl = True
                self.cancel_order(oid)

    def ioc(self, sg, q, now, slip=0):
        dm = self.dm
        side = Side.BUY if sg > 0 else Side.SELL
        px = dm.ask + slip if sg > 0 else dm.bid - slip
        oid = self.send_order(OrderRequest(side=side, price=px, quantity=q, order_type=OrderType.IOC))
        self.orders[oid] = _O(side, px, q, True)
        self.ioc_us = now

    def target(self):
        if not TARGET_K:
            return 0
        b, w = M2
        p = b
        for wi, v in zip(w, self.dm.x):
            p += wi * v
        t = int(round(TARGET_K * p))
        return max(-TARGET_MAX, min(TARGET_MAX, t))

    def on_order_book_change(self, snapshot, trades) -> None:
        dm = self.dm
        dm.update(snapshot, trades)
        if not dm.valid:
            return
        now = snapshot.timestamp_us
        p = dm.predict() if dm.on else 0.0
        if dm.on and (A50 or A1):
            x = dm.x
            for a, (b, w) in ((A50, M50), (A1, M1)):
                if a:
                    q = b
                    for wi, v in zip(w, x):
                        q += wi * v
                    p += a * q
        spread = dm.ask - dm.bid
        has_ioc = False
        for o in self.orders.values():
            if o.ioc:
                has_ioc = True
                break
        pos = self.pos
        # 1) take
        if dm.on and not has_ioc and (p >= TAKE_Z or p <= -TAKE_Z) and now - self.ioc_us >= TAKE_COOLDOWN_US:
            sg = 1 if p > 0 else -1
            room = CAP - self.exposure(sg)
            mine = 0
            for o in self.orders.values():
                if o.sign == sg and not o.ioc:
                    mine += o.rem
            if mine:
                self.cancel_side(sg)
            q = min(TAKE_MAX, room)
            if q > 0:
                self.ioc(sg, q, now, SLIP)
                return
        if pos == 0:
            for oid, o in self.orders.items():
                if not o.ioc and not o.cxl:
                    o.cxl = True
                    self.cancel_order(oid)
            return
        sp = 1 if pos > 0 else -1
        z = sp * p
        # 2) unwind by IOC when the prediction turns strongly against the position
        if dm.on and not has_ioc and z < -UNWIND_Z and now - self.unw_us >= UNWIND_COOLDOWN_US:
            self.cancel_side(-sp)
            q = min(abs(pos), CAP - self.exposure(-sp))
            if q > 0:
                self.ioc(-sp, q, now)
                self.unw_us = now
                return
        # 3) passive exit at the touch on the reducing side
        tgt = self.target() if dm.on else 0
        want_q = abs(pos)
        if TARGET_K and sp * tgt > 0:
            want_q = max(0, abs(pos) - abs(tgt))
        want = None
        if want_q >= MIN_EXIT and spread <= EXIT_SPREAD_MAX and spread < GUARD_SPREAD:
            px = dm.ask if sp > 0 else dm.bid
            thr = EXIT_Z if now - self.entry_us < HOLD_US else EXIT_Z_HOLD
            if z <= thr:
                want = px
        if want is not None and EXIT_IOC:
            if dm.on and not has_ioc and now - self.unw_us >= UNWIND_COOLDOWN_US:
                self.cancel_side(-sp)
                q = min(want_q, CAP - self.exposure(-sp))
                if q > 0:
                    self.ioc(-sp, q, now)
                    self.unw_us = now
            return
        keep = None
        if want is None and want_q >= MIN_EXIT and spread <= EXIT_SPREAD_MAX and spread < GUARD_SPREAD and z < HOLD_Z:
            keep = dm.ask if sp > 0 else dm.bid   # hysteresis: a resting exit survives until z >= HOLD_Z
        for oid, o in self.orders.items():
            if o.ioc or o.cxl:
                continue
            if keep is not None and o.sign == -sp and o.price == keep and o.rem <= want_q:
                continue
            if o.sign == sp or want is None or o.price != want or o.rem > want_q:
                o.cxl = True
                self.cancel_order(oid)
            else:
                want = None   # already resting at the right price
        if want is not None and not has_ioc:
            q = min(want_q, CAP - self.exposure(-sp))
            for o in self.orders.values():
                if o.sign == -sp:
                    q = 0   # wait for the cancel to finish
            if q > 0:
                side = Side.BUY if sp < 0 else Side.SELL
                oid = self.send_order(OrderRequest(side=side, price=want, quantity=q, order_type=OrderType.LIMIT))
                self.orders[oid] = _O(side, want, q, False)

    def on_private_order_update(self, event) -> None:
        o = self.orders.get(event.order_id)
        if o is None:
            return
        st = event.status
        if st == OrderStatus.PARTIALLY_FILLED or st == OrderStatus.FILLED:
            before = self.pos
            self.pos += o.sign * event.filled_quantity
            o.rem = event.remaining_quantity
            if abs(self.pos) > abs(before) or self.pos * before < 0:
                self.entry_us = self.now_us()
        if st in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            if st == OrderStatus.REJECTED and o.cxl and not o.ioc:
                return
            del self.orders[event.order_id]
