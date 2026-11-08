

class _O:
    __slots__ = ('side', 'sign', 'price', 'rem', 'ioc', 'cxl')

    def __init__(self, side, price, rem, ioc):
        self.side, self.price, self.rem, self.ioc, self.cxl = side, price, rem, ioc, False
        self.sign = 1 if side == Side.BUY else -1


class GwFlip(BaseStrategy):
    """Hold +-CAP in the direction of the short-horizon prediction (hysteresis), flip by IOC; optional passive legs."""

    def __init__(self) -> None:
        super().__init__()
        self.dm = DirectionModel()
        self.pos = 0
        self.cur = 0
        self.orders = {}
        self.ioc_us = -10 ** 15
        self.sig = 0.0

    def exposure(self, sign):
        e = sign * self.pos
        for o in self.orders.values():
            if o.sign == sign:
                e += o.rem
        return e

    def signal(self):
        dm = self.dm
        s = A25 * dm.predict()
        if A50:
            b, w = M50
            p = b
            for wi, v in zip(w, dm.x):
                p += wi * v
            s += A50 * p
        return s

    def on_order_book_change(self, snapshot, trades) -> None:
        dm = self.dm
        dm.update(snapshot, trades)
        if not dm.valid:
            return
        now = snapshot.timestamp_us
        if dm.on:
            s = self.sig = self.signal()
            if s > ENTER:
                self.cur = 1
            elif s < -ENTER:
                self.cur = -1
        else:
            s = self.sig = 0.0
        tgt = self.cur * CAP
        gap = tgt - self.pos
        has_ioc = any(o.ioc for o in self.orders.values())
        # IOC towards target while the signal still supports the move
        if gap and dm.on and not has_ioc and now - self.ioc_us >= IOC_COOLDOWN_US:
            sg = 1 if gap > 0 else -1
            if sg * s >= KEEP:
                side = Side.BUY if sg > 0 else Side.SELL
                # cancel resting orders on the same side first if they block room
                q = min(abs(gap), CAP - self.exposure(sg))
                if q < abs(gap):
                    for oid, o in self.orders.items():
                        if o.sign == sg and not o.ioc and not o.cxl:
                            o.cxl = True
                            self.cancel_order(oid)
                if q > 0:
                    px = dm.ask + SLIP if sg > 0 else dm.bid - SLIP
                    oid = self.send_order(OrderRequest(side=side, price=px, quantity=q, order_type=OrderType.IOC))
                    self.orders[oid] = _O(side, px, q, True)
                    self.ioc_us = now
        if PASSIVE:
            self.passive(s, now)

    def passive(self, s, now):
        dm = self.dm
        # desired passive quote per side: (price, qty) or None
        want = {1: None, -1: None}
        if dm.on and dm.ask - dm.bid == 1:
            for sg in (1, -1):
                px = dm.bid if sg > 0 else dm.ask
                edge = sg * (dm.mid + dm.pred - px)
                lean = sg * s
                if sg * self.cur < 0 or self.cur == 0:
                    ok = lean > PASS_EN            # pre-position for a flip in direction sg
                    lim = CAP
                else:
                    ok = edge > PASS_EDGE          # top up towards target
                    lim = CAP
                if ok:
                    q = lim - sg * self.pos
                    if q > 0:
                        want[sg] = (px, q)
        for oid, o in list(self.orders.items()):
            if o.ioc or o.cxl:
                continue
            w = want[o.sign]
            if w is None or w[0] != o.price:
                o.cxl = True
                self.cancel_order(oid)
            else:
                want[o.sign] = None if o.rem >= min(w[1], PASS_LOTS) * 0 + 1 else w
        for sg in (1, -1):
            w = want[sg]
            if w is None:
                continue
            if any(o.sign == sg and not o.ioc for o in self.orders.values()):
                continue
            q = min(w[1], PASS_LOTS, CAP - self.exposure(sg))
            if q <= 0:
                continue
            side = Side.BUY if sg > 0 else Side.SELL
            oid = self.send_order(OrderRequest(side=side, price=w[0], quantity=q, order_type=OrderType.LIMIT))
            self.orders[oid] = _O(side, w[0], q, False)

    def on_private_order_update(self, event) -> None:
        o = self.orders.get(event.order_id)
        if o is None:
            return
        st = event.status
        if st == OrderStatus.PARTIALLY_FILLED or st == OrderStatus.FILLED:
            self.pos += o.sign * event.filled_quantity
            o.rem = event.remaining_quantity
            if not o.ioc and abs(self.pos) >= CAP - ADOPT:
                self.cur = 1 if self.pos > 0 else -1
        if st in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            if st == OrderStatus.REJECTED and o.cxl and not o.ioc:
                return  # cancel rejected: order already gone or filled; wait for its final event
            del self.orders[event.order_id]
