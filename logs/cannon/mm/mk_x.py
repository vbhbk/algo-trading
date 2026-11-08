# builds strategies/cannon_mm_x.py from cannon_ref.py with toggles (defaults reproduce cannon_ref)
src = open('strategies/cannon_ref.py').read()
def rep(a, b, n=1):
    global src
    assert src.count(a) == n, (a, src.count(a))
    src = src.replace(a, b)
rep('PULL_EDGE = 0.1      #', '''PASSIVE_MAX_SPREAD = 99   # mm: no resting quotes while spread > this
TOX_L = 1e9               # mm: new quote on side s only if s * long_pred >= -TOX_L (2 s model as toxicity filter)
TOX_L_CANCEL = 1e9        # mm: cancel a resting quote when s * long_pred < -TOX_L_CANCEL
KEEP_TICKS = 0            # mm: keep resting quotes up to this many ticks behind the touch (queue priority)
KEEP_SIZE = 1000          # mm: shrink a kept (behind-touch) quote to at most this size (modify keeps priority)
PULL_EDGE = 0.1      #''')
# long prediction stored
rep('''            t = int(round(TARGET_K * pl))''', '''            self.pl = pl
            t = int(round(TARGET_K * pl))''')
rep('''        self.dm = DirectionModel()''', '''        self.dm = DirectionModel()
        self.pl = 0.0''')
rep('''        if side == self.g2_block or not self.research_allow(side) or not self.dm.allow(1 if side == Side.BUY else -1, price):
            return''', '''        if side == self.g2_block or not self.research_allow(side) or not self.dm.allow(1 if side == Side.BUY else -1, price):
            return
        if (1 if side == Side.BUY else -1) * self.pl < -TOX_L:
            return''')
# cancel pass
rep('''            if buy:
                stale = o.price < bid or o.price >= ask
                adverse = flow < -ADVERSE_FLOW
            else:
                stale = o.price > ask or o.price <= bid
                adverse = flow > ADVERSE_FLOW''', '''            if buy:
                stale = o.price < bid - KEEP_TICKS or o.price >= ask
                adverse = flow < -ADVERSE_FLOW
                deep = o.price < bid
            else:
                stale = o.price > ask + KEEP_TICKS or o.price <= bid
                adverse = flow > ADVERSE_FLOW
                deep = o.price > ask
            if ask - bid > PASSIVE_MAX_SPREAD or (1 if buy else -1) * self.pl < -TOX_L_CANCEL:
                stale = True
            if deep and not stale and o.rem > KEEP_SIZE and not o.modify_pending:
                o.modify_pending = True
                self.modify_order(oid, KEEP_SIZE)''')
rep('''        if now < self.next_quote_us or bid is None or ask is None or ask <= bid:
            return''', '''        if now < self.next_quote_us or bid is None or ask is None or ask <= bid or ask - bid > PASSIVE_MAX_SPREAD:
            return''')
rep('''            mine = [o for o in active if o.side == side]''', '''            mine = [o for o in active if o.side == side and (o.price == price or o.accepted_us is None or o.cancel_pending)]''')
rep('''            qty = min(self.research_lots(), room, self.free_room(side, sign))''', '''            qty = min(self.research_lots(), room, self.free_room(side, sign))
            if KEEP_TICKS:
                qty = min(qty, room - sum(o.rem for o in active if o.side == side))''')
rep('class Round2MLMaker(BaseStrategy):', 'class CannonMM(BaseStrategy):')
open('strategies/cannon_mm_x.py', 'w').write(src)
src = open('strategies/cannon_mm_x.py').read()
rep('KEEP_TICKS = 0            #', '''SZ_BAND = 100000          # mm: a passive fill may move inventory at most SZ_BAND past the target (size from model)
SZ_SHRINK = 0             # mm: 1 = shrink resting quotes (modify keeps priority) when the band tightens
KEEP_TICKS = 0            #''')
rep('''    def free_room(self, side: Side, sign: int) -> int:''', '''    def band(self, sign: int) -> int:
        post = self.target + sign * SZ_BAND
        post = POSITION_CAP if post > POSITION_CAP else -POSITION_CAP if post < -POSITION_CAP else post
        b = sign * (post - self.position)
        return b if b > 0 else 0

    def free_room(self, side: Side, sign: int) -> int:''')
rep('''                room = self._room(sign)
                live =''', '''                room = self._room(sign)
                if SZ_SHRINK:
                    room = min(room, self.band(sign))
                live =''')
rep('''            room = self._room(sign)
            edge = sign * (fair - price)''', '''            room = min(self._room(sign), self.band(sign))
            edge = sign * (fair - price)''')
open('strategies/cannon_mm_x.py', 'w').write(src)
src = open('strategies/cannon_mm_x.py').read()
rep('KEEP_TICKS = 0            #', '''THIN_Q = 0                # mm: no resting quote at the touch while the market size at that level < THIN_Q
KEEP_TICKS = 0            #''')
rep('''            if ask - bid > PASSIVE_MAX_SPREAD or (1 if buy else -1) * self.pl < -TOX_L_CANCEL:
                stale = True''', '''            if ask - bid > PASSIVE_MAX_SPREAD or (1 if buy else -1) * self.pl < -TOX_L_CANCEL:
                stale = True
            if THIN_Q and not stale and o.price == (bid if buy else ask):
                lq = 0
                for v in (snapshot.bids if buy else snapshot.asks):
                    if v.price != o.price:
                        break
                    if v.owner == 'market':
                        lq += v.quantity
                if lq < THIN_Q:
                    stale = True''')
rep('''            queue = sum(o.quantity for o in levels if o.owner == 'market' and o.price == price)
            qty = min(''', '''            queue = sum(o.quantity for o in levels if o.owner == 'market' and o.price == price)
            if queue < THIN_Q:
                continue
            qty = min(''')
open('strategies/cannon_mm_x.py', 'w').write(src)
