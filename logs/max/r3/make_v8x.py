"""Build strategies/strategy_v8x.py: strategy_v8 + DirectionModelX (33 base + 8 extra features).

Usage: uv run python logs/max/r3/make_v8x.py [weights.json]
weights.json: {"0.25": {"b":..,"w":[41]}, "2.0": {"b":..,"w":[41]}}; without it the extra weights are 0
and the base weights are those of strategy_v8 (only for recording features).
"""

import json
import pathlib
import re
import sys

src = pathlib.Path('strategies/strategy_v8.py').read_text(encoding='utf-8')
NX = 8
EXTRA = '''

class DirectionModelX(DirectionModel):
    """DirectionModel + 8 features: mid move 1 s / 5 s, 5 s range, big-aggressor flow, taker intensity,
    gap to the second level on each side, time since the last 1 s move >= 5 ticks."""

    INTERCEPT_X = INTERCEPT_X_V
    WX = WX_V
    BIG = 300

    def __init__(self):
        super().__init__()
        self.q1 = deque()
        self.q5 = deque()
        self.qmax = deque()
        self.qmin = deque()
        self.big = 0.0
        self.cnt = 0.0
        self.t_x = 0
        self.jump_us = -(10**15)
        self.gb = self.ga = 0

    def _decay_x(self, now):
        if now != self.t_x:
            d = exp(-(now - self.t_x) * 1e-6)
            self.big *= d
            self.cnt *= d
            self.t_x = now

    def update(self, snap, trades):
        now = snap.timestamp_us
        if trades:
            takers = {}
            for t in trades:
                if t.taker_owner != 'market':
                    continue
                lv = takers.get(t.taker_order_id)
                s = 1 if t.book_side == 'ask' else -1
                if lv is None:
                    takers[t.taker_order_id] = [s, t.quantity]
                else:
                    lv[1] += t.quantity
            if takers:
                self._decay_x(now)
                for s, q in takers.values():
                    self.cnt += 1.0
                    if q >= self.BIG:
                        self.big += s * q * 1e-3
        ok = super().update(snap, trades)
        if ok:
            m = self.mid
            q1, q5, qx, qn = self.q1, self.q5, self.qmax, self.qmin
            q1.append((now, m))
            q5.append((now, m))
            while now - q1[0][0] > 1_000_000:
                q1.popleft()
            while now - q5[0][0] > 5_000_000:
                q5.popleft()
            while qx and qx[-1][1] <= m:
                qx.pop()
            qx.append((now, m))
            while now - qx[0][0] > 5_000_000:
                qx.popleft()
            while qn and qn[-1][1] >= m:
                qn.pop()
            qn.append((now, m))
            while now - qn[0][0] > 5_000_000:
                qn.popleft()
            if abs(m - q1[0][1]) >= 5:
                self.jump_us = now
            bids, asks = snap.bids, snap.asks
            self.gb = self.ga = 0
            for v in bids:
                if v.owner == 'market' and v.price < self.bid:
                    self.gb = self.bid - v.price
                    break
            for v in asks:
                if v.owner == 'market' and v.price > self.ask:
                    self.ga = v.price - self.ask
                    break
        return ok

    def predict(self):
        if not self.stale or not self.on:
            return self.pred
        super().predict()
        now = self.us
        self._decay_x(now)
        m = self.mid
        ext = (m - self.q1[0][1], m - self.q5[0][1], self.qmax[0][1] - self.qmin[0][1], self.big, self.cnt * 0.1,
               min(self.gb, 5), min(self.ga, 5), min(10.0, (now - self.jump_us) * 1e-6))
        self.x = self.x + ext
        p = self.INTERCEPT_X
        for w, v in zip(self.WX, self.x):
            p += w * v
        c = self.CLIP
        self.pred = c if p > c else -c if p < -c else p
        return self.pred

'''
wfile = sys.argv[1] if len(sys.argv) > 1 else None
if wfile:
    W = json.load(open(wfile))
    b, w = W['0.25']['b'], W['0.25']['w']
    bl, wl = W['2.0']['b'], W['2.0']['w']
else:
    import importlib.util
    spec = importlib.util.spec_from_file_location('v8', 'strategies/strategy_v8.py')
    v8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v8)
    b, w = v8.DirectionModel.INTERCEPT, list(v8.DirectionModel.W) + [0.0] * NX
    bl, wl = v8.LONG_MODELS[2][0], list(v8.LONG_MODELS[2][1]) + [0.0] * NX
extra = EXTRA.replace('INTERCEPT_X_V', repr(float(b))).replace('WX_V', '(' + ', '.join(repr(float(x)) for x in w) + ')')
i = src.index('\ndef _POS')
s = src[:i] + extra + src[i:]
assert s.count('DirectionModel()') == 1
s = s.replace('DirectionModel()', 'DirectionModelX()')
s = re.sub(r'LONG_MODELS = \{\n    2: \([^\n]+\n', 'LONG_MODELS = {\n    2: (' + repr(float(bl)) + ', (' + ', '.join(repr(float(x)) for x in wl) + ')),\n', s, count=1)
out = 'strategies/strategy_v8x.py' if not wfile else (sys.argv[2] if len(sys.argv) > 2 else 'strategies/strategy_v8x.py')
pathlib.Path(out).write_text('# v8 + DirectionModelX (41 features), weights: ' + (wfile or 'v8 base, extra = 0') + '\n' + s, encoding='utf-8')
print('written', out)
