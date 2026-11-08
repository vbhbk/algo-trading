"""Record DirectionModel's 33 features in ALL states (spread gate removed) + spread>1 extras.
Usage: OMP_NUM_THREADS=1 uv run python logs/gw/spread2/record_all.py data/X.csv logs/gw/spread2/feat_X.npz"""
import importlib.util, sys
from pathlib import Path
import numpy as np
from simulator import main as sim_main
from simulator.simulator import ExchangeSimulator, SimulationConfig
from simulator.strategy import BaseStrategy

spec = importlib.util.spec_from_file_location('v8', 'strategies/strategy_v8_r3.py')
v8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v8)

class DM(v8.DirectionModel):
    SPREAD_MAX = 10**9

def lvl(views, sgn, best, n=3):
    """sizes of the first 3 price levels and distance (ticks) best->2nd level, market orders only"""
    sizes = {}
    for v in views:
        if v.owner != 'market':
            continue
        d = sgn * (v.price - best)
        if d >= 10:
            break
        sizes[d] = sizes.get(d, 0) + v.quantity
    ks = sorted(sizes)
    gap2 = ks[1] if len(ks) > 1 else 10
    d3 = sum(q for k, q in sizes.items() if k < 3)
    d5 = sum(q for k, q in sizes.items() if k < 5)
    return gap2, d3, d5

class Rec(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.dm = DM()
        self.rows = []; self.X = []
        self.rb = self.ra = 0     # touch at the last spread==1 state
        self.ep_start = 0         # time spread>1 began
        self.last_sq = 0.0; self.last_t = 0   # last aggressive market trade (signed qty), time
        self.prev_sp = 1
    def on_order_book_change(self, snap, trades):
        dm = self.dm
        sv = 0; av = 0
        for t in trades:
            if t.taker_owner == 'market':
                s = 1 if t.book_side == 'ask' else -1
                sv += s * t.quantity; av += t.quantity
        now = snap.timestamp_us
        if sv:
            self.last_sq = sv; self.last_t = now
        ok = dm.update(snap, trades)
        if not ok:
            return
        bid, ask = dm.bid, dm.ask; sp = ask - bid
        if sp <= 1:
            self.rb, self.ra = bid, ask
        elif self.prev_sp <= 1:
            self.ep_start = now
        self.prev_sp = sp
        dm.predict(); x = dm.x
        g2b, d3b, d5b = lvl(snap.bids, -1, bid)
        g2a, d3a, d5a = lvl(snap.asks, 1, ask)
        gb = (self.rb - bid) if self.rb else 0   # >0: bid side emptied (bid dropped)
        ga = (ask - self.ra) if self.ra else 0   # >0: ask side emptied (ask rose)
        age = (now - self.ep_start) * 1e-6 if sp > 1 else 0.0
        self.rows.append((now, bid, ask, dm.bq, dm.aq, sv, av, d5b, d5a, d3b, d3a, g2b, g2a, gb, ga, age,
                          self.last_sq, min(5.0, (now - self.last_t) * 1e-6), self.rb, self.ra))
        self.X.append(x)
    def on_private_order_update(self, e): pass

COLS = 't bid ask bq aq sv av d5b d5a d3b d3a g2b g2a gb ga age lastq lastdt rb ra'.split()
src, dst = sys.argv[1], sys.argv[2]
r = Rec()
cfg = SimulationConfig(max_position=sim_main.MAX_POSITION, max_potential_position=sim_main.MAX_POTENTIAL_POSITION,
                       position_breach_penalty_per_second=sim_main.POSITION_BREACH_PENALTY_PER_SECOND,
                       order_accept_delay_ms=sim_main.ORDER_ACCEPT_DELAY_MS)
ExchangeSimulator(strategy=r, config=cfg).run(Path(src))
a = np.array(r.rows, dtype=np.float64)
np.savez_compressed(dst, X=np.array(r.X, dtype=np.float32), **{c: a[:, i] for i, c in enumerate(COLS)})
print(src, len(a))
