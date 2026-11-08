"""Record per-event L1 book, market aggressive flow and v8 features + v8_r3f95 predictions.
Usage: uv run python logs/gw/record.py data/X.csv logs/gw/feat/X.npz"""
import importlib.util, sys
from pathlib import Path
import numpy as np
from simulator import main as sim_main
from simulator.simulator import ExchangeSimulator, SimulationConfig
from simulator.strategy import BaseStrategy

spec = importlib.util.spec_from_file_location('v8', 'strategies/strategy_v8_r3f95.py')
v8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v8)
B, W = v8.LONG_MODELS[2]

class Rec(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.dm = v8.DirectionModel()
        self.rows = []; self.X = []
    def on_order_book_change(self, snap, trades):
        dm = self.dm
        sv = 0; av = 0; nt = 0
        for t in trades:
            if t.taker_owner == 'market':
                s = 1 if t.book_side == 'ask' else -1
                sv += s * t.quantity; av += t.quantity; nt += 1
        ok = dm.update(snap, trades)
        if not ok:
            if sv:
                self.rows.append((snap.timestamp_us, np.nan, np.nan, 0, 0, sv, av, np.nan, np.nan, 0, 0))
                self.X.append((np.nan,) * 33)
            return
        d5b = sum(v.quantity for v in snap.bids if dm.bid - v.price < 5)
        d5a = sum(v.quantity for v in snap.asks if v.price - dm.ask < 5)
        if dm.on:
            p = dm.predict(); pl = B + sum(w * v for w, v in zip(W, dm.x)); x = dm.x
        else:
            p = pl = np.nan; x = (np.nan,) * 33
        self.rows.append((snap.timestamp_us, dm.bid, dm.ask, dm.bq, dm.aq, sv, av, p, pl, d5b, d5a))
        self.X.append(x)
    def on_private_order_update(self, e): pass

src, dst = sys.argv[1], sys.argv[2]
r = Rec()
cfg = SimulationConfig(max_position=sim_main.MAX_POSITION, max_potential_position=sim_main.MAX_POTENTIAL_POSITION,
                       position_breach_penalty_per_second=sim_main.POSITION_BREACH_PENALTY_PER_SECOND,
                       order_accept_delay_ms=sim_main.ORDER_ACCEPT_DELAY_MS)
ExchangeSimulator(strategy=r, config=cfg).run(Path(src))
a = np.array(r.rows, dtype=np.float64)
np.savez_compressed(dst, t=a[:,0], bid=a[:,1], ask=a[:,2], bq=a[:,3], aq=a[:,4], sv=a[:,5], av=a[:,6],
                    p=a[:,7], pl=a[:,8], d5b=a[:,9], d5a=a[:,10], X=np.array(self.X if False else r.X, dtype=np.float32))
print(src, len(a))
