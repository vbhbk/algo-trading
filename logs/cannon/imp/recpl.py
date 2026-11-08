"""Replay a tape with no orders; save cannon_ref ridge predictions per event: (t_us, on, pred_0.25s, pl_2s)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '.')
from simulator import main as sim_main
from simulator.simulator import ExchangeSimulator, SimulationConfig
from simulator.strategy import BaseStrategy
from strategies.cannon_ref import DirectionModel, LONG_MODELS

B, Wl = LONG_MODELS[2]
class Rec(BaseStrategy):
    def __init__(self):
        super().__init__(); self.dm = DirectionModel(); self.R = []
    def on_order_book_change(self, s, trades):
        dm = self.dm; dm.update(s, trades)
        if dm.on:
            p = dm.predict(); pl = B
            for wi, v in zip(Wl, dm.x): pl += wi * v
            self.R.append((s.timestamp_us, 1, p, pl))
        else:
            self.R.append((s.timestamp_us, 0, 0.0, 0.0))
    def on_private_order_update(self, e): pass
src, dst = sys.argv[1], sys.argv[2]
r = Rec()
cfg = SimulationConfig(max_position=sim_main.MAX_POSITION, max_potential_position=sim_main.MAX_POTENTIAL_POSITION,
                       position_breach_penalty_per_second=sim_main.POSITION_BREACH_PENALTY_PER_SECOND,
                       order_accept_delay_ms=sim_main.ORDER_ACCEPT_DELAY_MS)
ExchangeSimulator(strategy=r, config=cfg).run(Path(src))
np.save(dst, np.array(r.R, dtype=np.float64)); print(dst, len(r.R))
