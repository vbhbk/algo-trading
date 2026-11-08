"""Replay a tape with a no-order strategy and save the DirectionModel feature vector at every event.

Usage: uv run python logs/max/r3/record.py data/round3_a.csv logs/max/r3/feat_round3_a.npz
Saves t (us), mid (all events with a valid book), and X / t_on / mid_on for events where the
model is live (spread <= 1), exactly as strategy_v8.DirectionModel computes them.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

from simulator import main as sim_main
from simulator.simulator import ExchangeSimulator, SimulationConfig
from simulator.strategy import BaseStrategy

spec = importlib.util.spec_from_file_location('v8', 'strategies/strategy_v8.py')
v8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v8)


class Recorder(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.dm = v8.DirectionModel()
        self.t, self.mid = [], []
        self.t_on, self.mid_on, self.X = [], [], []

    def on_order_book_change(self, snapshot, trades) -> None:
        dm = self.dm
        if not dm.update(snapshot, trades):
            return
        self.t.append(snapshot.timestamp_us)
        self.mid.append(dm.mid)
        if dm.on:
            dm.predict()
            self.t_on.append(snapshot.timestamp_us)
            self.mid_on.append(dm.mid)
            self.X.append(dm.x)

    def on_private_order_update(self, event) -> None:
        pass


src, dst = sys.argv[1], sys.argv[2]
rec = Recorder()
cfg = SimulationConfig(max_position=sim_main.MAX_POSITION, max_potential_position=sim_main.MAX_POTENTIAL_POSITION,
                       position_breach_penalty_per_second=sim_main.POSITION_BREACH_PENALTY_PER_SECOND,
                       order_accept_delay_ms=sim_main.ORDER_ACCEPT_DELAY_MS)
ExchangeSimulator(strategy=rec, config=cfg).run(Path(src))
np.savez_compressed(dst, t=np.array(rec.t, dtype=np.int64), mid=np.array(rec.mid),
                    t_on=np.array(rec.t_on, dtype=np.int64), mid_on=np.array(rec.mid_on),
                    X=np.array(rec.X, dtype=np.float32))
print(f'{src}: events {len(rec.t)}, model-live rows {len(rec.X)}, features {len(rec.X[0]) if rec.X else 0}')
