"""Replay a tape with no orders; save per-event top of book and all market trades.
uv run python logs/cannon/tools/rec.py data/round3_public.csv logs/cannon/rec_r3.npz"""
import sys
from pathlib import Path
import numpy as np
from simulator import main as sim_main
from simulator.simulator import ExchangeSimulator, SimulationConfig
from simulator.strategy import BaseStrategy


class Rec(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.E = []
        self.T = []

    def on_order_book_change(self, s, trades):
        b = s.bids; a = s.asks
        if b:
            bp = b[0].price; bq = 0; bd = 0; bn = 0
            for o in b:
                if bp - o.price >= 5: break
                bd += o.quantity
                if o.price == bp: bq += o.quantity; bn += 1
        else:
            bp = bq = bd = bn = 0
        if a:
            ap = a[0].price; aq = 0; ad = 0; an = 0
            for o in a:
                if o.price - ap >= 5: break
                ad += o.quantity
                if o.price == ap: aq += o.quantity; an += 1
        else:
            ap = aq = ad = an = 0
        self.E.append((s.timestamp_us, bp, ap, bq, aq, bd, ad, bn, an))
        for t in trades:
            self.T.append((t.timestamp_us, 1 if t.book_side == 'ask' else -1, t.quantity, t.price, int(t.taker_order_id) if t.taker_order_id.isdigit() else -1, len(self.E) - 1))

    def on_private_order_update(self, e):
        pass


src, dst = sys.argv[1], sys.argv[2]
r = Rec()
cfg = SimulationConfig(max_position=sim_main.MAX_POSITION, max_potential_position=sim_main.MAX_POTENTIAL_POSITION,
                       position_breach_penalty_per_second=sim_main.POSITION_BREACH_PENALTY_PER_SECOND,
                       order_accept_delay_ms=sim_main.ORDER_ACCEPT_DELAY_MS)
ExchangeSimulator(strategy=r, config=cfg).run(Path(src))
np.savez_compressed(dst, E=np.array(r.E, dtype=np.int64), T=np.array(r.T, dtype=np.int64))
print(dst, len(r.E), len(r.T))
