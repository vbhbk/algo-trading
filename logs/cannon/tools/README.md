Shared research data for cannon agents (read-only, do not delete):
- logs/cannon/rec_<round>.npz for round1, round2_public, round3_public: replay of the tape with no orders.
  E[:, :] per market event = (t_us, bid, ask, bid_qty_L1, ask_qty_L1, bid_depth_5ticks, ask_depth_5ticks, n_orders_bid_L1, n_orders_ask_L1)
  T[:, :] per trade = (t_us, sign (+1 = buyer-aggressor, hit the ask), qty, price, taker_order_id, event_index)
- logs/cannon/tools/load.py: load(r) -> E (invalid books forward-filled), T, ok mask; takers(T) -> per aggressive order
  (t_us, sign, total_qty, n_price_levels, event_index, first_price).
Halves: round3_a = round3_public t < 3000 s, round3_b = t >= 3000 s (round2: 3000 s too).
