# Spread > 1 states: predictability and tradable value (session gw, spread2)

Scripts (all in logs/gw/spread2/, single core):
- record_all.py: replays a round with DirectionModel (strategy_v8_r3) with the spread gate removed (SPREAD_MAX huge), dumps
  the 33 dm.x features in all states + extras (gap per side vs last 1-tick touch, episode age, last aggressive trade size/age,
  depth within 3/5 ticks per side, distance to 2nd level, mid - pre-gap mid) -> feat_<round>.npz
- episodes.py (task 1), ml.py -> ml.out (tasks 2+3), robust.py -> robust.out (r3a-only fits, regularised GBM, 1000 s blocks).
Split: train round3_a + round2_public t<3000 s; test round3_b (r3b) and round2_public t>=3000 s (r2h2).
Markout: IOC at the touch on arrival (first event >= t+5 ms; entry = ask/bid there, so cost = cross the wider spread),
mark = mid at t+H, ticks per lot, max one take per 100 ms. Touch size at arrival is large (median 170-420 lots), size ignored.

## 1. Episodes (spread > 1)
| | r3a | r3b | r2 |
|---|---|---|---|
| episodes / s | 2.18 | 2.34 | 1.76 |
| time share | 0.17 | 0.20 | 0.16 |
| duration median / p90 / p99 (s) | .020/.22/.70 | .021/.25/.73 | .025/.25/.79 |
| spread at open 2 / 3 / >=4 | 85/12/4 % | 83/13/4 % | 83/12/5 % |
| opened by aggressive trade (always in gap dir) | 61 % | 60 % | 62 % |
| close resolves by OTHER side following into gap | 75 % | 74 % | 70 % |
| emptied side refills back | 27 % | 28 % | 34 % |
| close mid - pre-gap mid, in gap dir (mean) | +0.99 | +1.01 | +0.97 |
| close mid - open mid, in gap dir | +0.34 | +0.36 | +0.30 |
| mid(close+0.5 s) - close mid, gap dir | +0.63 | +0.60 | +0.26 |
| mid(close+2 s) - close mid, gap dir | +1.17 | +0.89 | +0.22 |
Gaps are continuation events: the spread nearly always closes on the gapped side (new mid ~+1 tick vs pre-gap mid).
After closing, round3 keeps drifting in gap direction, round2 mostly does not. Most episodes last < 25 ms (5 ms latency
= the book is still >1 at IOC arrival in ~70-80% of strong-signal cases).

## 2. Out-of-sample fit on spread>1 events (n: train 332k, r3b 235k, r2h2 197k)
| target | ridge r3b R2 / corr | GBM r3b | ridge r2h2 | GBM r2h2 |
|---|---|---|---|---|
| 0.25 s | .090 / .30 | .101 / .34 | .040 / .21 | .037 / .24 |
| 0.5 s | .053 / .24 | .039 / .24 | -.012 / .10 | -.058 / .10 |
| until spread closes | .048 / .23 | .028 / .22 | -.001 / .10 | .033 / .21 |
GBM (min_leaf 300) overfits badly (train R2 .43-.58); regularised GBM (leaf 2000, 15 leaves): r3b .066, r2h2 .010 at 0.5 s.
Gap-side-only model: R2 ~0 (the gap direction itself is already priced in at the opening event).

## 3. Tradable value (IOC in sign of 0.5 s prediction, ticks/lot; cost = entry - mid at signal ~1.3-1.7)
| model, threshold | set | takes/s | mk 0.5 s | mk 1 s | mk 2 s (se) |
|---|---|---|---|---|---|
| ridge |p|>=1.5 | r3b | 0.59 | +0.79 | +1.17 | +1.79 (.22) |
| ridge |p|>=1.5 | r2h2 | 0.49 | -0.12 | -0.66 | -0.99 (.38) |
| ridge |p|>=2 | r3b | 0.29 | +1.11 | +1.68 | +2.38 (.31) |
| ridge |p|>=2 | r2h2 | 0.24 | -0.22 | -1.17 | -1.70 (.63) |
| GBM |p|>=2 | r3b | 0.56 | +1.38 | +2.01 | +2.61 (.38) |
| GBM |p|>=2 | r2h2 | 0.50 | -0.06 | -0.10 | -0.22 (.44) |
| reg-GBM |p|>=1.5 | r2h2 | 0.45 | | +0.29 | +0.09 |
| rule: gap dir at open, spread 2 | r3b / r2h2 | | -.24 / -.44 | -.05 / -.41 | +.04 / -.45 |
| rule: gap dir at close (spread 1) | r3b / r2h2 | | +.07 / -.29 | +.29 / -.27 | +.36 / -.36 |
Models fit on round3_a only give the same picture: r3b +1.3 at 1 s, r2 first half ~0, r2 second half -0.1..-0.4.

## Recommendation: do NOT integrate (as a taker).
- There is real statistical predictability at 0.25 s (corr .30 r3b / .21 r2h2), but the move is smaller than the cost of
  crossing a >=2-tick spread. At 0.5 s+ it generalises only within round3: after costs it is clearly positive on r3b
  (+1.2..+2 ticks/lot at 1 s, would be worth very roughly +50..100k on r3b if capacity were free) and zero-to-negative on
  round2_public in both halves, even when round2's first half is in the training set. Sign flips between regimes = not
  robust; we cannot tell the scoring round's regime in advance.
- Capacity is the binding constraint anyway (baseline is pinned at +-95), so spread>1 takes would displace the better
  spread=1 takes.
- If someone wants to try it anyway: ridge 0.5 s model (ml.py features, models.pkl), take only |p|>=2, 20-45 lots, and
  accept it only if a full backtest beats gw_base95 on all three rounds (expect a gain on round3, a loss on round2).
- Robust facts usable without a model: a gap closes on the gapped side 70-75% of the time (new mid ~+1 tick vs the pre-gap
  mid in every round), so after a gap do not quote passively against the gap direction at the stale touch; the
  existing guard already pulls quotes at spread >= 3.
