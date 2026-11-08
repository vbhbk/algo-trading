# Does a money-aligned target/loss give better take decisions? (offline study)

## Setup
- Data: logs/gw/feat/*.npz (33 DirectionModel features). Train = round3_a + round2_public t<3000 s.
  Test = round3_b (r3b) and round2_public t>=3000 s (r2h2), reported separately. Never selected on test.
- Inner split for all choices = last 30% of each training part (t in [2100, 3000) of r3a and of r2 first half).
  The threshold is picked on inner val (3-point smoothed PnL curve over 32 thresholds = val trigger rates 0.6..0.002),
  then the model is refit on full train and the same absolute threshold is used on test.
- Decision: long trigger if su>=thr and su>sd, short if sd>=thr and sd>su. Signed models: su=p, sd=-p.
  Pair models (b): su=E[long taker gain], sd=E[short taker gain]. Classifiers (c): su=P(up)-P(dn), sd=-su.
- Sim = an3.hyst + an4.run2 (hold +-95 by last trigger, flip by IOC at touch, 5 ms latency, marked to mid at end).
  Optimistic in absolute terms vs the exchange simulator, but the same for all variants.
- HGB: 150 trees, 15 leaves, lr 0.1, min_samples_leaf 1000, l2 1, every 3rd event. Ridge: all events, lam 1e-4 (standardised).
- Targets: d25 = mid(t+0.25)-mid(t); gl_H = mid(t+H)-ask(t+5ms), gs_H = bid(t+5ms)-mid(t+H) (H 0.5/1/2 s);
  up_H/dn_H = mid touches mid(t)+-1 within H.
- "band-avg" = mean test PnL over the 11 thresholds with val trigger rate 0.055..0.345 (ranking quality,
  independent of the threshold pick; not used for any choice). mo1/mo2 = markout per lot of the takes at 1 s / 2 s.

## Results (k$; offline sim)
| spec | thr (val rate) | r3b pnl / takes / mo1 / mo2 | r2h2 pnl / takes / mo1 / mo2 | sum | band-avg sum |
|---|---|---|---|---|---|
| ref: r3a ridge p25, 0.6 (production) | 0.60 | 270 / 1440 / +0.99 / +1.17 | 154 / 1396 / +0.48 / +0.60 | 424 | - |
| a ridge d25 (MSE) | 0.55 (.20) | 278 / 1502 / +0.97 / +1.17 | 163 / 1466 / +0.52 / +0.57 | 441 | 415 |
| **a HGB d25 (MSE)** seeds 0/1/2 | 0.80/0.79/0.52 | 298 / 287 / 294 (mo +1.27/+1.38 s0) | 222 / 218 / 193 (mo +0.92/+1.03 s0) | 520/506/487 | 438/439/436 |
| b ridge taker H=0.5 / 1 / 2 | | 255 / 256 / 230 | 125 / 167 / 177 | 380 / 423 / 407 | 396 / 385 / 363 |
| b HGB taker H=0.5 (seeds 0/1/2) | 0.53 (.14) | 269 / 270 / 264 (mo +1.27/+1.41) | 204 / 204 / 193 (mo +1.09/+1.11) | 473/474/457 | 402/401/393 |
| b HGB taker H=1 / 2 | | 263 / 172 | 172 / 127 | 435 / 299 | 358 / 267 |
| c logit up/dn H=0.5 / 1 | | 241 / 229 | 174 / 148 | 415 / 378 | 419 / 390 |
| c HGB-clf up/dn H=0.5 (s0/1/2) / H=1 | | 238 / 242 / 250 / 239 | 191 / 182 / 199 / 157 | 429/424/448 / 396 | 384/386/389 / 342 |
| d ridge d25 Huber | | 259 | 152 | 410 | 359 |
| d ridge d25 clip +-1 / +-2 / +-4 | | 227 / 247 / 263 | 166 / 169 / 170 | 393 / 417 / 433 | 393 / 400 / 407 |
| d HGB d25 clip +-1 | 0.32 (.29) | 271 / 2114 / +0.62 / +0.68 | 182 / 1958 / +0.42 / +0.41 | 453 | 452 |
| **d HGB d25 clip +-2** seeds 0/1/2 | 0.76 (.115) all | 299 / 294 / 301 (1030 takes, mo +1.34/+1.55 s0) | 210 / 211 / 212 (900 takes, mo +0.92/+1.11 s0) | 510/505/513 | 453/457/449 |
| d ridge taker clip3 H=0.5 / 1 | | 274 / 272 | 144 / 153 | 418 / 425 | 416 / 416 |
| d HGB taker clip3 H=0.5 / 1 | | 258 / 228 | 213 / 157 | 471 / 385 | 407 / 347 |
| d HGB taker H=1 absolute_error | | 261 | 177 | 437 | 252 |
| d HGB cost-sensitive clf (label gain>0, weight abs(gain)) H=1 / 2 | | 184 / 180 | 116 / 128 | 300 / 308 | 321 / 281 |
| d HGB d25 absolute_error (first pass) | degenerate: median of d25 = 0, ~1 take | | | | |

R2 of the d25 regression on test (reference): r3a ridge 0.228 / 0.164; refit ridge 0.226 / 0.170;
HGB d25 0.300 / 0.298 (all 3 seeds within 0.003).

## Findings
1. Money-aligned targets do NOT help. Taker-gain regression (b) is worse than d25 for both learners, and gets worse
   with H (H=2: HGB 299k vs 510k). Classification of 1-tick moves (c) is worse than regressing d25.
   The cost-sensitive money-weighted classifier (d) is clearly the worst. Longer/noisier targets (std of gl2 ~4.5 ticks
   vs 1.4 for d25) dilute the short-term signal; the 0.25 s target is the best proxy, even for 1-2 s markouts.
2. The real gain is NONLINEARITY: HGB on d25 has R2 0.30 vs 0.17-0.23 for ridge, and +60..90k offline PnL over the
   refit ridge (+80..95k over production), on BOTH test sets. Per-lot markouts of its takes are higher
   (r3b +1.3/+1.5, r2h2 +0.9/+1.1 at 1 s / 2 s vs +1.0/+1.2 and +0.5/+0.6 for production).
3. Clipping d25 at +-2 ticks does not change the ranking much (band-avg 453 vs 438), but it makes the threshold pick
   stable: all 3 seeds pick 0.76 and give 505-513k, while plain MSE HGB picked 0.52 for one seed (487k).
   Huber / clip +-1 / absolute-error losses are worse or degenerate.
4. Threshold choice is the dominant noise source (val PnL curves are flat; picks drift 0.5-0.8). PnL differences
   below ~20-30k between variants are not meaningful.

## Recommendation
Keep the mid-change-at-0.25 s target. Replace the ridge p25 with a small HGB regressor (150 trees x 15 leaves) trained on
d25 clipped to +-2 ticks, using round3_a + round2_public first half, threshold ~0.75 (val trigger rate ~11%,
higher than the ridge's 0.6 because HGB scores are better calibrated in the tails). Do not spend more time on taker /
classification / cost-sensitive targets. Verify in the exchange simulator (logs/gw/gbm/ already builds HGB-in-strategy
variants; the tree eval is ~150 x 4 comparisons per event).

## Files
- common.py (load, targets, range-max, hyst2/run2/sim), run.py (specs, inner-val threshold pick, test eval),
  curves.py (this table + band averages), check.py (sanity: reproduces the an4 baseline).
- results.json / run.log = final run. results_v1.json / run_v1.log = first pass (narrower grid, unseeded HGB binning).
- Run: OMP_NUM_THREADS=1 uv run python logs/gw/target/run.py [spec ...] [--force]; then curves.py.
