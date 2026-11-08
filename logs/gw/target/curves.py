# Summary table + rate-matched comparison. The threshold grid is the same set of inner-val trigger rates (QGRID) for
# every spec, so test PnL at a given index compares ranking quality independent of threshold-pick noise.
# Nothing here is used for selection. Usage: uv run python logs/gw/target/curves.py
import json, numpy as np
R = json.load(open('logs/gw/target/results.json'))
Q = np.geomspace(0.6, 0.002, 32)
band = [i for i in range(32) if 0.05 <= Q[i] <= 0.4]
print(f'band = val trigger rates {Q[band[-1]]:.3f}..{Q[band[0]]:.3f} ({len(band)} thresholds)')
print(f"{'spec':22s} {'thr':>6s} {'rate':>5s} {'val':>5s} | {'r3b':>5s} {'n':>5s} {'mo1':>5s} {'mo2':>5s} | {'r2h2':>5s} {'n':>5s} {'mo1':>5s} {'mo2':>5s} | {'sum':>5s} | band-avg r3b r2h2 sum | R2 r3b r2h2")
for k, r in R.items():
    a, b = r['r3b'], r['r2h2']
    s = f"{k:22s} {r['thr']:6.3f} {r.get('val_rate', np.nan):5.3f} {r.get('val_pnl', np.nan)/1e3:5.0f} | {a['pnl']/1e3:5.0f} {a['takes']:5d} {a['mo1']:+5.2f} {a['mo2']:+5.2f} | {b['pnl']/1e3:5.0f} {b['takes']:5d} {b['mo1']:+5.2f} {b['mo2']:+5.2f} | {(a['pnl']+b['pnl'])/1e3:5.0f}"
    if 'curve' in a:
        ba = np.mean([a['curve'][i] for i in band]) / 1e3; bb = np.mean([b['curve'][i] for i in band]) / 1e3
        s += f' | {ba:5.0f} {bb:5.0f} {ba+bb:5.0f}'
    if 'R2' in a: s += f" | {a['R2']:.3f} {b['R2']:.3f}"
    print(s)
