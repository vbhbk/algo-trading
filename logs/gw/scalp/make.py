"""Build gw_scalp variant: python logs/gw/scalp/make.py strategies/gw_scalp_X.py KEY=VAL ..."""
import json, sys
src = open('strategies/strategy_v8_r3.py').read().split('\n')
a = next(i for i, l in enumerate(src) if l.startswith('class DirectionModel'))
b = next(i for i, l in enumerate(src) if l.startswith('class _Ord'))
M = json.load(open('logs/gw/models_r3a.json'))
P = dict(CAP=95, TAKE_Z=0.6, TAKE_MAX=190, TAKE_COOLDOWN_US=20000, HOLD_US=1_000_000, EXIT_Z=0.5,
         EXIT_Z_HOLD=1.5, EXIT_IOC=False, HOLD_Z=0.6, UNWIND_Z=0.6, UNWIND_COOLDOWN_US=50000, TARGET_K=0.0, TARGET_MAX=95,
         EXIT_SPREAD_MAX=2, GUARD_SPREAD=3, MIN_EXIT=1, SLIP=0, A50=0.0, A1=0.0)
for kv in sys.argv[2:]:
    k, v = kv.split('=', 1); P[k] = eval(v)
hdr = ['# gw_scalp: take on the 0.25 s ridge (IOC), recycle capacity by exiting passively at the touch',
       'from bisect import bisect_left, bisect_right', 'from math import exp, log1p', '',
       'from simulator.models import OrderRequest, OrderStatus, OrderType, Side', 'from simulator.strategy import BaseStrategy', '']
hdr += [f'{k} = {v!r}' for k, v in P.items()]
hdr.append(f"M50 = ({M['0.5'][0]!r}, {tuple(M['0.5'][1])!r})")
hdr.append(f"M1 = ({M['1'][0]!r}, {tuple(M['1'][1])!r})")
hdr.append(f"M2 = ({M['2'][0]!r}, {tuple(M['2'][1])!r})")
open(sys.argv[1], 'w').write('\n'.join(hdr) + '\n\n\n' + '\n'.join(src[a:b]).rstrip() + '\n' + open('logs/gw/scalp/body.py').read())
