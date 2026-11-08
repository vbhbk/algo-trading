"""Build a gw_flip variant: python logs/gw/make_flip.py out.py KEY=VAL ..."""
import json, sys
src = open('strategies/strategy_v8_r3.py').read().split('\n')
a = next(i for i, l in enumerate(src) if l.startswith('class DirectionModel'))
b = next(i for i, l in enumerate(src) if l.startswith('class _Ord'))
M = json.load(open('logs/gw/models_r3a.json'))
P = dict(CAP=95, ENTER=0.6, KEEP=0.0, IOC_COOLDOWN_US=20000, SLIP=0, A25=1.0, A50=0.0, PASSIVE=False,
         PASS_EN=0.3, PASS_EDGE=0.15, PASS_LOTS=190, ADOPT=5, PULL_EDGE=0.1)
for kv in sys.argv[2:]:
    k, v = kv.split('='); P[k] = eval(v)
hdr = ['# gw_flip: hold +-CAP in the predicted direction (hysteresis on ridge 0.25 s / 0.5 s, fit on round3_a), flip by IOC',
       'from bisect import bisect_left, bisect_right', 'from math import exp, log1p', '',
       'from simulator.models import OrderRequest, OrderStatus, OrderType, Side', 'from simulator.strategy import BaseStrategy', '']
hdr += [f'{k} = {v!r}' for k, v in P.items()]
hdr.append(f"M50 = ({M['0.5'][0]!r}, {tuple(M['0.5'][1])!r})")
open(sys.argv[1], 'w').write('\n'.join(hdr) + '\n\n\n' + '\n'.join(src[a:b]).rstrip() + '\n' + open('logs/gw/flip_body.py').read())
