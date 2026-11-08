"""python logs/gw/v8exec/mk.py TAG KEY=VAL ... -> strategies/gw_v8exec_TAG.py (from strategies/gw_v8exec.py)"""
import re, sys
tag = sys.argv[1]
src = open('strategies/gw_v8exec.py').read()
for kv in sys.argv[2:]:
    k, v = kv.split('=', 1)
    src, n = re.subn(rf'^{k} = [^\n#]*', f'{k} = {v}  ', src, count=1, flags=re.M)
    assert n == 1, k
open(f'strategies/gw_v8exec_{tag}.py', 'w').write(src)
