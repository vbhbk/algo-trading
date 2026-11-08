"""mk.py base.py fit.json out.py header: put refit ridge weights into a v8-family strategy."""
import json, re, sys
base, fj, out, head = sys.argv[1:5]
t = open(base).read(); f = json.load(open(fj))
s, l = f['0.25'], f['2.0']
t, n1 = re.subn(r'^    INTERCEPT = .*$', f"    INTERCEPT = {s['b']!r}", t, flags=re.M)
t, n2 = re.subn(r'^    W = \(.*\)$', f"    W = {tuple(s['w'])!r}", t, flags=re.M)
t, n3 = re.subn(r'^    2: \(.*\),$', f"    2: ({l['b']!r}, {tuple(l['w'])!r}),", t, flags=re.M)
assert n1 == n2 == n3 == 1, (n1, n2, n3)
open(out, 'w').write('# ' + head + '\n' + t)
