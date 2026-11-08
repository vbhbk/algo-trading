"""blend.py out.json alpha fitA.json [fitB.json]: alpha*A + (1-alpha)*B; B defaults to cannon_ref weights."""
import json, re, sys, ast
out, al, fa = sys.argv[1], float(sys.argv[2]), sys.argv[3]
A = json.load(open(fa))
t = open('strategies/cannon_ref.py').read()
B = {'0.25': {'b': float(re.search(r'^    INTERCEPT = (.*)$', t, re.M).group(1)),
              'w': list(ast.literal_eval(re.search(r'^    W = (\(.*\))$', t, re.M).group(1)))}}
m = ast.literal_eval(re.search(r'^    2: (\(.*\)),$', t, re.M).group(1))
B['2.0'] = {'b': m[0], 'w': list(m[1])}
if len(sys.argv) > 4: B = json.load(open(sys.argv[4]))
R = {h: {'b': al * A[h]['b'] + (1 - al) * B[h]['b'], 'w': [al * x + (1 - al) * y for x, y in zip(A[h]['w'], B[h]['w'])]} for h in ('0.25', '2.0')}
json.dump(R, open(out, 'w'))
