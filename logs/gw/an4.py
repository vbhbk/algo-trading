import numpy as np, json
exec(open('logs/gw/an3.py').read().split('for f in')[0].replace("X=None","X=d['X'][ok].astype(np.float64)"))
M={float(k):(v[0],np.array(v[1])) for k,v in json.load(open('logs/gw/models_r3a.json')).items()}
def pred(X,H): b,w=M[H]; return np.nan_to_num(b+X@w)
def run2(t,bid,ask,tg,lat=0.005):
    j=np.minimum(np.searchsorted(t,t+lat),len(t)-1); ch=np.flatnonzero(np.diff(np.concatenate([[0],tg]))!=0)
    pos=0;cash=0.0;n=0
    for i in ch:
        k=j[i]; d=tg[i]-pos
        if d==0: continue
        cash-=d*(ask[k] if d>0 else bid[k]); pos=tg[i]; n+=1
    return cash+pos*(bid[-1]+ask[-1])/2,n
for f in ['round3_a','round3_b','round2_public']:
    t,bid,ask,pl,p,X=load(f)
    P={H:pred(X,H) for H in [0.25,0.5,1,2]}
    print('==',f)
    for name,sig in [('p25',P[0.25]),('p50',P[0.5]),('p1',P[1]),('p2',P[2]),('p25+p1',P[0.25]+P[1]),('p50+p2',P[0.5]+0.5*P[2])]:
        row=[]
        for en in [0.4,0.6,0.8,1.0,1.3,1.7]:
            r,n=run2(t,bid,ask,95*hyst(sig,en)); row.append(f'{en}:{r/1e3:5.0f}k/{n}')
        print(f'  {name:7s} '+'  '.join(row))
    # two-level
    s=P[0.25]
    for e1,e2 in [(0.4,0.8),(0.5,1.0),(0.6,1.2)]:
        tg=50*hyst(s,e1)+45*hyst(s,e2); r,n=run2(t,bid,ask,tg); print(f'  2lvl {e1}/{e2}: {r/1e3:.0f}k/{n}')
