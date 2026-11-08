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
