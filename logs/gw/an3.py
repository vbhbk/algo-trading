import numpy as np
def load(f):
    d=np.load(f'logs/gw/feat/{f}.npz'); ok=~np.isnan(d['bid'])
    t=d['t'][ok]/1e6; bid=d['bid'][ok]; ask=d['ask'][ok]
    pl=np.nan_to_num(d['pl'][ok]); p=np.nan_to_num(d['p'][ok]); X=None
    return t,bid,ask,pl,p,X
def hyst(sig,en):
    tg=np.zeros(len(sig)); cur=0
    up=sig>en; dn=sig<-en
    # vectorised hysteresis: last trigger wins
    trig=np.where(up,1,np.where(dn,-1,0)); idx=np.where(trig!=0,np.arange(len(sig)),0)
    idx=np.maximum.accumulate(idx); out=trig[idx]; out[~(np.arange(len(sig))>=np.argmax(trig!=0))]=0
    return out
def run(t,bid,ask,tg,cap=95,lat=0.005):
    j=np.searchsorted(t,t+lat)  # execution event index
    j=np.minimum(j,len(t)-1)
    ch=np.flatnonzero(np.diff(np.concatenate([[0],tg]))!=0)
    pos=0; cash=0.0; n=0
    # execute changes sequentially at delayed event
    last_exec=-1
    for i in ch:
        k=j[i]; new=tg[i]*cap; d=new-pos
        if d==0: continue
        px=ask[k] if d>0 else bid[k]; cash-=d*px; pos=new; n+=1
    mid=(bid[-1]+ask[-1])/2
    return cash+pos*mid, n
for f in ['round3_a','round3_b','round2_public']:
    t,bid,ask,pl,p,X=load(f)
    
    print('==',f)
    for name,sig in [('p',p),('pl',pl),('p+.5pl',p+0.5*pl)]:
        for en in [0.5,0.6,0.8,1.0,1.3]:
            r,n=run(t,bid,ask,hyst(sig,en)); print(f'  {name} en {en}: pnl {r:9.0f} trades {n}')
