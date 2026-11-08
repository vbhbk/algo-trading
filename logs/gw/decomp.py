import sys, numpy as np, pandas as pd
d=sys.argv[1]; f=sys.argv[2]
tr=pd.read_csv(f'{d}/trades.csv'); u=tr[(tr.taker_owner=='user')|(tr.maker_owner=='user')].copy()
u['passive']=u.maker_owner=='user'
# our side: if we are taker, side=taker_side; if maker, opposite
u['sg']=np.where(u.passive, np.where(u.taker_side=='B',-1,1), np.where(u.taker_side=='B',1,-1))
F=np.load(f'logs/gw/feat/{f}.npz'); ok=~np.isnan(F['bid']); t=F['t'][ok]; mid=(F['bid'][ok]+F['ask'][ok])/2
def mid_at(ts): return mid[np.clip(np.searchsorted(t,ts,side='right')-1,0,len(t)-1)]
m0=mid_at(u.timestamp_us.values)
print('total user fills',len(u),'volume',u.quantity.sum())
for H in [0,0.1,1,5,30]:
    mh=mid_at(u.timestamp_us.values+int(H*1e6))
    u[f'mo{H}']=u.sg*(mh-u.price)*u.quantity
for name,g in u.groupby('passive'):
    print('PASSIVE' if name else 'TAKER', 'n',len(g),'vol',g.quantity.sum(), ' markout(total $) at 0/0.1/1/5/30s:', [round(g[f'mo{H}'].sum()) for H in [0,0.1,1,5,30]])
g=u[~u.passive]
for H in [0.05,0.25,0.5,1,2,3,5,10,30,60]:
    mh=mid_at(g.timestamp_us.values+int(H*1e6)); print(f'  taker markout/lot H={H}s: {np.sum(g.sg*(mh-g.price)*g.quantity)/g.quantity.sum():+.3f}')
