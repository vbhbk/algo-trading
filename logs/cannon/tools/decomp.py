import pandas as pd, numpy as np, sys
sys.path.insert(0,'logs/cannon/tools'); from load import load
d=sys.argv[1]
tr=pd.read_csv(f'{d}/trades.csv')
u=tr[(tr.taker_owner=='user')|(tr.maker_owner=='user')].copy()
u['passive']=u.maker_owner=='user'
# our side: if we are maker, our side = opposite of taker
u['sgn']=np.where(u.passive, np.where(u.taker_side=='A',1,-1), np.where(u.taker_side=='B',1,-1))
# market mid series from recorded R3 (b half = t>=3000s, shift)
E,T,ok=load('round3_public'); t=E[:,0]-3_000_000_000; mid=(E[:,1]+E[:,2])/2
sel=t>=0; t=t[sel]; mid=mid[sel]
def m_at(x): return mid[np.clip(np.searchsorted(t,x,side='right')-1,0,None)]
u['mid0']=m_at(u.timestamp_us.values)
for h in [0,1,5,30]:
    u[f'mo{h}']=u.sgn*(m_at(u.timestamp_us.values+int(h*1e6+1))-u.price)*u.quantity
print(u.groupby('passive').agg(n=('quantity','size'),lots=('quantity','sum'),
   edge0=('mo0','sum'),mo1=('mo1','sum'),mo5=('mo5','sum'),mo30=('mo30','sum')))
