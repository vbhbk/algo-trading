import numpy as np
def L(r):
    d=np.load(f'logs/cannon/mr/grid_{r}.npz'); b=d['bid'].astype(float); a=d['ask'].astype(float)
    b[0]=b[1]; a[0]=a[1]
    return b,a,d['flow'],d['vol']
def ema(x,n):
    al=1.0/n; y=np.empty_like(x); s=x[0]
    # fast via scipy lfilter
    from scipy.signal import lfilter
    y=lfilter([al],[1,-(1-al)],x,zi=[x[0]*(1-al)])[0]
    return y
def sim(tgt,b,a,band=0,lag=1,cap=95):
    """target in lots -> position (integer), trades at touch lag steps later; return pnl series"""
    tgt=np.clip(np.round(tgt),-cap,cap)
    n=len(tgt); m=(a+b)/2
    if band>0:
        p=np.zeros(n); cur=0.0
        for i in range(n):
            if abs(tgt[i]-cur)>=band: cur=tgt[i]
            p[i]=cur
        tgt=p
    pos=np.r_[np.zeros(lag),tgt[:-lag]]   # position held during [i,i+1)
    dq=np.diff(np.r_[0,pos])
    cost=np.where(dq>0,dq*(a-m),-dq*(m-b))
    pnl=pos[:-1]*np.diff(m)
    out=np.r_[0,np.cumsum(pnl-cost[:-1])]
    return out, np.abs(dq).sum()
