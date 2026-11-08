import numpy as np, sys, itertools
sys.path.insert(0,'logs/cannon/imp'); from sim import data, run_targets
from es import feats
def score(F, q0, tau, spw=1.0, relbad=1.0, mode='lin'):
    q=F['q']; s=F['s']
    w=np.maximum(q-q0,0.0)
    if mode=='sqrt': w=np.sqrt(w)
    w=w*np.where(F['sp0']>1,spw,1.0)
    rel=q/np.maximum(F['tq'],1); w=w*np.where((rel>=1)&(rel<2),relbad,1.0)
    t=F['t']; S=np.empty(len(t)); x=0.0; lt=0.0
    for i in range(len(t)):
        x=x*np.exp(-(t[i]-lt)/tau)+s[i]*w[i]; lt=t[i]; S[i]=x
    return S
def targets(t,S,th,th0=None,cap=95):
    tg=np.zeros(len(S),int); cur=0
    for i,x in enumerate(S):
        if x>th: cur=cap
        elif x<-th: cur=-cap
        elif th0 is not None and abs(x)<th0: cur=0
        tg[i]=cur
    return t,tg
FC={}
def ev(r, half, **kw):
    if r not in FC: FC[r]=feats(r)
    F=FC[r]; E,_,_=data(r)
    th=kw.pop('th'); th0=kw.pop('th0',None)
    S=score(F,**kw); dt,dg=targets(F['t'],S,th,th0)
    t0,t1=(0,3e9) if half=='a' else (3e9,6e9)
    return run_targets(E,dt,dg,t0,t1)
if __name__=='__main__':
    res=[]
    for q0,tau,th in itertools.product([100,200,300],[0.5e6,2e6,5e6],[200,400,800]):
        a3=ev('round3_public','a',q0=q0,tau=tau,th=th); a2=ev('round2_public','a',q0=q0,tau=tau,th=th)
        res.append((min(a3['pnl'],a2['pnl']),a3['pnl'],a2['pnl'],q0,tau,th,a3['trades'],a3['avgabs']))
    for x in sorted(res)[::-1][:15]: print(x)
