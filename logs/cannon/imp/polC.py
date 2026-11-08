import numpy as np, sys, itertools
sys.path.insert(0,'logs/cannon/imp'); from sim import data, run_targets
from es import feats
from fitw2 import X_of, fit
from polB import targets
FC={}
def S_of(F, w, tau):
    t=F['t']; s=F['s']; S=np.empty(len(t)); x=0.0; lt=0.0
    for i in range(len(t)):
        x=x*np.exp(-(t[i]-lt)/tau)+s[i]*w[i]; lt=t[i]; S[i]=x
    return S
def ev(r, half, b, tau, th, th0=None, cap=95):
    if r not in FC: FC[r]=feats(r)
    F=FC[r]; E,_,_=data(r)
    w=np.maximum(X_of(F)@b,0)
    S=S_of(F,w,tau); dt,dg=targets(F['t'],S,th,th0,cap)
    t0,t1=(0,3e9) if half=='a' else (3e9,6e9)
    return run_targets(E,dt,dg,t0,t1)
if __name__=='__main__':
    b=fit(1.0)
    res=[]
    for tau,th in itertools.product([0.1e6,0.2e6,0.35e6,0.5e6,1e6],[1,1.5,2,3,4,6]):
        a3=ev('round3_public','a',b,tau,th); a2=ev('round2_public','a',b,tau,th)
        res.append((a3['pnl']+a2['pnl'],a3['pnl'],a2['pnl'],tau,th,a3['trades'],a2['trades']))
    res.sort(reverse=True)
    for x in res[:12]: print(x)
