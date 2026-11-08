import numpy as np, sys
sys.path.insert(0,'logs/cannon/imp'); from sim import data, run_targets
import polE
from polB import targets
P=dict(polE.P); P['r0']=1.25
def S2(F,w,tau,tau2):
    t=F['t']; s=F['s']; S=np.empty(len(t)); A=np.empty(len(t)); x=0.0; a=0.0; lt=0.0
    for i in range(len(t)):
        d=t[i]-lt; x=x*np.exp(-d/tau)+s[i]*w[i]; a=a*np.exp(-d/tau2)+w[i]; lt=t[i]; S[i]=x; A[i]=a
    return S,A
for tau2 in [2e6,10e6,60e6]:
  for c in [100,300,1000]:
    for th in [0.2,0.3,0.4]:
        out=[]
        for r in ['round3_public','round2_public']:
            if r not in polE.FC: polE.FC[r]=polE.feats(r)
            F=polE.FC[r]; E,_,_=data(r)
            S,A=S2(F,polE.W(F,P),P['tau'],tau2)
            dt,dg=targets(F['t'],S/(A*P['tau']/tau2+c),th)
            out.append(run_targets(E,dt,dg,0,3e9)['pnl'])
        print(tau2,c,th,sum(out),out)
