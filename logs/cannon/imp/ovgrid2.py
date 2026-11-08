import numpy as np, sys
sys.path.insert(0,'logs/cannon/imp'); from sim import data, run_targets
from ovsim import score_at_events
import polE
P=dict(polE.P); P['r0']=1.25
def hyst(t, x, th):
    # decisions only where state flips
    st=np.zeros(len(x),int); cur=0
    up=x>th; dn=x<-th
    ev=np.nonzero(up|dn)[0]
    dt=[];dg=[]
    for i in ev:
        g=95 if up[i] else -95
        if g!=cur: dt.append(t[i]); dg.append(g); cur=g
    return np.array(dt),np.array(dg)
for r in ['round3_public','round2_public']:
    E,_,_=data(r); R=np.load(f'logs/cannon/imp/pl_{r}.npy'); t=E[:,0]
    on=R[:,1]>0; pl=R[:,3]
    idx=np.where(on,np.arange(len(pl)),0); np.maximum.accumulate(idx,out=idx); plh=pl[idx]
    sc=score_at_events(r,P)
    m=t<3e9
    print(r, 'corr pl vs score', np.corrcoef(plh[m],sc[m])[0,1])
    for K in [0,300,600,1000,1e9]:
        for th in [0.5,1.0,1.5,2.0]:
            x = plh + sc/K if K<1e9 else sc/300.
            if K==0: x=plh
            dt,dg=hyst(t[m],x[m],th if K<1e9 else th)
            res=run_targets(E,dt,dg,0,3e9)
            print(r[:6],'K',K,'th',th,res['pnl'],res['cost'],res['trades'])
