import numpy as np, sys
sys.path.insert(0,'logs/cannon/imp'); from sim import data, run_targets
from ovsim import score_at_events, quantize
import polE
P=dict(polE.P); P['r0']=1.25
for r in ['round3_public','round2_public']:
    E,_,_=data(r); R=np.load(f'logs/cannon/imp/pl_{r}.npy'); t=E[:,0]
    assert len(R)==len(E)
    on=R[:,1]>0; pl=R[:,3]
    # hold last value when off
    idx=np.where(on,np.arange(len(pl)),0); np.maximum.accumulate(idx,out=idx); plh=pl[idx]
    sc=score_at_events(r,P)
    m=t<3e9
    for K in [0,0.1,0.2,0.3,0.5]:
        for TK in [200,100]:
            tg=np.clip(np.round(TK*plh+K*sc),-95,95).astype(int)
            dt,dg=quantize(t[m],tg[m],step=30)
            x=run_targets(E,dt,dg,0,3e9)
            print(r[:6],'K',K,'TK',TK,x['pnl'],x['cost'],x['trades'],x['avgabs'])
