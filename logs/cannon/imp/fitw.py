import numpy as np, sys
sys.path.insert(0,'logs/cannon/imp'); from es import feats
from sim import data
for r in ['round3_public','round2_public']:
    F=feats(r); E,_,_=data(r); t=E[:,0]; mid=(E[:,1]+E[:,2])/2
    A=F['t']<3e9; q=F['q']; rel=q/np.maximum(F['tq'],1)
    sp=(F['sp0']>1).astype(float); cl=((rel>=1)&(rel<2)).astype(float); big=(rel>=2).astype(float)
    X=np.c_[np.ones_like(q), np.minimum(q,2000)/100, np.minimum(q,2000)/100*sp, np.minimum(q,2000)/100*cl, np.minimum(q,2000)/100*big, (F['lv']>=2), sp, rel.clip(0,3), F['oq']/1000]
    names=['1','q','q*sp>1','q*cl1-2','q*rel2+','lv2','sp>1','rel','oppq']
    for H in [0.25,1,2]:
        j=np.searchsorted(t,F['t']+H*1e6,'right')-1
        y=F['s']*(mid[j]-F['mid0'])
        b=np.linalg.lstsq(X[A],y[A],rcond=None)[0]
        print(r[:6],H,' '.join(f'{n}:{v:+.2f}' for n,v in zip(names,b)))
