import numpy as np, sys
sys.path.insert(0,'logs/cannon/imp'); from es import feats
from sim import data
def X_of(F):
    q=F['q']; rel=q/np.maximum(F['tq'],1); q1=np.minimum(q,2000)/100
    sp=(F['sp0']>1).astype(float); cl=((rel>=1)&(rel<2)).astype(float); big=(rel>=2).astype(float)
    return np.c_[np.ones_like(q), q1, q1*sp, q1*cl, q1*big, (F['lv']>=2), sp, rel.clip(0,3)]
names=['1','q','q*sp>1','q*cl1-2','q*rel2+','lv2','sp>1','rel']
def fit(H, rounds=('round3_public','round2_public')):
    Xs=[];ys=[]
    for r in rounds:
        F=feats(r); E,_,_=data(r); t=E[:,0]; mid=(E[:,1]+E[:,2])/2
        A=F['t']<3e9
        j0=np.searchsorted(t,F['t']+5000,'left')-1
        j=np.searchsorted(t,F['t']+H*1e6,'right')-1
        y=F['s']*(mid[j]-mid[j0])
        Xs.append(X_of(F)[A]); ys.append(y[A])
    X=np.vstack(Xs); y=np.concatenate(ys)
    b=np.linalg.lstsq(X,y,rcond=None)[0]
    return b
if __name__=='__main__':
    for H in [0.5,1,2,5]:
        b=fit(H); print(H,' '.join(f'{n}:{v:+.3f}' for n,v in zip(names,b)))
