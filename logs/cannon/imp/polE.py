import numpy as np, sys
sys.path.insert(0,'logs/cannon/imp'); from sim import data, run_targets
from es import feats
from polB import targets
from polC import S_of
FC={}
def W(F, p):
    q=np.minimum(F['q'],p['qcap']); rel=F['q']/np.maximum(F['tq'],1)
    g=np.select([rel<0.25,rel<1,rel<2],[p['r0'],1.0,p['r2']],p['r3'])
    w=q*g*np.where(F['sp0']>1,p['spw'],1.0)
    w=w*np.where(F['d5']<p['dthr'],p['thin'],1.0)
    return w
def run(p, rounds=('round3_public','round2_public'), half='a'):
    out=[]
    for r in rounds:
        if r not in FC: FC[r]=feats(r)
        F=FC[r]; E,_,_=data(r)
        S=S_of(F,W(F,p),p['tau']); dt,dg=targets(F['t'],S,p['th'])
        t0,t1=(0,3e9) if half=='a' else (3e9,6e9)
        out.append(run_targets(E,dt,dg,t0,t1)['pnl'])
    return sum(out),out
P=dict(qcap=1000,r0=1.0,r2=0.3,r3=1.0,spw=2.0,dthr=0,thin=1.0,tau=5e5,th=300)
if __name__=='__main__':
    grid=dict(r0=[0.5,0.75,1.25,1.5],r3=[0.6,0.8,1.2],dthr=[1500,2500,4000],thin=[0.6,1.5,2.0],tau=[4e5,6e5],th=[250,350],spw=[1.5,2.5])
    best=run(P); print('start',best)
    for it in range(2):
        for k,vals in grid.items():
            for v in vals:
                q=dict(P); q[k]=v
                if k=='thin' and P['dthr']==0: q['dthr']=2500
                res=run(q)
                if res[0]>best[0]*1.005:
                    best=res; P=q; print('  ',k,v,res)
    print(P,best)
