import numpy as np, sys, itertools
sys.path.insert(0,'logs/cannon/imp'); from sim import data, run_targets
from es import feats
from polB import targets
from polC import S_of
FC={}
def W(F, alpha=1.0, spw=2.0, relbad=0.3, lvw=0.0, bigw=1.0, qcap=1e9, q0=0):
    q=np.minimum(F['q'],qcap); rel=F['q']/np.maximum(F['tq'],1)
    w=np.maximum(q-q0,0)**alpha
    w=w*np.where(F['sp0']>1,spw,1.0)*np.where((rel>=1)&(rel<2),relbad,1.0)*np.where(rel>=2,bigw,1.0)
    w=w*np.where(F['lv']>=2,1+lvw,1.0)
    return w
def ev(r, half, tau, th, th0=None, cap=95, **kw):
    if r not in FC: FC[r]=feats(r)
    F=FC[r]; E,_,_=data(r)
    S=S_of(F,W(F,**kw),tau); dt,dg=targets(F['t'],S,th,th0,cap)
    t0,t1=(0,3e9) if half=='a' else (3e9,6e9)
    return run_targets(E,dt,dg,t0,t1)
def both(**kw):
    a3=ev('round3_public','a',**kw); a2=ev('round2_public','a',**kw)
    return a3['pnl']+a2['pnl'],a3['pnl'],a2['pnl'],a3['trades']
if __name__=='__main__':
    base=dict(tau=5e5,th=300,spw=2,relbad=0.3)
    print('base',both(**base))
    for k,vals in [('alpha',[0.5,0.75,1.25]),('lvw',[-0.5,0.5,1]),('bigw',[0.5,1.5,2]),('qcap',[300,600,1000]),('spw',[1.5,3,4]),('relbad',[0,0.15,0.5])]:
        for v in vals:
            kw=dict(base); kw[k]=v
            if k=='alpha': kw['th']=300**v if v!=1 else 300
            print(k,v,both(**kw))
