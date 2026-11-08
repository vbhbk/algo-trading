import numpy as np, sys
sys.path.insert(0,'logs/cannon/imp'); from sim import data, run_targets
import polE
def add_stop(E, dt, dg, L, mode='flat'):
    t=E[:,0]; mid=(E[:,1]+E[:,2])/2
    odt=[];odg=[]
    n=len(dt); cur=0
    for i in range(n):
        odt.append(dt[i]); odg.append(dg[i])
        if dg[i]!=cur: cur=dg[i]; ent=None
        if dg[i]==0: continue
        # segment until next decision
        j0=np.searchsorted(t,dt[i],'left'); j1=np.searchsorted(t,dt[i+1],'left') if i+1<n else len(t)
        if j1<=j0: continue
        m=mid[j0:j1]*np.sign(dg[i])
        if ent is None: ent=m[0]; peak=m[0]
        pk=np.maximum.accumulate(np.r_[peak,m])[1:]
        hit=np.nonzero(m<=pk-L)[0]
        if len(hit):
            odt.append(t[j0+hit[0]]); odg.append(0 if mode=='flat' else -dg[i]); cur=odg[-1]
            # subsequent prints keep the score's target; we stop only until the score re-crosses (approx: next decision restores)
        peak=pk[-1]
    return np.array(odt),np.array(odg)
for L in [2,3,5,8]:
    out=[]
    for r in ['round3_public','round2_public']:
        if r not in polE.FC: polE.FC[r]=polE.feats(r)
        F=polE.FC[r]; E,_,_=data(r); P=dict(polE.P); P['r0']=1.25
        S=polE.S_of(F,polE.W(F,P),P['tau']); dt,dg=polE.targets(F['t'],S,P['th'])
        sdt,sdg=add_stop(E,dt,dg,L)
        out.append(run_targets(E,sdt,sdg,0,3e9)['pnl'])
    print(L,sum(out),out)
