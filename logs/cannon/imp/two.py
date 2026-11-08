import numpy as np, sys
sys.path.insert(0,'logs/cannon/imp'); from sim import data, run_targets
import polE
P=dict(polE.P); P['r0']=1.25
def tg2(t,S,th1,th2,mid=95):
    cur=0; dt=[];dg=[]
    for i,x in enumerate(S):
        g=cur
        if x>th2: g=95
        elif x<-th2: g=-95
        elif x>th1 and cur<0: g=min(0,cur+mid) if mid<95 else 0
        elif x<-th1 and cur>0: g=0
        if g!=cur: dt.append(t[i]); dg.append(g); cur=g
    return np.array(dt),np.array(dg)
for th1,th2 in [(300,300),(200,300),(200,400),(250,400),(300,450),(300,600)]:
    out=[]
    for r in ['round3_public','round2_public']:
        if r not in polE.FC: polE.FC[r]=polE.feats(r)
        F=polE.FC[r]; E,_,_=data(r)
        S=polE.S_of(F,polE.W(F,P),P['tau']); dt,dg=tg2(F['t'],S,th1,th2)
        out.append(run_targets(E,dt,dg,0,3e9)['pnl'])
    print(th1,th2,sum(out),out)
