import numpy as np, sys
sys.path.insert(0,'logs/cannon/imp'); from sim import data, run_targets
import polE
P=dict(polE.P); P['r0']=1.25
def tgp(t,S,th,lo,step):
    # proportional between lo*th and th, hold sign otherwise (hysteresis on magnitude: only increase |target| or flip)
    cur=0; dt=[];dg=[]
    for i,x in enumerate(S):
        g=int(np.clip(round(95*(abs(x)-lo*th)/((1-lo)*th)),0,95))*(1 if x>0 else -1)
        if np.sign(g)==np.sign(cur) and abs(g)<abs(cur): g=cur
        if g==0: g=cur
        if abs(g-cur)>=step or (abs(g)==95 and g!=cur): dt.append(t[i]); dg.append(g); cur=g
    return np.array(dt),np.array(dg)
for th in [300,400,600]:
  for lo in [0.3,0.6]:
    out=[]
    for r in ['round3_public','round2_public']:
        if r not in polE.FC: polE.FC[r]=polE.feats(r)
        F=polE.FC[r]; E,_,_=data(r)
        S=polE.S_of(F,polE.W(F,P),P['tau']); dt,dg=tgp(F['t'],S,th,lo,20)
        out.append(run_targets(E,dt,dg,0,3e9)['pnl'])
    print(th,lo,sum(out),out)
