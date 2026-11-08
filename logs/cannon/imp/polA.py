import numpy as np, sys, time
sys.path.insert(0,'logs/cannon/imp'); from sim import data, run_targets
def decisions(tt, sg, H, cap=95):
    dt=[]; dg=[]
    for i in range(len(tt)):
        dt.append(tt[i]); dg.append(int(sg[i])*cap)
        e = tt[i]+H
        if i+1==len(tt) or tt[i+1] > e:
            dt.append(e); dg.append(0)
    return np.array(dt), np.array(dg)
if __name__=='__main__':
    for r in ['round3_public','round2_public']:
        E,T,K=data(r)
        for Q in [150,250,400,800]:
            for lv in [1]:
                for H in [0.5,1,2,4,8,1e5]:
                    m=(K[:,2]>=Q)&(K[:,3]>=lv)
                    dt,dg=decisions(K[m,0],K[m,1],int(H*1e6))
                    a=run_targets(E,dt,dg,0,3e9); b=run_targets(E,dt,dg,3e9,6e9)
                    print(r[:6],Q,lv,H,'A',a,'| B pnl',b['pnl'])
