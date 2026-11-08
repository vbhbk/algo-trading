import numpy as np, sys; sys.path.insert(0,"logs/cannon/tools"); from load import load
for r in ['round2_public','round3_public']:
    E,T,ok=load(r)
    t=E[:,0]; bid=E[:,1].astype(float); ask=E[:,2].astype(float)
    mid=(bid+ask)/2
    # aggregate trades per taker order
    tid=T[:,4]; 
    o=np.lexsort((T[:,0],tid))
    T=T[o]; tid=T[:,4]
    starts=np.r_[0,np.nonzero(np.diff(tid))[0]+1]
    tt=T[starts,0]; sg=T[starts,1]; q=np.add.reduceat(T[:,2],starts); ei=T[starts,5]
    lv=np.add.reduceat(np.r_[0,(np.diff(T[:,3])!=0)].astype(int)*(np.r_[0,np.diff(tid)]==0),starts)+1
    print(f'== {r} aggressive orders {len(tt)} median q {np.median(q)} mean {q.mean():.0f}')
    def at(tq):  # state just before/at time tq
        i=np.searchsorted(t,tq,side='right')-1; return i
    i0=np.searchsorted(t,tt,side='left')-1  # before the trade
    for lab,m in [('q<20',q<20),('20-60',(q>=20)&(q<60)),('60-150',(q>=60)&(q<150)),('150-400',(q>=150)&(q<400)),('400+',q>=400),('sweep',lv>=2)]:
        if m.sum()<50: continue
        s=sg[m]; out=[]
        m0=mid[np.maximum(i0[m],0)]
        for h in [0.005,0.05,0.25,1,2,5,10,30]:
            j=at(tt[m]+int(h*1e6)); out.append((s*(mid[j]-m0)).mean())
        # entry at t+5ms touch, exit at mid +2s / +10s
        j5=at(tt[m]+5000); ent=np.where(s>0,ask[j5],bid[j5])
        j2=at(tt[m]+2_000_000); j10=at(tt[m]+10_000_000)
        p2=(s*(mid[j2]-ent)).mean(); p10=(s*(mid[j10]-ent)).mean()
        print(f'  {lab:8s} n={m.sum():6d} drift ' + ' '.join(f'{x:+.2f}' for x in out) + f' | take@5ms->mid2s {p2:+.2f} ->10s {p10:+.2f}')
    print('   horizons: 5ms 50ms 250ms 1s 2s 5s 10s 30s')
