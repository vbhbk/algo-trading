import numpy as np, sys
for r in ['round3_public','round4_public']:
    d=np.load(f'logs/cannon/rec_{r}.npz'); E=d['E']; T=d['T']
    t=E[:,0]/1e6; bid=E[:,1]; ask=E[:,2]; ok=(bid>0)&(ask>bid)
    t=t[ok]; mid=(bid[ok]+ask[ok])/2; sp=(ask-bid)[ok]
    # time weights
    dt=np.diff(t,append=t[-1])
    print(f'== {r} dur {t[-1]:.0f}s events {len(t)} trades {len(T)}')
    for s in [1,2,3]:
        m=sp==s if s<3 else sp>=3
        print(f'  spread {s}{"+" if s==3 else ""}: time {dt[m].sum()/dt.sum():.3f}')
    # 1s grid
    g=np.arange(0,t[-1],0.1); idx=np.searchsorted(t,g,side='right')-1; m=mid[idx]
    for h in [1,5,10,30,60,120,300]:
        k=int(h/0.1); r1=m[k:]-m[:-k]
        # autocorr of non-overlapping returns
        x=m[::k]; rr=np.diff(x); ac=np.corrcoef(rr[:-1],rr[1:])[0,1]
        print(f'  h={h:4d}s std {r1.std():7.2f} ac1 {ac:+.3f}  VR {np.var(rr)/(np.var(np.diff(m))*k):.2f}')
