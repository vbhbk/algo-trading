import numpy as np
for f in ['round3_a','round3_b','round2_public']:
    d=np.load(f'logs/gw/feat/{f}.npz'); t=d['t']/1e6; bid=d['bid']; ask=d['ask']
    ok=~np.isnan(bid); t=t[ok]; mid=(bid[ok]+ask[ok])/2; sp=(ask-bid)[ok]
    pl=d['pl'][ok]; p=d['p'][ok]; sv=d['sv'][ok]
    T=t[-1]; grid=np.arange(0,T,0.1); idx=np.searchsorted(t,grid,side='right')-1; m=mid[np.maximum(idx,0)]
    print(f'== {f}: T={T:.0f}s events {len(t)} spread1 frac {np.mean(sp==1):.2f} range {m.max()-m.min():.0f}')
    # return autocorr at horizons
    for h in [1,5,30,120]:
        k=int(h/0.1); r=m[k:]-m[:-k]; r=r[::k]
        ac=np.corrcoef(r[1:],r[:-1])[0,1]; print(f'  h={h}s ret sd {r.std():.2f} lag1 autocorr {ac:+.3f}')
    # predictive power of pl and p on future mid at horizons (on spread1 events, sampled)
    on=~np.isnan(pl); ti=t[on]; mi=mid[on]; pli=pl[on]; pi=p[on]
    s=np.arange(0,len(ti),20)
    for H in [0.25,2,5,10,30,60]:
        j=np.searchsorted(t,ti[s]+H,side='right')-1; y=mid[j]-mi[s]
        print(f'  H={H}s  corr(pl,y)={np.corrcoef(pli[s],y)[0,1]:.3f} corr(p,y)={np.corrcoef(pi[s],y)[0,1]:.3f}  E[y*sign(pl)]={np.mean(y*np.sign(pli[s])):.3f} sd y {y.std():.2f}')
    # signed flow -> future (1s bucket)
    nb=int(T); fb=np.bincount(np.minimum(t.astype(int),nb-1),weights=sv,minlength=nb)
    mb=m[::10][:nb]; r1=np.diff(mb)
    for L in [0,1,5,30]:
        if L==0: c=np.corrcoef(fb[:-1],r1)[0,1]
        else:
            fs=np.convolve(fb,np.ones(L),'full')[:nb-1]; fut=mb[np.minimum(np.arange(nb-1)+L+1,len(mb)-1)]-mb[np.arange(nb-1)+1]
            c=np.corrcoef(fs,fut)[0,1]
        print(f'  flow(last {L}s) vs fut ret ({L}s): {c:+.3f}')
