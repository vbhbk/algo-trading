import numpy as np, sys; sys.path.insert(0,'logs/cannon/tools'); from load import load
for r in ['round3_public','round4_public']:
    E,T,ok=load(r)
    t=E[:,0]; bid=E[:,1]; ask=E[:,2]; bq=E[:,3]; aq=E[:,4]
    sp=ask-bid
    m1=sp==1
    print(r,'touch size when spread1: median bq',np.median(bq[m1]),'p10',np.percentile(bq[m1],10),'p25',np.percentile(bq[m1],25))
    # how often mid changes; average time between mid changes
    mid=bid+ask
    ch=np.nonzero(np.diff(mid))[0]
    print('  mid changes',len(ch),'per s',len(ch)/(t[-1]/1e6), 'median abs jump (ticks)',np.median(np.abs(np.diff(mid)[ch]))/2)
