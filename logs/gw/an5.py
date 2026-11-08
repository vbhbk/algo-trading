exec(open('logs/gw/lib.py').read())
for f in ['round3_a','round3_b','round2_public']:
    t,bid,ask,pl,p,X=load(f); s=pred(X,0.25); rv=X[:,16]  # rvol*0.1
    print('==',f,'rv pct',np.percentile(rv[rv>0],[10,50,90]).round(2))
    for k in [0.0,0.01,0.02,0.04]:
        row=[]
        for c in [13,12]:
            sig=s-k*X[:,c]; r,n=run2(t,bid,ask,95*hyst(sig,0.6)); row.append(f'ema{c}:{r/1e3:.0f}k/{n}')
        print(f'  MR k={k}: '+' '.join(row))
    med=np.median(rv[rv>0])
    for a in [0.25,0.5,1.0]:
        en=0.6*(rv/med)**a; en=np.clip(en,0.3,1.5); sig=s/en*0.6
        r,n=run2(t,bid,ask,95*hyst(sig,0.6)); print(f'  volscale a={a}: {r/1e3:.0f}k/{n}')
