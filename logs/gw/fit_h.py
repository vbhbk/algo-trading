import numpy as np, importlib.util, json
spec=importlib.util.spec_from_file_location('v','strategies/strategy_v8_r3.py'); v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
def load(f):
    d=np.load(f'logs/gw/feat/{f}.npz'); ok=~np.isnan(d['bid'])
    t=d['t'][ok]/1e6; bid=d['bid'][ok]; ask=d['ask'][ok]; X=d['X'][ok].astype(np.float64); on=~np.isnan(X[:,0])
    return t,bid,ask,X,on
def ridge(X,y,lam):
    mu,sd=X.mean(0),X.std(0); keep=sd>1e-12; Z=(X[:,keep]-mu[keep])/sd[keep]; ym=y.mean()
    ws=np.linalg.solve(Z.T@Z+lam*len(y)*np.eye(Z.shape[1]),Z.T@(y-ym)); w=np.zeros(X.shape[1]); w[keep]=ws/sd[keep]
    return ym-float(w@mu),w
D={f:load(f) for f in ['round3_a','round3_b','round2_public']}
def target(f,H):
    t,bid,ask,X,on=D[f]; mid=(bid+ask)/2; j=np.searchsorted(t,t+H,side='right')-1; return np.clip(mid[j]-mid,-3*H**0.5*4,3*H**0.5*4)
res={}
for H in [0.25,0.5,1,2,5,10,30]:
    t,bid,ask,X,on=D['round3_a']; y=target('round3_a',H); m=on&(t<t[-1]-H)
    Xa,ya=X[m],y[m]; k=int(.7*len(ya)); best=None
    for lam in [1e-5,1e-4,1e-3,1e-2]:
        b,w=ridge(Xa[:k],ya[:k],lam); r=1-np.mean((ya[k:]-b-Xa[k:]@w)**2)/np.var(ya[k:])
        if best is None or r>best[0]: best=(r,lam)
    b,w=ridge(Xa,ya,best[1]); res[H]=(b,w.tolist())
    out=[]
    for f in ['round3_b','round2_public']:
        t2,b2,a2,X2,on2=D[f]; y2=target(f,H); m2=on2&(t2<t2[-1]-H); pr=b+X2[m2]@w
        out.append(f'{f} R2 {1-np.mean((y2[m2]-pr)**2)/np.var(y2[m2]):.3f} corr {np.corrcoef(pr,y2[m2])[0,1]:.3f}')
    print(f'H={H}: lam {best[1]} inner {best[0]:.3f} | '+' | '.join(out))
json.dump({str(k):v for k,v in res.items()},open('logs/gw/models_r3a.json','w'))
