import numpy as np, json, time
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
M={float(k):(v[0],np.array(v[1])) for k,v in json.load(open('logs/gw/models_r3a.json')).items()}
def load(f):
    d=np.load(f'logs/gw/feat/{f}.npz'); ok=~np.isnan(d['bid'])
    t=d['t'][ok]/1e6; bid=d['bid'][ok]; ask=d['ask'][ok]; X=d['X'][ok].astype(np.float64); on=~np.isnan(X[:,0])
    return t,bid,ask,X,on
D={f:load(f) for f in ['round3_a','round3_b','round2_public']}
def ds(f,H,step):
    t,bid,ask,X,on=D[f]; mid=(bid+ask)/2; j=np.searchsorted(t,t+H,side='right')-1
    y=np.clip(mid[j]-mid,-12,12); m=np.flatnonzero(on&(t<t[-1]-H))[::step]
    return X[m],y[m],m
r2=lambda y,p:1-np.mean((y-p)**2)/np.var(y)
print('### 1) nonlinear vs ridge, out of sample R2')
for H in [0.25,0.5,1.0]:
    Xa,ya,_=ds('round3_a',H,4); b,w=M[H]
    g=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_leaf_nodes=31,min_samples_leaf=200,l2_regularization=1.0)
    t0=time.time(); g.fit(Xa,ya); ft=time.time()-t0
    row=[]
    for f in ['round3_b','round2_public']:
        Xb,yb,_=ds(f,H,4); pr=b+Xb@w; pg=g.predict(Xb)
        row.append(f'{f}: ridge {r2(yb,pr):.3f} gbm {r2(yb,pg):.3f} avg {r2(yb,(pr+pg)/2):.3f}')
    print(f'H={H}: '+' | '.join(row)+f' (fit {ft:.0f}s)')
print('### 2) meta-label: given a take signal (|p25|>=0.6), will 2s markout beat the spread cost?')
def meta(f):
    t,bid,ask,X,on=D[f]; mid=(bid+ask)/2; b,w=M[0.25]; p=np.where(on,b+np.nan_to_num(X)@w,0)
    idx=np.flatnonzero(on&(np.abs(p)>=0.6)&(t<t[-1]-2))
    # thin to one event per 100ms
    keep=[];last=-1
    for i in idx:
        if t[i]-last>=0.1: keep.append(i); last=t[i]
    idx=np.array(keep); s=np.sign(p[idx]); j=np.searchsorted(t,t[idx]+2,side='right')-1; k=np.searchsorted(t,t[idx]+0.005)
    entry=np.where(s>0,ask[k],bid[k]); gain=s*(mid[j]-entry)
    Z=np.column_stack([X[idx]*s[:,None],np.abs(p[idx])])
    return Z,gain
Za,ga=meta('round3_a'); c=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05,min_samples_leaf=100).fit(Za,ga>0)
for f in ['round3_b','round2_public']:
    Zb,gb=meta(f); pr=c.predict_proba(Zb)[:,1]; q=np.quantile(pr,[0.2,0.4])
    print(f'{f}: n {len(gb)} mean gain {gb.mean():+.2f} AUC {roc_auc_score(gb>0,pr):.3f} | drop worst 20%: mean {gb[pr>q[0]].mean():+.2f}, total {gb[pr>q[0]].sum()/gb.sum():.2f}x | drop 40%: {gb[pr>q[1]].mean():+.2f}, {gb[pr>q[1]].sum()/gb.sum():.2f}x | AUC of |p| alone {roc_auc_score(gb>0,Zb[:,-1]):.3f}')
