exec(open('logs/gw/ml_probe.py').read().split("print('### 1)")[0])
D['round2_a']=None
import numpy as np
# (a) GBM trained on round2_public first half + round3_a, test round3_b
t2,_,_,_,_=D['round2_public']
for H in [0.25,0.5]:
    Xa,ya,_=ds('round3_a',H,4); X2,y2,m2=ds('round2_public',H,4)
    half=t2[m2]<3000
    Xtr=np.vstack([Xa,X2[half]]); ytr=np.concatenate([ya,y2[half]])
    g=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_leaf_nodes=31,min_samples_leaf=200,l2_regularization=1.0).fit(Xtr,ytr)
    g3=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_leaf_nodes=31,min_samples_leaf=200,l2_regularization=1.0).fit(Xa,ya)
    b,w=M[H]; Xb,yb,_=ds('round3_b',H,4)
    print(f'H={H} round3_b: ridge(r3a) {r2(yb,b+Xb@w):.3f} gbm(r3a) {r2(yb,g3.predict(Xb)):.3f} gbm(r3a+r2 1st half) {r2(yb,g.predict(Xb)):.3f} | r2 2nd half: ridge {r2(y2[~half],b+X2[~half]@w):.3f} gbm(r3a) {r2(y2[~half],g3.predict(X2[~half])):.3f} gbm(mix) {r2(y2[~half],g.predict(X2[~half])):.3f}')
# (b) price movement while spread>1 (model off)
for f in ['round3_b','round2_public']:
    t,bid,ask,X,on=D[f]; mid=(bid+ask)/2; dm=np.abs(np.diff(mid)); sp=(ask-bid)[:-1]
    print(f'{f}: time-share spread>1 {np.sum(np.diff(t)[sp>1])/t[-1]:.2f}, share of |mid moves| starting in spread>1 {dm[sp>1].sum()/dm.sum():.2f}')
