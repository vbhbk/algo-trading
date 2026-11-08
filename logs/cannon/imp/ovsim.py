import numpy as np, sys
sys.path.insert(0,'logs/cannon/imp'); from sim import data, run_targets
import polE
def score_at_events(r, P):
    if r not in polE.FC: polE.FC[r]=polE.feats(r)
    F=polE.FC[r]; E,_,_=data(r); t=E[:,0]
    S=polE.S_of(F,polE.W(F,P),P['tau'])
    # value at each event: last print score decayed
    ii=np.searchsorted(F['t'],t,'right')-1
    sc=np.where(ii>=0,S[np.maximum(ii,0)]*np.exp(-(t-F['t'][np.maximum(ii,0)])/P['tau']),0.0)
    return sc
def quantize(t, tg, step=20):
    out_t=[];out_g=[];last=None
    for i in np.nonzero(np.r_[True,np.diff(tg)!=0])[0]:
        g=tg[i]
        if last is None or abs(g-last)>=step or (g!=0 and np.sign(g)!=np.sign(last)) or (abs(g)==95 and g!=last):
            out_t.append(t[i]); out_g.append(g); last=g
    return np.array(out_t),np.array(out_g)
