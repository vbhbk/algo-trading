import numpy as np
def load(r):
    d=np.load(f'logs/cannon/rec_{r}.npz'); E=d['E'].copy(); T=d['T']
    ok=(E[:,1]>0)&(E[:,2]>E[:,1])
    # forward fill invalid rows
    idx=np.where(ok,np.arange(len(E)),0); np.maximum.accumulate(idx,out=idx)
    E[:,1:]=E[idx,1:]
    return E,T,ok
def takers(T):
    o=np.lexsort((T[:,0],T[:,4])); T=T[o]; tid=T[:,4]
    st=np.r_[0,np.nonzero(np.diff(tid))[0]+1]
    pc=np.r_[0,(np.diff(T[:,3])!=0)&(np.diff(tid)==0)].astype(int)
    lv=np.add.reduceat(pc,st)+1
    q=np.add.reduceat(T[:,2],st)
    r=np.c_[T[st,0],T[st,1],q,lv,T[st,5],T[st,3]]
    return r[np.argsort(r[:,0],kind='stable')]
