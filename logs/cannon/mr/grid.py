# build 0.1 s grids (bid, ask, signed aggressive volume) from recorded npz
import sys; sys.path.insert(0,'logs/cannon/tools')
from load import load
import numpy as np
for r in ['round2_public','round3_public']:
    E,T,ok=load(r)
    g=np.arange(0,6000*10)*100000
    i=np.searchsorted(E[:,0],g,side='right')-1; i=np.clip(i,0,None)
    bid=E[i,1]; ask=E[i,2]
    k=np.minimum((T[:,0]//100000).astype(int),len(g)-1)
    flow=np.bincount(k,weights=T[:,1]*T[:,2],minlength=len(g))
    vol=np.bincount(k,weights=T[:,2],minlength=len(g))
    np.savez(f'logs/cannon/mr/grid_{r}.npz',bid=bid,ask=ask,flow=flow,vol=vol)
    print(r, bid[:3],ask[:3])
