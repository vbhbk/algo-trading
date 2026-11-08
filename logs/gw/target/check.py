import numpy as np, json, time, sys
sys.path.insert(0,'logs/gw/target')
from common import *
M={float(k):(v[0],np.array(v[1])) for k,v in json.load(open('logs/gw/models_r3a.json')).items()}
t0=time.time(); D=load('round3_b'); T=targets(D); print('load+targets',time.time()-t0)
for k,v in T.items(): print(k, np.nanmean(v), np.nanstd(v))
b,w=M[0.25]; p=np.nan_to_num(b+D['X'].astype(float)@w)
m=seg_mask(D,0,3001)
for thr in [0.6,0.8]: print(thr, sim(D,m,p,-p,thr))
