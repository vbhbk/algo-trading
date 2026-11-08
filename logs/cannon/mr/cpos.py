# cannon_ref inventory path on the 0.1 s grid from a full-logs run
import numpy as np, pandas as pd
def cpos(logdir,n=30000):
    t=pd.read_csv(f'{logdir}/trades.csv')
    u=t[(t.taker_owner=='user')|(t.maker_owner=='user')]
    # user buys: taker user & taker_side B, or maker user & taker_side A
    buy=((u.taker_owner=='user')&(u.taker_side=='B'))|((u.maker_owner=='user')&(u.taker_side=='A'))
    dq=np.where(buy,u.quantity,-u.quantity)
    k=np.minimum((u.timestamp_us.values//100000).astype(int),n-1)
    d=np.bincount(k,weights=dq,minlength=n)
    cash=-np.bincount(k,weights=dq*u.price.values,minlength=n)
    return np.cumsum(d), np.cumsum(cash)
