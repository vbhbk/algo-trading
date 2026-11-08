import numpy as np
def load(f):
    d=np.load(f'logs/gw/feat/{f}.npz'); ok=~np.isnan(d['bid'])
    t=d['t'][ok]/1e6; mid=(d['bid'][ok]+d['ask'][ok])/2; sp=(d['ask']-d['bid'])[ok]
    pl=d['pl'][ok]; p=d['p'][ok]; sv=d['sv'][ok]
    pl=np.where(np.isnan(pl),0,pl); p=np.where(np.isnan(p),0,p)
    return t,mid,sp,pl,p,sv
def ema(t,x,tau):
    out=np.empty_like(x); e=0.0; tp=t[0]
    for i in range(len(x)):
        e=x[i]+(e-x[i])*np.exp(-(t[i]-tp)/tau) if False else e*np.exp(-(t[i]-tp)/tau); e+=x[i]; tp=t[i]; out[i]=e
    return out
def run(t,mid,sp,sig,enter,exit_,cap=95,cost=0.5):
    # hysteresis: go +cap when sig>enter, -cap when sig<-enter, flat when |sig|<exit (exit<0 means hold)
    pos=0; pnl=0.0; tr=0
    tgt=np.zeros(len(t))
    cur=0
    for i in range(len(t)):
        s=sig[i]
        if s>enter: cur=cap
        elif s<-enter: cur=-cap
        elif exit_>=0 and abs(s)<exit_: cur=0
        tgt[i]=cur
    dpos=np.diff(np.concatenate([[0],tgt]))
    pnl=np.sum(tgt[:-1]*np.diff(mid)) - np.sum(np.abs(dpos)*sp*cost)
    return pnl, np.sum(np.abs(dpos))/cap/2
import sys
for f in ['round3_a','round3_b','round2_public']:
    t,mid,sp,pl,p,sv=load(f)
    print('==',f)
    for name,sig in [('pl',pl),('p',p),('pl+p',pl+p)]:
        for en in [0.2,0.4,0.6,1.0]:
            for ex in [-1,0.1]:
                r,fl=run(t,mid,sp,sig,en,ex); r0,_=run(t,mid,sp,sig,en,ex,cost=0)
                print(f'  {name:5s} enter {en} exit {ex}: pnl {r:9.0f} (nocost {r0:9.0f}) flips {fl:6.0f}')
