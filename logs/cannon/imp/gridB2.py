import numpy as np, sys, itertools
sys.path.insert(0,'logs/cannon/imp'); from polB import ev
res=[]
for q0,tau,th,spw,relbad in itertools.product([0,50,100,200,300],[0.2e6,0.5e6,1e6],[100,200,300,500],[1,2],[1,0.3]):
    a3=ev('round3_public','a',q0=q0,tau=tau,th=th,spw=spw,relbad=relbad); a2=ev('round2_public','a',q0=q0,tau=tau,th=th,spw=spw,relbad=relbad)
    res.append((a3['pnl']+a2['pnl'],a3['pnl'],a2['pnl'],q0,tau,th,spw,relbad,a3['trades'],a2['trades']))
res.sort(reverse=True)
for x in res[:25]: print(x)
