exec(open('logs/gw/lib.py').read())
def scal(t,bid,ask,s,en,H,cap=95,lat=0.005,exitcost=1.0):
    # target: +-cap for H s after last trigger (refreshed by new same-dir triggers; opposite trigger flips), else 0
    trig=np.where(s>en,1,np.where(s<-en,-1,0)); n=len(t)
    tg=np.zeros(n,dtype=int); cur=0; until=-1
    ti=np.flatnonzero(trig)
    last=np.full(n,-1); last[ti]=ti; last=np.maximum.accumulate(last)
    has=last>=0; dirn=np.where(has,trig[np.maximum(last,0)],0)
    alive=has&(t-t[np.maximum(last,0)]<=H)
    tg=np.where(alive,dirn,0)*cap
    j=np.minimum(np.searchsorted(t,t+lat),n-1); ch=np.flatnonzero(np.diff(np.concatenate([[0],tg]))!=0)
    pos=0;cash=0.0;nt=0
    for i in ch:
        k=j[i]; d=tg[i]-pos
        entering=abs(tg[i])>abs(pos) or np.sign(tg[i])!=np.sign(pos)
        m=(bid[k]+ask[k])/2; half=(ask[k]-bid[k])/2
        if tg[i]==0: px=m+np.sign(d)*half*exitcost   # exit leg
        else: px=ask[k] if d>0 else bid[k]
        cash-=d*px; pos=tg[i]; nt+=1
    return cash+pos*(bid[-1]+ask[-1])/2, nt
for f in ['round3_a','round3_b','round2_public']:
    t,bid,ask,pl,p,X=load(f); s=pred(X,0.25)
    print('==',f)
    for en in [0.6,0.8,1.0]:
        row=[]
        for H in [0.5,1,2,3,5]:
            r1,_=scal(t,bid,ask,s,en,H); r0,n=scal(t,bid,ask,s,en,H,exitcost=-1.0)
            row.append(f'H{H}: {r1/1e3:4.0f}k|{r0/1e3:4.0f}k')
        print(f'  en {en}: '+'  '.join(row)+f'  (taker-exit | passive-exit at touch)')
