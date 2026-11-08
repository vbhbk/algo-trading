import numpy as np, sys
sys.path.insert(0,'logs/cannon/imp'); from sim import data
def feats(r):
    E,T,K=data(r); t=E[:,0]; bid=E[:,1]; ask=E[:,2]; mid=(bid+ask)/2
    tt=K[:,0]; s=K[:,1]; q=K[:,2]; lv=K[:,3]; ei=K[:,4].astype(int)
    pre=np.maximum(ei-1,0)
    tq=np.where(s>0,E[pre,4],E[pre,3])          # touch qty on the hit side before
    oq=np.where(s>0,E[pre,3],E[pre,4])          # opposite touch qty
    d5=np.where(s>0,E[pre,6],E[pre,5])
    sp0=ask[pre]-bid[pre]
    post=ei; sp1=ask[post]-bid[post]
    moved=s*(mid[post]-mid[pre])
    j5=np.searchsorted(t,tt+5000,'left')-1
    ent=np.where(s>0,ask[j5],bid[j5]); spe=ask[j5]-bid[j5]
    out={}
    for H in [0.5,1,2,5,10,30]:
        j=np.searchsorted(t,tt+H*1e6,'right')-1
        out[H]=s*(mid[j]-ent)
    # mid-to-mid drift from pre
    return dict(t=tt,s=s,q=q,lv=lv,tq=tq,oq=oq,d5=d5,sp0=sp0,sp1=sp1,moved=moved,spe=spe,ent=ent,mid0=mid[pre],m=out)
def tab(F, mask, lab):
    m=F['m']; n=mask.sum()
    print(f'{lab:28s} n={n:5d} '+' '.join(f'{m[H][mask].mean():+.2f}' for H in m))
if __name__=='__main__':
    for r in ['round3_public','round2_public']:
        F=feats(r); A=F['t']<3e9
        print('==',r,'first half; net/lot entering at touch+5ms, mark mid @ .5 1 2 5 10 30 s')
        q=F['q']; rel=q/np.maximum(F['tq'],1)
        for lo,hi in [(60,150),(150,400),(400,800),(800,1e9)]:
            b=A&(q>=lo)&(q<hi)
            tab(F,b,f'q {lo}-{hi}')
            tab(F,b&(rel>=1),'   cleared level')
            tab(F,b&(rel<1),'   not cleared')
            tab(F,b&(F['sp0']==1),'   spread0=1')
            tab(F,b&(F['sp0']>1),'   spread0>1')
            tab(F,b&(F['spe']==1),'   spread at entry=1')
            tab(F,b&(F['spe']>1),'   spread at entry>1')
        for lo,hi in [(0,.25),(.25,.5),(.5,1),(1,2),(2,1e9)]:
            tab(F,A&(q>=60)&(rel>=lo)&(rel<hi),f'q>=60 rel {lo}-{hi}')
