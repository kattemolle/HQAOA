"""
Compute the probability that the fluctuator is excited m times.
"""

import numpy as np
from matplotlib import pyplot as plt
from itertools import product

def trans(p,ep):
    return np.array([[ep+(1-ep)*(1-p), (1-ep)*(1-p)],
                     [(1-ep)*p,         ep+(1-ep)*p]])

def get_bin(x,mloc):
    return format(x, 'b').zfill(mloc)

def pstring(p,ep,mloc,i):
    T=trans(p,ep)
    probi=1
    if i[0]==0:
        probi=probi*(1-p)
    elif i[1]==1:
        probi=probi*p
    for l in range(mloc-1):
        probi=probi*T[i[l+1],i[l]]

    return probi

def pmerror(p,ep,mloc,merror):
    pmerr=0
    for i in range(2**mloc):
        bin=get_bin(i,mloc)
        bin=list(bin)
        bin=[int(i) for i in bin]
        if sum(bin)==merror:
            pmerr+=pstring(p,ep,mloc,bin)

    return pmerr

def makeplot(p,mloc):
    fig,ax=plt.subplots()
    merrorlist=range(0,mloc+1)
    eplist=np.linspace(0,1,5)
    for ep in eplist:
        pmerrorlist=[pmerror(p,ep,mloc,merror) for merror in merrorlist]
        ax.plot(merrorlist,pmerrorlist,'-o')

    ax.set_xlabel('Number of errors at p={}'.format(p))
    ax.set_ylabel('Probability')
    ax.legend(eplist,title='ep')
    fig.savefig('plots/pmerror.pdf')

if __name__=='__main__':
    makeplot(0.3,16)
    print('done')
