#%%
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Fold
import scipy.signal as sig
from scipy.optimize import linear_sum_assignment

def db2mag(x):
	return torch.pow(torch.ones_like(x)*10.0,x/20.0)

def mag2db(x):
    return 20.0*torch.log10(torch.abs(x))

def torchToNumpy(x):
    return x.detach().cpu().numpy()
    

def optimal_permute(y,x):
    n = x.size(0)
    nx = x.size(1)
    ny = y.size(1)
    out = torch.zeros_like(y)
    for i in range(n):
        cost = torch.zeros(ny,nx)
        for j in range(ny):
            cost[j] = (y[i,j].unsqueeze(0) - x[i,:]).abs().sum(-1)

        _, col_ind = linear_sum_assignment(cost.detach().numpy().T)
        
        out[i,:nx] = y[i,col_ind]
        if ny > nx:
            k = np.arange(ny)
            k_rem = np.array(list(set(k) - set(col_ind)))
            out[i,nx:] = y[i,k_rem]

    return out


def panning(nsrc):
    pspace = np.linspace(-1,1,nsrc)
    no = nsrc//2
    ne = nsrc - no
    e = pspace[:ne]
    o = pspace[ne:][::-1]

    pans = np.zeros(nsrc)
    pans[::2] = e
    pans[1::2] = o
    pans += 1.0
    pans /= 2.0
    return pans



# %%
