#%%
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Fold
import scipy.signal as sig
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict

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
    perm = np.zeros((n,ny)).astype(int)
    for i in range(n):
        cost = torch.zeros(ny,nx)
        for j in range(ny):
            cost[j] = (y[i,j].unsqueeze(0) - x[i,:]).abs().sum(-1)

        _, col_ind = linear_sum_assignment(cost.detach().numpy().T)
        
        out[i,:nx] = y[i,col_ind]
        perm[i,:nx] = col_ind
        if ny > nx:
            k = np.arange(ny)
            k_rem = np.array(list(set(k) - set(col_ind)))
            out[i,nx:] = y[i,k_rem]
            perm[i,nx:] = k_rem

    return out, perm


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

def ideal_masks(stft_sources,alpha=1):
    eps = np.finfo(np.float).eps
    # compute spectrogram of target source
    stft_mix = stft_sources.sum(1).unsqueeze(1)
    X = stft_mix.abs() ** alpha
    S = stft_sources.abs() ** alpha
    model = S.sum(1).unsqueeze(1) + eps
    Mask = S / model
    Y = X * Mask
    Y_binary = X * Mask.round()
    phase_part = torch.exp(1j*stft_mix.angle())
    Y = Y * phase_part
    Y_binary = Y_binary * phase_part
    return Y, Y_binary

def load_state_dict(path,device="cpu"):
    # original saved file with DataParallel
    state_dict = torch.load(path,map_location=torch.device(device))
    # create new OrderedDict that does not contain `module.`
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if 'module.' == name[:7]:
            name = name[7:] # remove `module.`
        new_state_dict[name] = v
    
    return new_state_dict

def make_scale_invariant(estimates,targets):
    k = targets.shape[1]
    estimates = estimates[:,:k]
    Rss = (targets * targets).sum(-1)
    alpha = (estimates * targets).sum(-1) / Rss
    return targets * alpha.unsqueeze(-1)

def compute_measures_batch(estimates, targets, salt=1e-12):

    n = targets.shape[0]
    k = targets.shape[1]
    
    SDR = torch.zeros(n,k,device=estimates.device)
    SIR = torch.zeros(n,k,device=estimates.device)
    SAR = torch.zeros(n,k,device=estimates.device)

    Rss = torch.zeros(n,k,k,device=estimates.device)
    I = torch.eye(k,device=estimates.device)*salt
    for ii in range(n):
        Rss[ii] = (targets[ii] @ targets[ii].T + I)
    
    for jj in range(k):
        SDR[:,jj], SIR[:,jj], SAR[:,jj] = compute_measures_batch_one(estimates=estimates[:,jj],targets=targets,Rss=Rss,j=jj,salt=salt)

    return SDR, SIR, SAR

def compute_measures_batch_one(estimates, targets, Rss, j, salt=1e-14):
    e_true = targets[:,j] 
    # Get the SIR
    e_res= estimates - e_true
    Rsr = (e_res.unsqueeze(1) * targets).sum(-1)
    b = torch.linalg.solve(Rss,Rsr)

    e_interf = (targets * b.unsqueeze(-1)).sum(1)
    e_artif = e_res - e_interf

    E_target = (e_true*e_true).sum(-1) + salt
    E_residual = (e_res*e_res).sum(-1) + salt
    E_interf = (e_interf*e_interf).sum(-1) + salt
    E_artif = (e_artif*e_artif).sum(-1) + salt

    SDR = 10*(E_target.log10() - E_residual.log10())
    SIR = 10*(E_target.log10() - E_interf.log10())
    SAR = 10*(E_target.log10() - E_artif.log10())
    return SDR, SIR, SAR

def receptive_field(kernel_size=3,stride=2):
    kernel_size = np.atleast_1d(kernel_size)
    stride = np.atleast_1d(stride)
    
    field = np.concatenate([np.atleast_1d(1),kernel_size],0)
    stride = np.concatenate([np.atleast_1d(1),stride],0)
    layers = len(field)
    r = np.zeros(layers).astype(int)
    
    r[-1] = 1
    for i in range(layers-1,0,-1):
        r[i-1] = stride[i]*r[i] + (field[i]-stride[i])
    return r
# %%
