#%%
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Fold
import scipy.signal as sig


def db2mag(x):
	return torch.pow(torch.ones_like(x)*10.0,x/20.0)

def mag2db(x):
    return 20.0*torch.log10(torch.abs(x))

def torchToNumpy(x):
    return x.detach().cpu().numpy()
