import os
os.sys.path.append(r"/Users/julian/Documents/GitHub/svae-bss/")

import torch
from utils.utils import load_state_dict, optimal_permute
import numpy as np
from dataset import get_data_loaders
from torch.nn import functional as F
from model import *
from dataset import *
from pathlib import Path
import json
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times",
    "axes.titlesize": "medium",
    "axes.labelsize": "medium",
})
def get_model(path:Path,name:str):
    fname = path / "configuration.txt"
    with open(fname) as f:
        data = f.read()
    args = json.loads(data)

    chans = list(args['cnn_channels'])
    chans = chans + [chans[-1]]
    kwds = dict(num_sources=args['num_sources'],
                dimx=257,
                dimz=args['latent_dim'],
                dimh=args['rnn_dim'],
                dimg=args['rnn_dim'],
                kernel_size=(3,3),
                stride=(2,2),
                padding=(1,1),
                layer_channels=tuple(chans),
                activate_final_layer=True
                )

    model = DCAE(**kwds)
    model.load_state_dict(load_state_dict(path=path / name))
    model = Separator(model,compression=args['compression'])
    model.eval()
    return model

def get_dataset(path:Path,batch_size=1):
    test_set = NSYNTH(root=path)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=True)
    return test_loader

def plot_spectrogram(y,ax=None,**kwds):
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.imshow(y.log10()*20,aspect='auto',origin='lower',interpolation='none',cmap='jet',**kwds);
    # ax.set_ylabel('frequency (kHz)')
    # ax.set_xlabel('time (s)')
    ax.set_xlim(0)
    ax.set_ylim(0)

K = 5
path = Path("/Users/julian/Documents/Code/Python/Results/SVAE/RWC/sources/K{}/".format(K))
name =  "svae_rwc_K{}_c5.pt".format(K)
model = get_model(path,name)
model.eval()


dataset = get_dataset(path=Path("/Volumes/JulianSSD/data/NSYNTH_test"))

torch.manual_seed(7)
with torch.no_grad():
    batch = next(iter(dataset))
    y = batch['mix'][:,0]
    s = batch['stems']

    s_hat = model(y)
    s_hat, perm = optimal_permute(s_hat,s)

Y = model.preprocess(y).abs()
S = model.preprocess(s.view(-1,s.shape[-1])).abs()
S_hat = model.preprocess(s_hat.view(-1,s_hat.shape[-1])).abs()

plt.figure()
T = y.shape[-1]/16e3
plot_spectrogram(Y[0],extent=[0,T,0,8],vmin=-60,vmax=7)
fig,ax = plt.subplots(3,K,sharey=True,sharex=True,figsize=(6,3))
ax = ax.T

plot_spectrogram(Y[0],extent=[0,T,0,8],vmin=-60,vmax=7,ax=ax[0,0])
for i in range(1,K):
    ax[i,0].remove()
for i in range(2):
    plot_spectrogram(S[i],extent=[0,T,0,8],vmin=-60,vmax=7,ax=ax[i,1])
for i in range(2,K):
    ax[i,1].remove()
for i in range(K):
    plot_spectrogram(S_hat[i],extent=[0,T,0,8],vmin=-60,vmax=7,ax=ax[i,2])
    ax[i,2].set_xlabel(r'$t$ (s)')
for i in range(3):
    ax[0,i].set_ylabel(r'$f$ (kHz)')
plt.tight_layout()


fig,ax = plt.subplots(1,K,figsize=(K*1.2,1.5),sharey=True)
for i in range(K):
    ax[i].plot(model.model.smu[0,perm[0,i]].detach(),'k',linewidth=1)
ax[0].set_ylabel(r"$m$")