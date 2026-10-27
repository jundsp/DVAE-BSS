#%%
import os
os.sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from utils.utils import load_state_dict, optimal_permute
import numpy as np
from dataset import get_data_loaders
from torch.nn import functional as F
from torchaudio.transforms import GriffinLim

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

    kwds = dict(num_sources=args['num_sources'],
                dimx=257,
                dimz=args['latent_dim'],
                dimh=args['rnn_dim'],
                dimg=args['rnn_dim'],
                kernel_size=(3,3),
                stride=(2,1),
                padding=(1,1),
                layer_channels=tuple(chans),
                activate_final_layer=True,
                separator_layers=args['separator_layers'],
                separator_kernel_size=args['separator_kernel_size'],
                separator_channels=args['separator_channels']
                )

    model = DSVAE(**kwds)
    model.load_state_dict(load_state_dict(path=path / name))
    model = Separator(model,compression=args['compression'],mask=True)
    model.eval()
    return model

def get_dataset(path:Path,name='RWC',batch_size=2,duration=4):

    test_set = RWC(root=path / 'RWC_test')
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

K = 4
path = Path("output/svae/RWC/dilated/230208-13576125")
name =  "svae_RWC_S2_K4_L1_pooled_D16_C5-10.pt"
model = get_model(path,name)

T = 8
dataset = get_dataset(path=Path("data"),duration=T,batch_size=2,name='RWC')

# torch.manual_seed(7)
with torch.no_grad():
    batch = next(iter(dataset))
    y = batch['mix'][:,0]
    s = batch['stems']

    s_hat = model(y)
    s_hat, perm = optimal_permute(s_hat,s)

    input = model.preprocess(y).abs()
    output = model.model(input)
    

Y = model.preprocess(y).abs()
S = model.preprocess(s[0]).abs()
S_hat = model.preprocess(s_hat[0]).abs()

mu = output['mu_q'][perm.ravel()]
mu = mu.reshape(-1,4,mu.shape[-2],mu.shape[-1])[0]

plt.figure()
plot_spectrogram(Y[0],extent=[0,T,0,8],vmin=-60,vmax=7)
fig,ax = plt.subplots(K,3,figsize=(7,3/2*K),sharex=True)
for i in range(2):
    plot_spectrogram(S[i],extent=[0,T,0,8],vmin=-60,vmax=7,ax=ax[i,0])
for i in range(2,K):
    ax[i,0].remove()
for i in range(K):
    plot_spectrogram(S_hat[i],extent=[0,T,0,8],vmin=-60,vmax=7,ax=ax[i,1])
    ax[i,2].plot(np.linspace(0,T,mu.shape[1]),mu[i],'k',alpha=.1)
plt.tight_layout()



#%%
with torch.no_grad():
    s_samp, z_samp, h_samp = model.model.sample(N=2048,M=1)

#%%
k = 0
plot_spectrogram(s_samp[0,k])
plt.figure()
plt.imshow(z_samp[k].T,aspect='auto',origin='lower',interpolation='none')

gl = GriffinLim(n_fft=512,n_iter=32,win_length=512,hop_length=128,power=1)
samp_source = gl(s_samp[0,k] ** (1/.5))
samp_source /= samp_source.abs().max() * 2
plt.figure()
plt.plot(samp_source)

torchaudio.save('sampled.wav',samp_source.unsqueeze(0),16000)



# %%
