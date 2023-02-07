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

    kwds = dict(num_sources=args['num_sources'],
                dimx=257,
                dimz=args['latent_dim'],
                dimh=args['rnn_dim'],
                dimg=args['rnn_dim'],
                kernel_size=(3,3),
                stride=(2,2),
                padding=(1,1),
                layer_channels=tuple(chans),
                activate_final_layer=True,
                separator_layers=args['separator_layers'],
                separator_kernel_size=args['separator_kernel_size'],
                )

    model = DSVAE(**kwds)
    model.load_state_dict(load_state_dict(path=path / name))
    model = Separator(model,compression=args['compression'])
    model.eval()
    return model

def get_dataset(path:Path,name='RWC',batch_size=2,duration=4):
    data_root = path / "RWC_dataset"
    path_csv = 'test.csv'

    if name == 'composite':
        data_root = path
        path_csv = 'composite/test.csv'
    if name == 'MUMS':
        data_root = path / "MUMS"
        path_csv = None
    if name == 'nsynth':
        data_root = path / "nsynth-test/audio"
        path_csv = None
    if name == 'MUSDB':
        data_root = path / "MUSDB"
        path_csv = 'test.csv'

    test_set = Audioset(root=data_root,path_csv=path_csv,augment_levels=False,duration=duration,silence=0,starting=0,max_files=4)
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
path = Path("/Users/julian/Documents/Code/Python/Results/SVAE/RWC/dilated")
name =  "svae_RWC_S2_K4_L1_pooled_D24_C5-10.pt"
model = get_model(path,name)

T = 8
dataset = get_dataset(path=Path("/Volumes/JulianSSD/data"),duration=T,batch_size=4,name='RWC')
mixer = Mixer(n_sources=2)

# torch.manual_seed(7)
with torch.no_grad():
    batch,_ = next(iter(dataset))
    batch += torch.randn_like(batch)*0
    y,s = mixer(batch)

    input = model.preprocess(y).abs()

    output = model.model(input)
    # s_hat = optimal_permute(s_hat,s)

# Y = model.preprocess(y).abs()
# S = model.preprocess(s.view(-1,s.shape[-1])).abs()
# S_hat = model.preprocess(s_hat.view(-1,s_hat.shape[-1])).abs()

# plt.figure()
# plot_spectrogram(Y[0],extent=[0,T,0,8],vmin=-60,vmax=7)
# fig,ax = plt.subplots(K,2,sharey=True,sharex=True,figsize=(5,3))
# for i in range(2):
#     plot_spectrogram(S[i],extent=[0,T,0,8],vmin=-60,vmax=7,ax=ax[i,0])
# for i in range(2,K):
#     ax[i,0].remove()
# for i in range(K):
#     plot_spectrogram(S_hat[i],extent=[0,T,0,8],vmin=-60,vmax=7,ax=ax[i,1])
# plt.tight_layout()

plot_spectrogram(input[0].log(),extent=[0,T,0,8],vmin=-60,vmax=7)