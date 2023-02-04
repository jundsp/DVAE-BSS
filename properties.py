#%%
import torch
from utils.utils import load_state_dict
import numpy as np
from lossfun import FreqSignalLoss
from dataset import get_data_loaders
from torch.nn import functional as F
from model import *
from dataset import AudioPipeline, AuditoryStreams
import torchaudio
from pathlib import Path
from torchaudio.transforms import GriffinLim

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times",
    "axes.titlesize": "medium",
    "axes.labelsize": "medium",
})

sr = 16000
T = 8
L = int(T*sr)
audio = torch.zeros(1,L)
audio[:,sr::sr//2] = 1
dataset = AuditoryStreams(root="/Users/julian/Documents/Code/data/dataset_harmonic/")
data = dataset.__getitem__(30*42+3)
audio,sources_true = data['mix'], data['stems']
audio = audio[:,sr//2:]
T = audio.shape[-1]/sr

audio += torch.randn(audio.shape)*1e-5
pipeline = AudioPipeline(n_fft=512,n_time=257+128,hop_length=128,compression=.5)
x = pipeline(audio)

model = DCAE(num_sources=5,dimx=257,dimz=256,dimh=512,dimg=512,kernel_size=(3,3),stride=(2,2),padding=(1,1),layer_channels=(32,64,128,128),activate_final_layer=True)

output_img_path = Path("/Users/julian/Documents/Code/Python/Results/SVAE/RWC/compression/K5/230122-13058078/")
path = output_img_path / "svae_RWC_S2_K5_L1_pooled_D256_C5-10.pt"
model.load_state_dict(load_state_dict(path=path))


model.eval()

with torch.no_grad():
    features_encoded = model.feature_encoder(x)
    output_stoch = model.forward(x,stochastic=True)
    x_hat = output_stoch['recon_x'].log()
    s_hat = output_stoch['recon_s'].log()
    z_hat = output_stoch['z'].abs()

    output_deter = model.forward(x,stochastic=False)
    s_deter = output_deter['recon_s'].log()
    z_deter = output_deter['z'].abs()

    samp_s,samp_z,samp_h = model.sample(N=z_hat.shape[1],M=2)

#%%
kwds = dict(aspect='auto',origin='lower',interpolation='none',cmap='Greys')
kwds_subplots = dict(nrows=1,ncols=5,sharey=True,figsize=(6,2))
plt.figure(figsize=(2.5,2))
plt.imshow(x[0].log(),**kwds,vmin=-8,extent=[0,T,0,sr//2/1e3]);
plt.xlabel('time (s)')
plt.ylabel('frequency (kHz)')
plt.tight_layout()
plt.savefig(output_img_path/'stft_mix.pdf')

plt.figure(figsize=(2.5,2))
plt.imshow(x_hat[0],**kwds,vmin=-8,extent=[0,T,0,sr//2/1e3]);
plt.xlabel('time (s)')
plt.ylabel('frequency (kHz)')
plt.tight_layout()
plt.savefig(output_img_path/'stft_mix_estimate.pdf')

# %%
fig,ax = plt.subplots(**kwds_subplots)
for i in range(5):
    ax[i].imshow(z_hat[i,1:].T,**kwds,vmin=0,vmax=z_hat.max(),extent=[0,T,0,z_hat.shape[-1]])
    ax[i].set_xlabel('time (s)')
plt.tight_layout()
plt.savefig(output_img_path/'encoded_sources_z.pdf')
fig,ax = plt.subplots(**kwds_subplots)
for i in range(5):
    ax[i].imshow(z_deter[i,1:].T,**kwds,vmin=0,vmax=z_deter.max(),extent=[0,T,0,z_hat.shape[-1]])
    ax[i].set_xlabel('time (s)')
plt.tight_layout()
plt.savefig(output_img_path/'encoded_sources_mu.pdf')

# %%
fig,ax = plt.subplots(**kwds_subplots)
for i in range(5):
    ax[i].imshow(s_hat[0,i],**kwds,vmin=-10,vmax=s_hat.max(),extent=[0,T,0,sr//2/1e3])
    ax[i].set_xlabel('time (s)')
ax[0].set_ylabel('frequency (kHz)')
plt.tight_layout()
plt.savefig(output_img_path/'stft_sources_z.pdf')
fig,ax = plt.subplots(**kwds_subplots)
for i in range(5):
    ax[i].imshow(s_deter[0,i],**kwds,vmin=-10,vmax=s_deter.max(),extent=[0,T,0,sr//2/1e3])
    ax[i].set_xlabel('time (s)')
ax[0].set_ylabel('frequency (kHz)')
plt.tight_layout()
plt.savefig(output_img_path/'stft_sources_mu.pdf')


# %%
torch.manual_seed(1)
with torch.no_grad():
    M = 1
    samp_s,samp_z,samp_h = model.sample(N=z_hat.shape[-2],M=M)
fig,ax = plt.subplots(M,5,figsize=(6,2*M),sharex=True,sharey=True)
ax = np.atleast_2d(ax)
vmax = samp_s.log().max()
for j in range(M):
    for i in range(5):
        ax[j,i].imshow(samp_s[j,i].log(),**kwds,vmin=-10,vmax=vmax,extent=[0,samp_s.shape[-1]*128/sr,0,sr//2/1e3])
        ax[j,0].set_ylabel('frequency (kHz)')
        ax[-1,i].set_xlabel('time (s)')
plt.tight_layout()
plt.savefig(output_img_path/'stft_sources_sampled.pdf')
# %%
# samp_source = pipeline.inverse_stft(samp_s.exp() * torch.exp(0*1j*torch.randn(samp_s.shape)))
gl = GriffinLim(n_fft=512,n_iter=100,win_length=512,hop_length=128,power=1)
samp_source = gl(samp_s ** (1/.5))
samp_source /= samp_source.abs().max() * 2

#%%
samp_source = samp_source.view(-1,samp_source.shape[-1])
# idx = samp_source.abs().sum(-1).argsort(descending=True)
# samp_source = samp_source[idx]
#%%
for j in range(5):
    torchaudio.save(output_img_path / "samped_{}.wav".format(j),samp_source[[j]],16000)
# %%
