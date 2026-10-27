#%%
import torch
from utils.utils import load_state_dict, optimal_permute
import numpy as np
from lossfun import FreqSignalLoss
from dataset import get_data_loaders
from torch.nn import functional as F
from model import *
from dataset import AudioPipeline, AuditoryStreams
import torchaudio
from pathlib import Path
from torchaudio.transforms import GriffinLim
import os
import json

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
dataset = AuditoryStreams(root="data/TwoNote/")
idx = dataset._info[(dataset._info['instrument1']=='forced_vibrato') & (dataset._info['instrument2']=='forced_vibrato') & (dataset._info['delta_time']==0.4425) & (dataset._info['delta_semitone']==9)].index.item()
data = dataset.__getitem__(idx)

audio,sources_true = data['mix'], data['stems']
T = audio.shape[-1]/sr

audio += torch.randn(audio.shape)*1e-5
pipeline = AudioPipeline(n_fft=512,n_time=257+128,hop_length=128,compression=.5)
x = pipeline(audio)
sources_true_spec = pipeline(sources_true.unsqueeze(0))

#%%
for K in range(2,7):

    model_path = Path("output/svae/RWC/sources/K{}/".format(K))
    output_img_path = model_path / "properties"
    os.makedirs(output_img_path,exist_ok=True)

    fname = model_path / "configuration.txt"
    with open(fname) as f:
        data = f.read()
    args = json.loads(data)

    model = DCAE(num_sources=K,dimx=257,dimz=256,dimh=512,dimg=512,kernel_size=(3,3),stride=(2,2),padding=(1,1),layer_channels=(32,64,128,128,128),activate_final_layer=True)


    path = model_path / "svae_rwc_K{}_c5.pt".format(K)
    model.load_state_dict(load_state_dict(path=path))
    model.eval()

    with torch.no_grad():
        features = model.Encoder(x=x.unsqueeze(1))
        features_encoded = model.feature_encoder(x)
        output_stoch = model.forward(x,stochastic=True)
        x_hat = output_stoch['recon_x'].log()
        s_hat = output_stoch['recon_s'].log()
        z_for_decoder = output_stoch['z'].transpose(-2,-1)
        z_hat = output_stoch['z'].abs()


        output_deter = model.forward(x,stochastic=False)
        s_deter = output_deter['recon_s']
        z_from_encoder = output_deter['z'].transpose(-2,-1)
        z_deter = output_deter['z'].abs()

        samp_s,samp_z,samp_h = model.sample(N=z_hat.shape[1],M=2)

        temp,_ = optimal_permute(s_deter.view(s_deter.shape[0],s_deter.shape[1],-1),sources_true_spec.reshape(sources_true_spec.shape[0],sources_true_spec.shape[1],-1))
        s_deter = temp.reshape(s_deter.shape).log()


    kwds = dict(aspect='auto',origin='lower',interpolation='none',cmap='Greys',vmin=-8,extent=[0,T,0,sr//2/1e3])
    kwds_subplots = dict(nrows=1,ncols=6,sharey=True,figsize=(8,2))
    plt.figure(figsize=(2.5,2))
    plt.imshow(x[0].log(),**kwds);
    plt.xlabel('time (s)')
    plt.ylabel('frequency (kHz)')
    plt.xlim(0)
    plt.tight_layout()
    plt.savefig(output_img_path/'stft_mix.pdf')

    plt.figure(figsize=(2.5,2))
    plt.imshow(x_hat[0],**kwds);
    plt.xlabel('time (s)')
    plt.ylabel('frequency (kHz)')
    plt.tight_layout()
    plt.savefig(output_img_path/'stft_mix_estimate.pdf')


    # fig,ax = plt.subplots(**kwds_subplots)
    # for i in range(K):
    #     ax[i].imshow(z_hat[i,1:].T,**kwds,vmin=0,vmax=z_hat.max(),extent=[0,T,0,z_hat.shape[-1]])
    #     ax[i].set_xlabel('time (s)')
    # plt.tight_layout()
    # plt.savefig(output_img_path/'encoded_sources_z.pdf')
    # fig,ax = plt.subplots(**kwds_subplots)
    # for i in range(K):
    #     ax[i].imshow(z_deter[i,1:].T,**kwds,vmin=0,vmax=z_deter.max(),extent=[0,T,0,z_hat.shape[-1]])
    #     ax[i].set_xlabel('time (s)')
    # plt.tight_layout()
    # plt.savefig(output_img_path/'encoded_sources_mu.pdf')


    # fig,ax = plt.subplots(**kwds_subplots)
    # for i in range(K):
    #     ax[i].imshow(s_hat[0,i],**kwds,vmin=-10,vmax=s_hat.max(),extent=[0,T,0,sr//2/1e3])
    #     ax[i].set_xlabel('time (s)')
    # ax[0].set_ylabel('frequency (kHz)')
    # plt.tight_layout()
    # plt.savefig(output_img_path/'stft_sources_z.pdf')

    fig,ax = plt.subplots(**kwds_subplots)
    for i in range(K):
        ax[i].imshow(s_deter[0,i],**kwds)
        ax[i].set_xlabel('time (s)')
    for i in range(K,6):
        ax[i].remove()
    ax[0].set_ylabel('frequency (kHz)')
    plt.tight_layout()
    plt.savefig(output_img_path/'stft_sources_mu.pdf')
    plt.close()

fig,ax = plt.subplots(**kwds_subplots)
for i in range(2):
    ax[i].imshow(sources_true_spec[0,i].log(),**kwds)
    ax[i].set_xlabel('time (s)')
for i in range(2,6):
    ax[i].remove()
ax[0].set_ylabel('frequency (kHz)')
plt.tight_layout()
plt.savefig(output_img_path/'stft_sources_ground_truth.pdf')
plt.close()

# %%
K = 4

model_path = Path("output/svae/RWC/sources/K{}/".format(K))
path = model_path / "svae_rwc_K{}_c5.pt".format(K)
K_plot = K

# model_path = Path("output/svae/VCTK/230129-13274778/t2")
# path = model_path / "svae_VCTK_S2_K4_L1_pooled_D256_C5-10.pt"
# K_plot = 2 

output_img_path = model_path / "properties"
os.makedirs(output_img_path,exist_ok=True)

fname = model_path / "configuration.txt"
with open(fname) as f:
    data = f.read()
args = json.loads(data)

model = DCAE(num_sources=K,dimx=257,dimz=256,dimh=512,dimg=512,kernel_size=(3,3),stride=(2,2),padding=(1,1),layer_channels=(32,64,128,128,128),activate_final_layer=True)


model.load_state_dict(load_state_dict(path=path))
model.eval()

torch.manual_seed(2)
with torch.no_grad():
    M = 5
    samp_s,samp_z,samp_h = model.sample(N=40,M=M)

kwds_sample = {}
for i in kwds:
    if i not in ["vmin","vmax","extent"]:
        kwds_sample[i] = kwds[i]
vmax = samp_s.log().max()

#%%
for j in range(M):
    fig,ax = plt.subplots(1,K_plot,figsize=(6,2),sharex=True,sharey=True)
    ax = np.atleast_2d(ax)
    for i in range(K_plot):
        ax[0,i].imshow(samp_s[j,i].log(),**kwds_sample,vmin=-10,vmax=vmax,extent=[0,samp_s.shape[-1]*128/sr,0,sr//2/1e3])
        ax[0,0].set_ylabel('frequency (kHz)')
        ax[0,i].set_xlabel('time (s)')
        ax[0,i].set_xlim(0)
    plt.tight_layout()
    plt.savefig(output_img_path/'stft_sources_sampled_{}.pdf'.format(j+1))

# %%
# samp_source = pipeline.inverse_stft(samp_s.exp() * torch.exp(0*1j*torch.randn(samp_s.shape)))
gl = GriffinLim(n_fft=512,n_iter=100,win_length=512,hop_length=128,power=1)
samp_source = gl(samp_s ** (1/.5))
samp_source /= samp_source.abs().max() * 2

#%%
samp_source = samp_source.view(-1,samp_s.shape[1],samp_source.shape[-1])
# idx = samp_source.abs().sum(-1).argsort(descending=True)
# samp_source = samp_source[idx]
#%%
for i in range(samp_source.shape[0]):
    for j in range(K_plot):
        torchaudio.save(output_img_path / "samped_{}_{}.wav".format(i+1,j+1),samp_source[i,[j]],16000)
# %%

Features = []
f = x.unsqueeze(1)
for i,m in enumerate(model.Encoder.Network):
    f = m(f)
    temp = f.clone().detach()[0]
    temp -= temp.min()
    temp /= temp.max()
    # temp_sum = temp.abs().sum(-1).sum(-1)
    # idx = torch.argsort(temp_sum,descending=True)
    # temp = temp[idx]
    Features.append(temp)


#%%
for ii,f in enumerate(Features):
    fig,ax = plt.subplots(4,3,sharex=True,sharey=True,figsize=(3,2.8))
    
    ax = ax.ravel()
    
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05,hspace=0.05)

    for i in range(12):
        ax[i].axis('off')
        ax[i].imshow(f[i],interpolation="none",origin='lower',aspect='equal',cmap='Greys')
    
    plt.savefig(output_img_path/'features_{}.png'.format(ii+1))
# %%

fig,ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=(1.3,1))
ax = np.atleast_2d(ax)
ax = ax.ravel()
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.subplots_adjust(wspace=0.05,hspace=0.05)
# ax[0].axis('off')
# ax[0].set_title(r"$y$")
ax[0].set_xlabel(r"$t$")
ax[0].imshow(x[0],interpolation="none",origin='lower',aspect='equal',cmap='Greys')

plt.savefig(output_img_path/'features_{}.png'.format(0))
# %%


z = model.fc_decoder(z_for_decoder)
z = z.view(z.shape[0],model.conv_channels,-1,z.shape[-1])

n = 1
Features_dec = []
f = z
temp = f.clone().detach()[n]
temp -= temp.min()
temp /= temp.max()
Features_dec.append(temp)
for i,m in enumerate(model.Decoder.Network):
    f = m(f)
    temp = f.clone().detach()[n]
    temp -= temp.min()
    temp /= temp.max()
    # temp_sum = temp.abs().sum(-1).sum(-1)
    # idx = torch.argsort(temp_sum,descending=True)
    # temp = temp[idx]
    Features_dec.append(temp)


#%%
for ii,f in enumerate(Features_dec[:-1]):
    fig,ax = plt.subplots(4,3,sharex=True,sharey=True,figsize=(3,2.8))
    
    ax = ax.ravel()
    
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05,hspace=0.05)

    for i in range(12):
        ax[i].axis('off')
        ax[i].imshow(f[i],interpolation="none",origin='lower',aspect='equal',cmap='Greys')
    
    plt.savefig(output_img_path/'features_dec_{}.png'.format(ii+1))
# %%
fig,ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=(1.33,1))
ax = np.atleast_2d(ax)
ax = ax.ravel()
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.subplots_adjust(wspace=0.05,hspace=0.05)
# ax[0].axis('off')
ax[0].set_xlabel(r"$t$")
# ax[0].set_title(r"$y$")
ax[0].imshow(Features_dec[-1][0],interpolation="none",origin='lower',aspect='equal',cmap='Greys')

plt.savefig(output_img_path/'features_dec_{}.png'.format(6))
# %%

fig,ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=(1.33,1))
ax = np.atleast_2d(ax)
ax = ax.ravel()
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.subplots_adjust(wspace=0.05,hspace=0.05)
# ax[0].axis('off')
ax[0].imshow(z_for_decoder[n].abs(),interpolation="none",origin='lower',aspect='auto',cmap='Greys')
ax[0].set_xlabel(r"$\tau$")
# ax[0].set_title(r"$x$")
plt.savefig(output_img_path/'features_dec_z.png')
# %%

fig,ax = plt.subplots(2,1,sharex=True,sharey=True,figsize=(1.33,1.5))
ax = np.atleast_2d(ax)
ax = ax.ravel()
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.subplots_adjust(wspace=0.05,hspace=0.05)
# ax[0].axis('off')
ax[0].imshow(z_from_encoder[1].abs(),interpolation="none",origin='lower',aspect='auto',cmap='Greys')
ax[1].imshow(z_from_encoder[4].abs(),interpolation="none",origin='lower',aspect='auto',cmap='Greys')
ax[1].set_xlabel(r"$\tau$")
# ax[0].set_title(r"$x$")
plt.savefig(output_img_path/'features_enc_z.png')
# %%
