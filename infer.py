#%%
import torch
from torch import nn, optim
from torch.nn import functional as F
import math
import torchaudio
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import argparse
from torchvision import datasets, transforms
from model import *
from dataset import *
from torchvision.utils import make_grid, save_image
from argparser import parser
from utils.utils import optimal_permute
from collections import OrderedDict
import seaborn as sns
from utils.utils import torchToNumpy, mag2db
import glob
import re

#* Make a class / module to process the bss eval measures.

def load_state_dict(path):
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


def mp2comp(magnitude,phase):
    Zc = magnitude*torch.exp(1j*phase)
    Z = torch.view_as_real(Zc)
    return Z

def pad_full(Z,nfft):
    out = torch.zeros(Z.size(0),nfft//2+1,Z.size(2),2)
    out[:,:Z.size(1)] = Z
    return out

# Batch Synthesize
def synthesize(magnitude,phase,istft):
    X = mp2comp(magnitude=magnitude,phase=phase)
    X = pad_full(X) 
    x = istft(X)
    return x


#%%

#%%
args = parser.parse_args(args=[])
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Check the parameters of the model saved in that directory.

os.chdir(args.model_directory)
model_name = glob.glob("*.pt")[0]
match = re.match(r'svae_([^_]+)_K(\d+)_L1_pooled_D(\d+)_C(\d+)-10.pt', model_name)

args.num_sources = int(match.group(2))
args.latent_dim = int(match.group(3))
args.compression = float(match.group(4)) * .1
args.conv2d = True
args.rnn_dim = 1024

print(args.num_sources)
print(args.latent_dim)
print(args.compression)

args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

nfft = 512
averaged = False
dim_data = (nfft//2+1,args.duration)

mixer = Mixer(n_sources=2,average=averaged)
args.n_time = 257
pipeline = AudioPipeline(n_fft=nfft,hop_length=128,compression=args.compression,n_time=args.n_time)
mixer = mixer.to(device)
pipeline = pipeline.to(device)

#%% LOAD MODEL

dimx = dim_data[0]
dimz = args.latent_dim
dimh = args.rnn_dim
dimg = args.rnn_dim
kernel_size=(3,3)
stride=(2,2)
padding=(1,1)

model = DCAE(
        dimx=dimx,
        dimz=dimz,
        dimh=dimh,
        dimg=dimg,
        num_sources=args.num_sources,
        net_type=args.net_type,
        num_layers=args.num_layers,
        scale_invariant=True,
        layer_channels=(32,64,128,128),
        kernel_size=kernel_size,stride=stride,padding=padding,
        )

model.load_state_dict(load_state_dict(path=args.model_directory + "/" + model_name))
model.to(device)
model.eval()
pipeline.compression = args.compression

input_file = "/Users/julian/Documents/MATLAB/sounds/OriginalMix.wav"
data,sr = torchaudio.load(input_file)

with torch.no_grad():
    mix, sources = mixer(waveform=data.to(device))
    mix_stft = pipeline.forward_complex(mix)
    source_stft = pipeline.forward_complex(sources)

    data = mix_stft.abs()
    output = model(data)

    s_hat = output['recon_s']
    m_hat = create_masks(data,s_hat)
    estimates_stft = m_hat * torch.exp(1j*mix_stft.angle().unsqueeze(1))

    mix = pipeline.invert(mix_stft)
    estimates = pipeline.invert(estimates_stft)

    estimates = optimal_permute(estimates,torch.stack([mix]*2,1))


# %%
kwds = dict(cmap='jet',origin='lower',aspect='auto',interpolation='none',vmax=15)
pipeline.compression = 1.0
n = 0
fig,ax = plt.subplots(3,args.num_sources,figsize=(3*args.num_sources,9))
ax[0,0].imshow(mag2db(pipeline(mix[n])+1e-9).cpu(),**kwds)
ax[1,0].imshow(mag2db(pipeline(sources[n,0])+1e-9).cpu(),**kwds)
ax[1,1].imshow(mag2db(pipeline(sources[n,1])+1e-9).cpu(),**kwds)
for i in range(args.num_sources):
    ax[2,i].imshow(mag2db(pipeline(estimates[n,i])+1e-9).cpu(),**kwds)

ax[0,0].set_title('Input Mixture')
ax[1,0].set_title('Target')
ax[2,0].set_title('Estimates')

for i in range(1,args.num_sources):
    ax[0,i].remove()
for i in range(2,args.num_sources):
    ax[1,i].remove()
# %%

torchaudio.save(args.model_directory+ "/mix.wav",mix[n].unsqueeze(0).cpu() / 2,sr)
for i in range(estimates.shape[1]):
    torchaudio.save(args.model_directory + "/estimate_{}.wav".format(i+1),estimates[n,i].unsqueeze(0).cpu() / 2,sr)
    
# %%
