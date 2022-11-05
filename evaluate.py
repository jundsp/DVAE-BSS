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
from sklearn.manifold import  MDS
import seaborn as sns
from argparser import parser
from utils.utils import optimal_permute


jet = cm.get_cmap('jet', 256)
newcolors = jet(np.linspace(0, 1, 256))
newcolors[0] = (0.0,0.0,0.0,1.0)
mycmp = ListedColormap(newcolors)

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

def dict_average(input):
    avg = {}
    for k,v in input.items():
        avg[k] = sum(v) / float(len(v))
    return avg

#%%
args = parser.parse_args(args=[])
torch.manual_seed(args.seed)
np.random.seed(args.seed)
args.data_directory = "/Volumes/JulianSSD/data"
args.duration = 4
args.n_time = 257
args.compression = .5
args.batch_size = 4

args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

nfft = 512
dim_data = (nfft//2+1,args.duration)
test_set = Audioset(root=args.data_directory + '/RWC_dataset',path_csv='test.csv',augment_levels=args.augment_levels,duration=float(args.duration),silence=0,max_files=2)
mixer = Mixer(n_sources=args.num_sources)
pipeline = AudioPipeline(n_fft=nfft,hop_length=128,compression=args.compression,n_time=args.n_time)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,shuffle=True)

mixer = mixer.to(device)
pipeline = pipeline.to(device)

#%% LOAD MODEL

args.latent_dim = 128
args.rnn_dim = 512

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

model_path = "/Users/julian/Desktop/svae_RWC_K2_L1_pooled_D128.pt"
# original saved file with DataParallel
state_dict = torch.load(model_path,map_location=torch.device(device))
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)


#%% EVALUATE

def si_sdr(estimate,target,salt=1e-14):
    Rss = (target * target).sum(-1)
    alpha = (estimate * target).sum(-1) / Rss
    e_target = target * alpha.unsqueeze(-1)
    e_residual = estimate - e_target
    E_target = (e_target * e_target).sum(-1) + salt
    E_residual = (e_residual * e_residual).sum(-1) + salt

    output = 10*(E_target.log10() - E_residual.log10())
    return output

model.eval()

Mix = []
Targets = []
Estimates = []
SiSDR = []

#%%
with torch.no_grad():
    # for batch_idx, (data,_) in enumerate(test_loader):
    data,_ = next(iter(test_loader))
    mix, sources = mixer(data.to(device))
    mix_stft = pipeline.forward_complex(mix)
    source_stft = pipeline.forward_complex(sources)

    data = mix_stft.abs()
    output = model(data)

    m_hat = create_masks(data,output['recon_s'])
    estimates_stft = m_hat * torch.exp(1j*mix_stft.angle().unsqueeze(1))

    mix = pipeline.invert(mix_stft)
    sources = pipeline.invert(source_stft)
    estimates = pipeline.invert(estimates_stft)

    estimates = optimal_permute(estimates,sources)

    sisdr = si_sdr(estimate=estimates,target=sources)

    Mix.append(mix)
    Targets.append(sources)
    Estimates.append(estimates)
    SiSDR.append(sisdr)


# %%
n = 1
kwds = dict(cmap='jet',origin='lower',aspect='auto',interpolation='none',vmin=-100)
fig,ax = plt.subplots(2,3,figsize=(7,4))
ax[0,0].imshow(mag2db(pipeline(mix[n])+1e-8),**kwds)
ax[0,1].imshow(mag2db(pipeline(sources[n,0])+1e-8),**kwds)
ax[1,1].imshow(mag2db(pipeline(sources[n,1])+1e-8),**kwds)
ax[0,2].imshow(mag2db(pipeline(estimates[n,0])+1e-8),**kwds)
ax[1,2].imshow(mag2db(pipeline(estimates[n,1])+1e-8),**kwds)
ax[1,0].remove()

ax[0,0].set_title('Input Mixture')
ax[0,1].set_title('Target')
ax[0,2].set_title('Estimates')
plt.tight_layout()

# %%
torchaudio.save("/Users/julian/Desktop/mix.wav",mix[n].unsqueeze(0),16000)
for i in range(sources.shape[1]):
    torchaudio.save("/Users/julian/Desktop/source_{}.wav".format(i+1),sources[n,i].unsqueeze(0),16000)
    torchaudio.save("/Users/julian/Desktop/estimate_{}.wav".format(i+1),estimates[n,i].unsqueeze(0),16000)
# %%
