#%%
import torch
from torch import nn, optim
from torch.nn import functional as F
import os
from torchvision import datasets, transforms
from model import *
from dataset import *
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from argparser import parser

import matplotlib.pyplot as plt

args = parser.parse_args(args=[])
torch.manual_seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print(args)

kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

nfft = 512
dim_data = (nfft//2+1,args.n_time)

args.complex_version = True
args.num_sources = 2
args.compression = .5
args.n_time = 257
mixer = Mixer(n_sources=args.num_targets,average=args.average_mix)
pipeline = AudioPipeline(n_fft=nfft,hop_length=128,compression=args.compression,n_time=args.n_time,output_realimag=args.complex_version)
pipeline = pipeline.to(device)

dimx = dim_data[0]
dimz = args.latent_dim
dimh = args.rnn_dim
dimg = args.rnn_dim
lr = 1e-3

kernel_size=(3,3)
stride=(2,2)
padding=(1,1)

kwds_model = dict(
    dimx=dimx,
    dimz=dimz,
    dimh=dimh,
    dimg=dimg,
    num_sources=args.num_sources,
    net_type=args.net_type,
    num_layers=args.num_layers,
    scale_invariant=True,
    layer_channels=tuple(args.cnn_channels),
    kernel_size=kernel_size,stride=stride,padding=padding,
    variance=args.variance,
    activate_final_layer=args.activate_final_layer
    )
    
if args.complex_version:
    model = SVAE(**kwds_model)
else:
    model = DCAE(**kwds_model)

model.to(device)

lossfun = LossFunction().to(device)

optimizer = optim.AdamW(params=model.parameters(),lr=lr,weight_decay=2e-5)

filename = "clar"
input_file = "/Users/julian/Documents/MATLAB/sounds/{}.wav".format(filename)
audio,sr = torchaudio.load(input_file)
audio = torchaudio.functional.resample(audio,orig_freq=sr,new_freq=16000.0,resampling_method='kaiser_window')
audio = audio.to(device)

audio = torch.cat([audio]*5,0)

#%%
model.train() 
for epoch in range(1,200):
    data = pipeline(audio)
    output = model(data)

    ell, kld = lossfun(data,output)
    loss = ell + kld
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 1:
        print("Epoch {} : Loss = {}".format(epoch,loss.item()))

kwds = dict(aspect='auto',origin='lower',cmap='jet',interpolation='none')

#%%

if args.complex_version:
    output_complex = torch.complex(output['recon_x'][:,0],output['recon_x'][:,1]).detach()
    output_mag = output_complex.abs()
    source_complex = torch.complex(output['recon_s'][:,:,0],output['recon_s'][:,:,1]).detach()
    source_mags = source_complex.abs()
    data_mag = torch.complex(data[:,0],data[:,1]).abs()
else:
    output_mag = output['recon_x'].detach()
    data_mag = data
    source_mags = output['recon_s'].detach()

plt.subplot(211)
plt.imshow(mag2db(data_mag[0]),**kwds)
plt.subplot(212)
plt.imshow(mag2db(output_mag[0]),**kwds)

#%%
fig,ax = plt.subplots(args.num_sources)
ax = np.atleast_1d(ax).ravel()
for k in range(args.num_sources):
    ax[k].imshow(mag2db(source_mags[0,k]),**kwds)

# # %%
# sources = pipeline.invert(source_complex)
# plt.subplot(212)
# plt.plot(sources[0].T)
# plt.subplot(211)
# plt.plot(audio[0])
# # %%

# %%
