import torch
from torch import nn, optim
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model import *
from dataset import *
import numpy as np
from argparser import parser

args = parser.parse_args(args=[])
torch.manual_seed(args.seed)

args.batch_size = 8

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

train_set = NSYNTH(root="/Users/julian/Documents/Code/data/TwoNote")
train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch_size,shuffle=True)

args.complex_version = False
nfft = 512
dim_data = (nfft//2+1,args.n_time)
mixer = Mixer(n_sources=args.num_targets,average=args.average_mix)
pipeline = AudioPipeline(n_fft=nfft,hop_length=128,compression=args.compression,n_time=args.n_time,output_realimag=args.complex_version)
pipeline = pipeline.to(device)

print("Trainset -> found {} files.".format(len(train_set)))

model_type = "mixit"

dimx = dim_data[0]
dimz = args.latent_dim
dimh = args.rnn_dim
dimg = args.rnn_dim
lr = 1e-4 * args.world_size

model_name = "model_mixit.pt"

kernel_size=(3,3)
stride=(2,2)
padding=(1,1)

args.autoencoder = True

args.num_sources = 4

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
    activate_final_layer=args.activate_final_layer,
    rnn_type=args.rnn_type,
    inference_rnn=(not args.inference_rnn_off),
    autoencoder=args.autoencoder,
    )
        
model = DCAE(**kwds_model)

model.to(device)


lossfun = nn.MSELoss(reduction="none")

comb_possible = [
    [0,1,2,3],
    [0,2,1,3],
    [0,3,1,2]
]

combs1 = np.array([[0,1],[0,2],[0,3]])
combs2 = np.array([[2,3],[1,3],[1,2]])

the_combs1 = np.concatenate([combs1,combs2])
the_combs2 = np.concatenate([combs2,combs1])

optimizer = torch.optim.Adam(params=model.parameters(),lr=1e-3)


for batch_idx, batch in enumerate(train_loader):
    
    optimizer.zero_grad()

    mix = data['mix'].to(device)
    mix = pipeline(mix.squeeze(1))
    inputs, targets = mixer(mix)

    output = model(inputs.unsqueeze(1))
    estim_sources = output['recon_s']

    estim_mix_1 = estim_sources[:,the_combs1].sum(2)
    estim_mix_2 = estim_sources[:,the_combs1].sum(2)
    
    estims = torch.stack([estim_mix_1,estim_mix_2],2)


    losses = lossfun(torch.stack(estims,targets.unsqueeze(1)).mean([2,3,4])
    best_option = losses.argmin(1)

    loss = losses[np.arange(8),best_option].mean()
    
    loss.backward()
    optimizer.step()
    
    print("{:4.4f}".format(loss.item()))