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
from datetime import datetime
import matplotlib.pyplot as plt
from utils.utils import torchToNumpy, mag2db, load_state_dict

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import time
from datetime import datetime
from subprocess import call
import json
from pathlib import Path
from itertools import permutations

def get_combs(n_sources):
    n_outputs = n_sources * 2

    l = np.array(list(permutations(range(n_outputs))))
    l1 = l[:,:n_sources]
    l2 = l[:,n_sources:]

    l1 = np.sort(l1,-1)
    l2 = np.sort(l2,-1)

    indexes = np.unique(l1, return_index=True,axis=0)[1]
    indexes = np.sort(indexes)
    l1 = l1[indexes]

    indexes = np.unique(l2, return_index=True,axis=0)[1]
    indexes = np.sort(indexes)
    l2 = l2[indexes]

    return [l1,l2]

def dict_average(input):
    avg = {}
    for k,v in input.items():
        avg[k] = sum(v) / float(len(v))
    return avg

def save_result_grid(x,x_hat,s_hat,filename='result.png',epoch=1):
    nrows = 2+args.num_sources
    togrid = torch.zeros(nrows,1,x.size(-2),x.size(-1))
    togrid[0,0] = x[0]
    togrid[1,0] = x_hat[0].detach()
    for k in range(args.num_sources):
        togrid[2+k,0] = s_hat[0,k].detach()
    grid = make_grid(togrid.detach(),nrow=1).cpu()
    save_image(grid,filename)
    writer.add_image('Results', grid, epoch)
    
def save_result(x_hat,s_hat,filename='result.png',epoch=1):            
        
    X_hat = torchToNumpy(mag2db(x_hat[0]))
    S_hat = torchToNumpy(mag2db(s_hat[0]))
    
    cols = S_hat.shape[0]
    
    kwds = dict(cmap='jet',interpolation='none',vmin=-100,vmax=12)
    if args.dataset != 'MNIST': 
        kwds['origin'] = 'lower'
        kwds['aspect'] = 'auto'
    fig = plt.figure(figsize=(10,7))
    fig,ax = plt.subplots(2,cols,figsize=((1+cols)*2.5,6))
    for s in range(cols):
        ax[0,s].imshow(X_hat[s],**kwds)
        ax[1,s].imshow(S_hat[s],**kwds)
    
    ax[0,0].set_title('epoch = {}'.format(epoch))
    plt.tight_layout()
    plt.savefig(filename)    
    writer.add_figure(tag='Results', figure=fig)
    plt.close()    

def save_sample(N=10,filename='result.png',epoch=1):
    N = int(N)
    
    s,z,_ = model.module.sample(N=N,device=device)
    S = torchToNumpy( mag2db(s[0,:]) )
    Z = torchToNumpy( z[0,:] )
            
    fig,ax = plt.subplots(2,figsize=(6,6))
    kwds = dict(cmap='jet',interpolation='none')
    if args.dataset != 'MNIST': 
        kwds['origin'] = 'lower'
        kwds['aspect'] = 'auto'
    ax[0].imshow(S[0],**kwds)
    ax[0].set_title('epoch = {}'.format(epoch))
    ax[1].imshow(Z.T,**kwds)
    plt.tight_layout()
    plt.savefig(filename)    
    writer.add_figure(tag='Samples', figure=fig)
    plt.close()  

def save_info(info,category='Train',epoch=1):
    writer.add_scalar(str(category)+'/Loss',info['loss'],epoch)

def train(epoch,lossfun,train_loader,train_rank):
    model.train()
    info = {'loss':[]}
    
    epoch_start = time.time()
    
    for batch_idx, (data,_) in enumerate(train_loader):
        
        mix, sources = mixer(data)
        mix = pipeline(mix.squeeze(1).to(device))
        inputs, targets = mixer(mix)
        
        optimizer.zero_grad()

        estim_sources = model(inputs.unsqueeze(1))

        estim_mix_1 = estim_sources[:,the_combs[0]].sum(2)
        estim_mix_2 = estim_sources[:,the_combs[1]].sum(2)
        
        estims = torch.stack([estim_mix_1,estim_mix_2],2)

        losses = lossfun(estims,targets.unsqueeze(1)).mean([2,3,4])
        best_option = losses.argmin(1)
        
        loss = losses[np.arange(best_option.shape[0]),best_option].mean()
    
        loss.backward()
        optimizer.step()

        info['loss'].append(loss)

        if (batch_idx+1) % args.log_interval == 0:
            if train_rank == 0:
                print(('Epoch {:2.0f} - {:3.0f} % ==> | Loss = {:4.6f}\t ').format(
                    epoch,100.0*(batch_idx+1)/float(len(train_loader)),loss.item()))
                
    
    info = dict_average(info)
        
    if train_rank == 0:
        elapse_time = time.time() - epoch_start
        print("From Rank: {}, Training time {:4.4f}".format(train_rank, elapse_time))
        print(('Epoch {:2.0f} ==> \t TRAIN | Loss = {:4.6f}').format(epoch, info['loss'].item()))
        save_info(info,category='Train',epoch=epoch)
        
    return info

def test(epoch,lossfun,test_loader,test_rank):
    model.eval()
    info = {'loss':[]}
    with torch.no_grad():
        for batch_idx, (data,_) in enumerate(test_loader):
            
            mix, sources = mixer(data)
            mix = pipeline(mix.squeeze(1).to(device))
            inputs, targets = mixer(mix)
            
            estim_sources = model(inputs.unsqueeze(1))

            estim_mix_1 = estim_sources[:,the_combs[0]].sum(2)
            estim_mix_2 = estim_sources[:,the_combs[1]].sum(2)
            
            estims = torch.stack([estim_mix_1,estim_mix_2],2)

            losses = lossfun(estims,targets.unsqueeze(1)).mean([2,3,4])
            best_option = losses.argmin(1)
            
            loss = losses[np.arange(best_option.shape[0]),best_option].mean()

            info['loss'].append(loss)

        info = dict_average(info)
                
        if test_rank == 0:
            print(('\t \t VALIDATE | Loss = {:4.6f}\t ').format(info['loss'].item()))
            save_result(targets.detach().cpu(),estims[np.arange(best_option.shape[0]),best_option].detach().cpu(),filename=path_results+'/result.png',epoch=epoch)
            # save_sample(N=output['mu_q'].size(1),filename=path_results+'/sample.png',epoch=epoch)
            save_info(info,category='Test',epoch=epoch)
    return info


args = parser.parse_args()
torch.manual_seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print(args)

kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}


train_set, test_set = get_datasets(dataset=args.dataset,path=args.data_directory,args=args)

nfft = 512
dim_data = (nfft//2+1,args.n_time)

mixer = Mixer(n_sources=args.num_targets,average=args.average_mix,random_num=False)
pipeline = AudioPipeline(n_fft=nfft,hop_length=128,compression=args.compression,n_time=args.n_time,output_realimag=args.complex_version)
# mixer = mixer.to(device)
pipeline = pipeline.to(device)

print("Trainset -> found {} files.".format(len(train_set)))
print("Testset -> found {} files.".format(len(test_set)))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,  num_workers=args.num_workers)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)

root = args.output_directory

current_time = datetime.now().strftime('%y%m%d')
sub_dir = os.path.join(args.dataset,current_time + '-' + args.job_id)

path_results = os.path.join(root,'results',sub_dir)
path_saves = os.path.join(root,'saves',sub_dir)
path_runs = os.path.join(root,'runs',sub_dir)

os.makedirs(path_results,exist_ok=True)
os.makedirs(path_saves,exist_ok=True)
os.makedirs(path_runs,exist_ok=True)

model_type = str(args.model_type)
writer = SummaryWriter(log_dir=path_runs,comment=model_type + '_' + str(args.dataset))

dimx = dim_data[0]
dimz = args.latent_dim
dimh = args.rnn_dim
dimg = args.rnn_dim
lr = args.lr * args.world_size

args.autoencoder = True

with open(path_saves+'/configuration.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

model_name = ('{}/{}_{}_S{}_K{}_L{}_{}_D{}_C{}-{}.pt').format(
    path_saves,
    model_type,
    args.dataset,
    str(args.num_targets),
    str(args.num_sources),
    str(args.num_layers),
    args.net_type,
    args.latent_dim,
    int(args.compression*10),
    int(10)
    )

if model_type == 'vae':
    dimx = int(dim_data[0]*dim_data[1])
    model = VAE(dimx=dimx,dimz=int(args.latent_dim),n_sources=int(args.num_sources))
else:
    if args.conv2d:
        if args.separator_version:
            kernel_size=(3,3)
            stride=(2,1)
            padding=(1,1)
        else:
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
            activate_final_layer=args.activate_final_layer,
            rnn_type=args.rnn_type,
            inference_rnn=(not args.inference_rnn_off),
            autoencoder=args.autoencoder,
            )
            
        if args.complex_version:
            model = SVAE(out_type="activated",**kwds_model)
        elif args.separator_version:
            model = DSVAE(separator_layers=args.separator_layers,separator_kernel_size=args.separator_kernel_size,separator_channels=args.separator_channels,**kwds_model)
        else:
            model = DCAE_MixIT(**kwds_model)
    else:
        model = DSAE(
        dimx=dimx,
        dimz=dimz,
        dimh=dimh,
        dimg=dimg,
        num_sources=args.num_sources,
        net_type='pooled',
        num_layers=args.num_layers,
        scale_invariant=True,
        layer_channels=tuple(args.cnn_channels),
        )

if args.checkpoint is not None:
    try:
        path_checkpoint =((Path(root) / 'saves') / args.dataset)  / args.checkpoint
        path_checkpoint = list(path_checkpoint.glob("*.pt"))[0]
        model.load_state_dict(load_state_dict(path=path_checkpoint))
        model.eval()
        model.train()
        print("Loaded model from checkpoint!")
        torch.manual_seed(np.random.randint(0,1000))
    except Exception as e: 
        print(e)
        print("Failed to load checkpoint! Continuing with new model.")
    print(args.checkpoint)

print(model)
    
model.to(device)

lossfun = nn.MSELoss(reduction="none").to(device)

the_combs = get_combs(n_sources=args.num_sources//2)

optimizer = optim.AdamW(params=model.parameters(),lr=lr,weight_decay=2e-5)
        
for epoch in range(1,args.epochs+1):
    train_info = train(epoch,lossfun=lossfun,train_loader=train_loader,train_rank=0)
    test_info = test(epoch,lossfun=lossfun,test_loader=test_loader,test_rank=0)
    
    if epoch % args.save_interval == 0:
        torch.save(model.state_dict(),model_name)

