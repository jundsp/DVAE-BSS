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
from utils.utils import torchToNumpy, mag2db

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import time
from datetime import datetime
from subprocess import call
import json

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
    
def save_result(x,x_hat,s_hat,filename='result.png',epoch=1):    
    # Mask estimate
    if x.ndim == 4:
        x = (x**2.0).sum(1).sqrt()
    if x_hat.ndim == 4:
        x_hat = (x_hat**2.0).sum(1).sqrt()
    if s_hat.ndim == 5:
        s_hat = (s_hat**2.0).sum(2).sqrt()
        
    M = create_masks(x,s_hat)
        
    alpha = make_invariant(x_hat[0], x[0])
    x_hat[0] *= alpha
    s_hat[0] *= alpha
        
    X = torchToNumpy(mag2db(x[0]))
    X_hat = torchToNumpy(mag2db(x_hat[0]))
    S_hat = torchToNumpy(mag2db(s_hat[0]))
    M = torchToNumpy(mag2db(M[0]))
    
    kwds = dict(cmap='jet',interpolation='none',vmin=-100,vmax=12)
    if args.dataset != 'MNIST': 
        kwds['origin'] = 'lower'
        kwds['aspect'] = 'auto'
    fig = plt.figure(figsize=(10,7))
    fig,ax = plt.subplots(2,1+args.num_sources,figsize=((1+args.num_sources)*2.5,6))
    ax[0,0].imshow(X,**kwds)
    ax[1,0].imshow(X_hat,**kwds) 
    for s in range(args.num_sources):
        ax[0,s+1].imshow(M[s],**kwds)
        ax[1,s+1].imshow(S_hat[s],**kwds)
    
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
    writer.add_scalar(str(category)+'/ELL',info['ell'], epoch)
    writer.add_scalar(str(category)+'/KLD',info['kld'], epoch)

def train(epoch,lossfun,train_loader,train_rank):
    model.train()
    info = {'loss':[],'ell':[],'kld':[]}
    
    epoch_start = time.time()
    
    for batch_idx, (data,_) in enumerate(train_loader):
        
        data, sources = mixer(data)
        data = pipeline(data.to(device))

        output = model(data)
        ell, kld = lossfun(data,output)
        loss = ell + kld
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        info['loss'].append(loss)
        info['ell'].append(ell)
        info['kld'].append(kld)

        if (batch_idx+1) % args.log_interval == 0:
            if train_rank == 0:
                print(('Epoch {:2.0f} - {:3.0f} % ==> | Loss = {:4.6f}\t| ELL = {:4.6f}\t| KLD = {:4.6f}').format(
                    epoch,100.0*(batch_idx+1)/float(len(train_loader)),loss.item(),ell.item(),kld.item()))
                
    
    info = dict_average(info)
    for l in info:
        dist.reduce(info[l], dst=0, op = dist.ReduceOp.AVG)
        
    if train_rank == 0:
        elapse_time = time.time() - epoch_start
        print("From Rank: {}, Training time {:4.4f}".format(train_rank, elapse_time))
        print(('Epoch {:2.0f} ==> \t TRAIN | Loss = {:4.6f}\t| ELL = {:4.6f}\t| KLD = {:4.6f}').format(epoch, info['loss'].item(), info['ell'].item(), info['kld'].item()))
        save_info(info,category='Train',epoch=epoch)
        
    return info

def test(epoch,lossfun,test_loader,test_rank):
    model.eval()
    info = {'loss':[],'ell':[],'kld':[]}
    with torch.no_grad():
        for batch_idx, (data,_) in enumerate(test_loader):
            data, sources = mixer(data)
            data = pipeline(data.to(device))

            output = model(data)
            ell, kld = lossfun(data,output)
            loss = ell + kld

            info['loss'].append(loss)
            info['ell'].append(ell)
            info['kld'].append(kld)

        
        info = dict_average(info)
        for l in info:
            dist.reduce(info[l], dst=0, op = dist.ReduceOp.AVG)
                
        if test_rank == 0:
            print(('\t \t VALIDATE | Loss = {:4.6f}\t| ELL = {:4.6f}\t| KLD = {:4.6f}').format(info['loss'].item(), info['ell'].item(), info['kld'].item()))
            save_result(data,output['recon_x'],output['recon_s'],filename=path_results+'/result.png',epoch=epoch)
            # save_sample(N=output['mu_q'].size(1),filename=path_results+'/sample.png',epoch=epoch)
            save_info(info,category='Test',epoch=epoch)
    return info


if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    print(args)
    
    ngpus_per_node = torch.cuda.device_count()

    """ This prints information about the GPUs.
    """
    call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])

    """ This next line is the key to getting DistributedDataParallel working on SLURM:
		SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the 
 		current process inside a node and is also 0 or 1 in this example."""

    local_rank = int(os.environ.get("SLURM_LOCALID")) 
    rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank

    current_device = local_rank

    torch.cuda.set_device(current_device)
    
    """ this block initializes a process group and initiate communications
		between all processes running on all nodes """

    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    #init the process group
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)
    print("process group ready!")
    
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

    train_set, test_set = get_datasets(dataset=args.dataset,path=args.data_directory,args=args)

    # pipeline = ImagePipeline(n_src=args.num_sources,duration=args.duration,shift=.1)

    nfft = 512
    dim_data = (nfft//2+1,args.n_time)

    mixer = Mixer(n_sources=args.num_targets,average=args.average_mix)
    pipeline = AudioPipeline(n_fft=nfft,hop_length=128,compression=args.compression,n_time=args.n_time,output_realimag=args.complex_version)
    # mixer = mixer.to(device)
    pipeline = pipeline.to(device)

    print("Trainset -> found {} files.".format(len(train_set)))
    print("Testset -> found {} files.".format(len(test_set)))
    print('From Rank: {}, ==> Preparing data..'.format(rank))
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=(test_sampler is None), num_workers=args.num_workers, sampler=test_sampler)
    
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
    lr = 1e-4

    with open(path_saves+'/configuration.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    print('From Rank: {}, ==> Making model..'.format(rank))

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
                model = SVAE(mag_output=True,**kwds_model)
            else:
                model = DCAE(**kwds_model)
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
        
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[current_device])
    
    lossfun = LossFunction(complex_version=args.complex_version,laplace=args.laplace).to(device)

    optimizer = optim.AdamW(params=model.parameters(),lr=lr,weight_decay=2e-5)
            
    for epoch in range(1,args.epochs+1):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        
        train_info = train(epoch,lossfun=lossfun,train_loader=train_loader,train_rank=rank)
        test_info = test(epoch,lossfun=lossfun,test_loader=test_loader,test_rank=rank)
        
        if epoch % args.save_interval == 0:
            torch.save(model.module.state_dict(),model_name)

