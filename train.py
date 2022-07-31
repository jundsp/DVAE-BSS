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
import socket
from datetime import datetime

def dict_average(input):
    avg = {}
    for k,v in input.items():
        avg[k] = sum(v) / float(len(v))
    return avg

def save_result(x,x_hat,s_hat,filename='result.png',epoch=1):
    m = 8
    nrows = 2+args.num_sources
    n = nrows*m
    togrid = torch.zeros(n,1,x.size(-2),x.size(-1))
    togrid[::nrows] = x[:m].unsqueeze(1)
    togrid[1::nrows] = x_hat[:m].detach().unsqueeze(1)
    for k in range(args.num_sources):
        togrid[2+k::nrows] = s_hat[:m,k].detach().unsqueeze(1)
    grid = make_grid(togrid.detach(),nrow=nrows).cpu()
    # save_image(grid,filename)
    writer.add_image('Results', grid, epoch)

def save_sample(N=10,filename='result.png',epoch=1):
    sample = model.sample(M=4,N=N,device=device)[0][0].unsqueeze(1)
    grid = make_grid(sample.detach(),nrow=1).cpu()
    # save_image(grid,filename)
    writer.add_image('Samples', grid, epoch)

def save_info(info,category='Train',epoch=1):
    writer.add_scalar(str(category)+'/Loss',info['loss'],epoch)
    writer.add_scalar(str(category)+'/ELL',info['ell'], epoch)
    writer.add_scalar(str(category)+'/KLD',info['kld'], epoch)

def process_data(data):
    perms = torch.randperm(data.size(0))
    data = torch.cat([data,data[perms]],-1)
    if args.mix_data == 'True':
        perms = torch.randperm(data.size(0))
        data = 0.5*(data + data[perms])
    return data.squeeze(1)

def train(epoch):
    model.train()
    info = {'loss':[],'ell':[],'kld':[]}
    for batch_idx, (data,_) in enumerate(train_loader):
        data = process_data(data.to(device))
        if args.dataset == 'MUMS':
            data = torch.norm(data,2,-1)

        optimizer.zero_grad()
        recon_x, recon_s, mu_q, logvar_q, mu_p, logvar_p, mu0_q, logvar0_q = model(data)
        ell, kld = model.loss_function(data,recon_x, mu_q, logvar_q, mu_p, logvar_p, mu0_q, logvar0_q)
        loss = ell + kld
        loss.backward()
        optimizer.step()

        info['loss'].append(loss.item())
        info['ell'].append(ell.item())
        info['kld'].append(kld.item())

        if batch_idx % args.log_interval == 0:
            print(('Epoch {:2.0f} -{:3.0f} % ==> | Loss = {:4.2f}\t| ELL = {:4.2f}\t| KLD = {:4.2f}').format(
                epoch,100.0*(batch_idx+1)/float(len(train_loader)),loss.item(),ell.item(),kld.item()))

    info = dict_average(info)
    print(('Epoch {:2.0f} ==> Train Loss = {:4.2f}').format(epoch, info['loss']))
    save_info(info,category='Train',epoch=epoch)
    return info

def test(epoch):
    model.eval()
    info = {'loss':[],'ell':[],'kld':[]}
    with torch.no_grad():
        for batch_idx, (data,_) in enumerate(test_loader):
            data = process_data(data.to(device))
            if args.dataset == 'MUMS':
                data = torch.norm(data,2,-1)

            recon_x, recon_s, mu_q, logvar_q, mu_p, logvar_p, mu0_q, logvar0_q = model(data)
            ell, kld = model.loss_function(data,recon_x, mu_q, logvar_q, mu_p, logvar_p, mu0_q, logvar0_q)
            loss = ell + kld

            info['loss'].append(loss.item())
            info['ell'].append(ell.item())
            info['kld'].append(kld.item())

        info = dict_average(info)
        print(('Epoch {:2.0f} ==> Test Loss = {:4.2f}').format(epoch, info['loss']))
        save_result(data,recon_x,recon_s,filename=path_results+'/result_'+str(epoch)+'.png',epoch=epoch)
        save_sample(N=mu_q.size(1)*4,filename=path_results+'/sample_'+str(epoch)+'.png',epoch=epoch)
        save_info(info,category='Test',epoch=epoch)
    return info


if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
    print(device)

    if args.dataset == 'MNIST':
        dim_data = (32,32)
        # args.data_directory = '/Users/julian/Documents/Code/Compute_Canada/data'
        transform = transforms.Compose([transforms.CenterCrop(size=dim_data),transforms.RandomAffine(degrees=0,translate=(0.25,0.0)),transforms.ToTensor()])
        train_set = datasets.MNIST(args.data_directory, train=True, transform=transform, download=True)
        test_set = datasets.MNIST(args.data_directory, train=False, transform=transform, download=True)

    if args.dataset == 'MUMS':
        dim_data = (128,256)
        # args.data_directory = '/Volumes/Julian SSD/data/MUMS'
        train_set = MUMS(args.data_directory,train=True,duration=dim_data[1], bins=dim_data[0],magnitude=False)
        test_set = MUMS(args.data_directory, train=False, duration=dim_data[1], bins=dim_data[0],magnitude=False)

    if args.dataset == 'Audio':
        dim_data = (128,256)
        # args.data_directory = '/Users/julian/Documents/Code/data/nsynth'
        train_set = Audioset(args.data_directory, spectrogram=True, duration=dim_data[1], bins=dim_data[0])
        test_set = Audioset(args.data_directory, spectrogram=True, duration=dim_data[1], bins=dim_data[0])
        
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    root = os.path.join(args.output_directory,'SVAE')
    
    current_time = datetime.now().strftime('%y%m%d_%H-%M-%S')
    sub_dir = os.path.join(args.dataset,current_time+ '_' + socket.gethostname())

    path_results = os.path.join(root,'results',sub_dir)
    path_saves = os.path.join(root,'saves',sub_dir)
    path_runs = os.path.join(root,'runs',sub_dir)

    os.makedirs(path_results,exist_ok=True)
    os.makedirs(path_saves,exist_ok=True)
    os.makedirs(path_runs,exist_ok=True)

    writer = SummaryWriter(log_dir=path_runs,comment='SVAE_' + str(args.dataset))

    dimx = dim_data[0]
    dimz = args.latent_dim
    dimh = args.rnn_dim
    dimg = args.rnn_dim
    lr = 2e-4

    model_name = ('{}/svae_{}_K{}_L{}_{}_D{}.pt').format(path_saves,args.dataset,str(args.num_sources),str(args.num_layers),args.net_type,args.latent_dim)

    model = DSAE(
        dimx=dimx,
        dimz=dimz,
        dimh=dimh,
        dimg=dimg,
        num_sources=args.num_sources,
        net_type=args.net_type,
        num_layers=args.num_layers,
        ).to(device)

    optimizer = optim.Adam(params=model.parameters(),lr=lr)
        
    for epoch in range(1,args.epochs+1):
        train_info = train(epoch)
        test_info = test(epoch)

        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(),model_name)
