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
    M = s_hat * (x /  s_hat.sum(1)).unsqueeze(1)
    
    X = torchToNumpy(x)
    X_hat = torchToNumpy(x_hat)
    S_hat = torchToNumpy(s_hat)
    M = torchToNumpy(M)
    kwds = dict(cmap='jet',interpolation='none',vmin=0,vmax=2)
    if args.dataset != 'MNIST': 
        kwds['origin'] = 'lower'
        kwds['aspect'] = 'auto'
    fig = plt.figure(figsize=(10,7))
    fig,ax = plt.subplots(2,3,figsize=(8,6))
    ax[0,0].imshow(X[0],**kwds)
    ax[1,0].imshow(X_hat[0],**kwds)
    ax[1,1].imshow(S_hat[0,0],**kwds)
    ax[0,1].imshow(M[0,0],**kwds)    
    for s in range(1,args.num_sources):
        ax[1,s+1].imshow(S_hat[0,s],**kwds)
        ax[0,s+1].imshow(M[0,s],**kwds)
    
    ax[0,0].set_title('epoch = {}'.format(epoch))
    plt.tight_layout()
    plt.savefig(filename)    
    writer.add_figure('Results', fig, epoch)
    plt.close()    

def save_sample(N=10,filename='result.png',epoch=1):
    
    s = model.sample(N=N,device=device)[0]
    S = torchToNumpy( s[0,:] )
            
    fig = plt.figure(figsize=(6,4))
    kwds = dict(cmap='jet',interpolation='none',vmin=0,vmax=2)
    if args.dataset != 'MNIST': 
        kwds['origin'] = 'lower'
        kwds['aspect'] = 'auto'
    plt.imshow(S[0],**kwds)
    plt.title('epoch = {}'.format(epoch))
    plt.savefig(filename)    
    writer.add_figure('Samples', fig, epoch)
    plt.close()  

def save_info(info,category='Train',epoch=1):
    writer.add_scalar(str(category)+'/Loss',info['loss'],epoch)
    writer.add_scalar(str(category)+'/ELL',info['ell'], epoch)
    writer.add_scalar(str(category)+'/KLD',info['kld'], epoch)

def process_data(data,duration=256):
    n = data.shape[0]
        
    k = duration // data.shape[-1]
    m = n // k
    datacat = torch.zeros(m,data.shape[-2],duration)
    for mm in range(m):
        idx = mm*k
        datacat[mm] = torch.cat([data[i+idx] for i in range(k)],-1)
                
    data = datacat + 0
    if args.mix_data is True:
        n = data.size(0)
        perms = torch.randperm(n)
        data = (data[:n//2] + data[n//2:n//2*2])
                        
    return data

def train(epoch):
    model.train()
    info = {'loss':[],'ell':[],'kld':[]}
    for batch_idx, (data,_) in enumerate(train_loader):
        data = pipeline(data.to(device))

        optimizer.zero_grad()
        recon_x, recon_s, mu_q, logvar_q, mu_p, logvar_p, mu0_q, logvar0_q = model(data)
        print(recon_x.shape)
        print(data.shape)
        ell, kld = model.loss_function(data,recon_x, mu_q, logvar_q, mu_p, logvar_p, mu0_q, logvar0_q)
        loss = ell + kld
        loss.backward()
        optimizer.step()

        info['loss'].append(loss.item())
        info['ell'].append(ell.item())
        info['kld'].append(kld.item())

        if (batch_idx+1) % args.log_interval == 0:
            print(('Epoch {:2.0f} - {:3.0f} % ==> | Loss = {:4.2f}\t| ELL = {:4.2f}\t| KLD = {:4.2f}').format(
                epoch,100.0*(batch_idx+1)/float(len(train_loader)),loss.item(),ell.item(),kld.item()))

    info = dict_average(info)
    print(('Epoch {:2.0f} ==> \t TRAIN | Loss = {:4.2f}\t| ELL = {:4.2f}\t| KLD = {:4.2f}').format(epoch, info['loss'], info['ell'], info['kld']))
    save_info(info,category='Train',epoch=epoch)
    return info

def test(epoch):
    model.eval()
    info = {'loss':[],'ell':[],'kld':[]}
    with torch.no_grad():
        for batch_idx, (data,_) in enumerate(test_loader):
            data = pipeline(data.to(device))

            recon_x, recon_s, mu_q, logvar_q, mu_p, logvar_p, mu0_q, logvar0_q = model(data)
            ell, kld = model.loss_function(data,recon_x, mu_q, logvar_q, mu_p, logvar_p, mu0_q, logvar0_q)
            loss = ell + kld

            info['loss'].append(loss.item())
            info['ell'].append(ell.item())
            info['kld'].append(kld.item())

        info = dict_average(info)
        print(('\t \t VALIDATE | Loss = {:4.2f}\t| ELL = {:4.2f}\t| KLD = {:4.2f}').format(info['loss'], info['ell'], info['kld']))
        save_result(data,recon_x,recon_s,filename=path_results+'/result.png',epoch=epoch)
        save_sample(N=mu_q.size(1),filename=path_results+'/sample.png',epoch=epoch)
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
        transform = transforms.Compose([transforms.CenterCrop(size=dim_data),transforms.RandomAffine(degrees=0,translate=(0.2,0.2)),transforms.ToTensor()])
        train_set = datasets.MNIST(args.data_directory, train=True, transform=transform, download=True)
        test_set = datasets.MNIST(args.data_directory, train=False, transform=transform, download=True)
        pipeline = ImagePipeline(n_src=args.num_sources,duration=args.duration)

    if args.dataset == 'MUMS':
        dim_data = (128,args.duration)
        train_set = Audioset(args.data_directory + '/MUMS/DVDs', spectrogram=True, duration=dim_data[1], bins=dim_data[0])
        test_set = Audioset(args.data_directory + '/MUMS/DVDs/WOODWINDS/SAXOPHONES/', spectrogram=True, duration=dim_data[1], bins=dim_data[0])
                
    if args.dataset == 'RWC':
        nfft = 512
        dim_data = (nfft//2+1,args.duration)
        transform = transforms.Compose([Shift(shift_amount=.1)])
        train_set = Audioset(root=args.data_directory,path_csv='rwc_dataset/validate.csv',duration=6,nsrc=2,transform=transform,)
        test_set =  Audioset(root=args.data_directory,path_csv='rwc_dataset/test.csv',duration=6,nsrc=2,transform=transform,)
        pipeline = AudioPipeline(n_time=args.duration,n_fft=nfft,hop_length=64,compression=1)
        
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    pipeline = pipeline.to(device)
    
    print("Trainset -> found {} files.".format(len(train_set)))
    print("Testset -> found {} files.".format(len(test_set)))

    root = args.output_directory
    
    current_time = datetime.now().strftime('%y%m%d')
    sub_dir = os.path.join(args.dataset,current_time + '-' + args.job_id)

    path_results = os.path.join(root,'results',sub_dir)
    path_saves = os.path.join(root,'saves',sub_dir)
    path_runs = os.path.join(root,'runs',sub_dir)

    os.makedirs(path_results,exist_ok=True)
    os.makedirs(path_saves,exist_ok=True)
    os.makedirs(path_runs,exist_ok=True)

    writer = SummaryWriter(log_dir=path_runs,comment='svae_' + str(args.dataset))

    dimx = dim_data[0]
    dimz = args.latent_dim
    dimh = args.rnn_dim
    dimg = args.rnn_dim
    lr = 2e-4

    model_name = ('{}/svae_{}_K{}_L{}_{}_D{}.pt').format(path_saves,args.dataset,str(args.num_sources),str(args.num_layers),args.net_type,args.latent_dim)

    model = DCAE(
        dimx=dimx,
        dimz=dimz,
        dimh=dimh,
        dimg=dimg,
        num_sources=args.num_sources,
        net_type=args.net_type,
        num_layers=args.num_layers,
        scale_invariant=True,
        layer_channels=(32,64,128)
        ).to(device)
    
    print(model)
    
    optimizer = optim.Adam(params=model.parameters(),lr=lr)
            
    for epoch in range(1,args.epochs+1):
        train_info = train(epoch)
        test_info = test(epoch)

        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(),model_name)
