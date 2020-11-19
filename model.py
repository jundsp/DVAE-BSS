import torch
from torch import nn, optim
from torch.nn import functional as F
import math
import torchaudio
import os
import matplotlib.pyplot as plt
import argparse
from torchvision import datasets, transforms

class DSAE(nn.Module):
    def __init__(self, dimx, dimz, dimh=64, dimg=64, num_sources=1, net_type='pooled', num_layers=1):
        super(DSAE, self).__init__()
        self.dimx, self.dimz, self.dimh, self.dimg = dimx, dimz, dimh, dimg
        self.num_sources = num_sources
        self.num_layers = num_layers

        self.gru_generate = nn.LSTM(self.dimz,self.dimh,num_layers=self.num_layers,batch_first=True)
        self.gru_infer = nn.LSTM(dimg,dimg,num_layers=self.num_layers,bidirectional=True,batch_first=True)

        chans = 64

        self.mu_q = nn.Sequential(
            nn.Linear(dimg,chans),
            nn.ReLU(),
            nn.Linear(chans,dimz),
            )
        self.logvar_q = nn.Sequential(
            nn.Linear(dimg,chans),
            nn.ReLU(),
            nn.Linear(chans,dimz),
            )

        self.mu0_q = nn.Sequential(
            nn.Linear(dimg,chans),
            nn.ReLU(),
            nn.Linear(chans,dimz),
            )
        self.logvar0_q = nn.Sequential(
            nn.Linear(dimg,chans),
            nn.ReLU(),
            nn.Linear(chans,dimz),
            )

        self.mu_p = nn.Sequential(
            nn.Linear(dimh,chans),
            nn.ReLU(),
            nn.Linear(chans,dimz),
            )
        self.logvar_p = nn.Sequential(
            nn.Linear(dimh,chans),
            nn.ReLU(),
            nn.Linear(chans,dimz),
            )

        self.transition_z = nn.Sequential(
            nn.Linear(self.dimz,self.dimh),
            nn.Tanh(),
            )
        
        if net_type == 'pooled':
            self.Encoder = PooledEncoder(dimx,dimg*num_sources)
            self.Decoder = PooledDecoder(dimz,dimx)
        else: 
            self.Encoder = StridedEncoder(dimx,dimg*num_sources)
            self.Decoder = StridedDecoder(dimz,dimx)

        self.likelihood = NormalLikelihood()
        self.criteria = nn.BCELoss(reduction='sum')
        self.register_buffer('variance',torch.tensor(0.1))
        self.register_parameter('smu',nn.Parameter(torch.zeros(1,num_sources,self.dimz)))
        self.register_parameter('smu0',nn.Parameter(torch.zeros(1,num_sources,self.dimz)))
        self.register_buffer('slogvar',torch.zeros(1,num_sources,self.dimz))
        self.register_buffer('slogvar0',torch.zeros(1,num_sources,self.dimz))

        self.varmin = 1e-6

    def get_variance(self):
        # return F.softplus(100.0*self.variance)/100.0+self.varmin
        return self.variance*1.0

    def encode(self,x):
        gx = self.Encoder(x)
        gx = gx.view(-1,self.dimg,gx.size(-1)).transpose(-2,-1)
        g0 = torch.zeros(2*self.num_layers,gx.size(0),self.dimg,device=x.device)
        c0 = torch.zeros(2*self.num_layers,gx.size(0),self.dimg,device=x.device)
        g,_ = self.gru_infer(gx,(g0,c0))
        g = 1/2*(g[:,:,:self.dimg] + g[:,:,self.dimg:])
        mu = self.mu_q(g)
        logvar = self.logvar_q(g)
        mu0 = self.mu0_q(g[:,0].unsqueeze(1))
        logvar0 = self.logvar0_q(g[:,0].unsqueeze(1))
        return mu, logvar, mu0, logvar0

    def decode(self,z):
        x = self.Decoder(z.transpose(-2,-1))
        return x
 
    def combiner(self,z,h):
        return 1/2*(self.transition_z(z) + h)

    def marginal(self,mean_1,logvar_1,mean_2,logvar_2):
        ivar1 = torch.exp(-logvar_1)
        ivar2 = torch.exp(-logvar_2)
        ell = mean_1*ivar1 + mean_2*ivar2
        lam = ivar1 + ivar2
        mu = ell/lam
        logvar = -lam.log()
        return mu, logvar

    def batch_and_source(self,x,n):
        return x.repeat(n,1,1).view(-1,1,self.dimz)

    def prior_stats(self,zh):
        n = zh.size(0) // self.num_sources
        mu, logvar = self.mu_p(zh), self.logvar_p(zh)
        source_mean = self.batch_and_source(self.smu,n)
        source_logvar = self.batch_and_source(self.slogvar,n)
        mu, logvar = self.marginal(mu,logvar,source_mean,source_logvar)
        return mu, logvar

    def prior(self,z,z0):
        h0 = torch.zeros(self.num_layers,z.size(0),self.dimh,device=z.device)
        c0 = torch.zeros(self.num_layers,z.size(0),self.dimh,device=z.device)
        zt = torch.cat([z0,z[:,:-1]],1)
        h,_ = self.gru_generate(zt,(h0,c0))
        zh = self.combiner(zt,h)
        mu,logvar = self.prior_stats(zh)
        return mu, logvar

    def sample(self,N=10,M=1,device='cpu'):
        MK = M * self.num_sources
        z = torch.zeros(MK,N,self.dimz,device=device)
        h = torch.zeros(MK,N,self.dimh,device=device)
        h_t = torch.zeros(self.num_layers,MK,self.dimh,device=device)
        c_t = torch.zeros(self.num_layers,MK,self.dimh,device=device)
        prior_mean = self.batch_and_source(self.smu0,M)
        prior_logvar = self.batch_and_source(self.slogvar0,M)
        z_t = torch.randn(MK,1,self.dimz,device=device)*torch.exp(0.5*prior_logvar) + prior_mean
        for n in range(N):
            _,(h_t,c_t) = self.gru_generate(z_t,(h_t,c_t))
            zh = self.combiner(z_t,h_t[-1].unsqueeze(1))
            mu, logvar = self.prior_stats(zh)
            z_t = reparameterize(mu,logvar)
            z[:,n] = z_t.squeeze(1)
            h[:,n] = h_t[-1]
        
        s = self.decode(z)
        s = s.view(-1,self.num_sources,s.size(-2),s.size(-1))
        return s, z, h

    def loss_function(self,x,mu_x,mu,logvar,mu_prior,logvar_prior, mu0, logvar0 ):
        n = x.size(0)
        ell = self.likelihood(x, mu_x, var=self.get_variance()) / n
        # ell = self.criteria(mu_x,x) / n        
        kld = KLD(mu,logvar,prior_mean=mu_prior,prior_logvar=logvar_prior) / n
        prior_mean = self.batch_and_source(self.smu0,n)
        prior_logvar = self.batch_and_source(self.slogvar0,n)
        kld += KLD(mu0,logvar0,prior_mean=prior_mean,prior_logvar=prior_logvar) / n
        return ell , kld

    def forward(self, x):
        mu_q, logvar_q, mu0_q, logvar0_q = self.encode(x)
        z = reparameterize(mu_q,logvar_q)
        z0 = reparameterize(mu0_q,logvar0_q)
        mu_p, logvar_p = self.prior(z,z0)
        mu_s = self.decode(z)
        recon_s = mu_s.view(-1,self.num_sources,mu_s.size(-2),mu_s.size(-1))
        recon_x = recon_s.sum(1)
        return recon_x, recon_s, mu_q, logvar_q, mu_p, logvar_p, mu0_q, logvar0_q


'''
Encoder/Decoders for Features.
1) Strided
2) Pooled
'''

class StridedEncoder(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(StridedEncoder, self).__init__()
        self.Network = nn.Sequential(
            nn.Conv1d(input_channels,128,kernel_size=4,stride=2,padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            # nn.Conv1d(64,128,kernel_size=4,stride=2,padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128,256,kernel_size=4,stride=2,padding=1), nn.ReLU(), nn.BatchNorm1d(256),
            # nn.Conv1d(256,256,kernel_size=4,stride=2,padding=1), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Conv1d(256,output_channels,kernel_size=3,stride=1,padding=1), nn.ReLU(),
            )
    def forward(self, x):
        return self.Network(x)

class StridedDecoder(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(StridedDecoder, self).__init__()
        self.Network = nn.Sequential(
            nn.ConvTranspose1d(input_channels,256,kernel_size=4,stride=2,padding=1), nn.ReLU(), nn.BatchNorm1d(256),
            # nn.ConvTranspose1d(256,256,kernel_size=4,stride=2,padding=1), nn.ReLU(), nn.BatchNorm1d(256),
            nn.ConvTranspose1d(256,128,kernel_size=4,stride=2,padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            # nn.ConvTranspose1d(128,64,kernel_size=4,stride=2,padding=1), nn.ReLU(),
            nn.ConvTranspose1d(128,output_channels,kernel_size=3,stride=1,padding=1), nn.Sigmoid(), 
            )
    def forward(self, x):
        return self.Network(x)

# Pooled
class PooledEncoder(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(PooledEncoder, self).__init__()
        pooling = nn.MaxPool1d(kernel_size=2,stride=2)
        # pooling = SincInterpolation(scale_factor=.5)
        self.Network = nn.Sequential(
            nn.Conv1d(input_channels,64,kernel_size=3,stride=1,padding=1), pooling, nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64,128,kernel_size=3,stride=1,padding=1), pooling, nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128,256,kernel_size=3,stride=1,padding=1), pooling, nn.ReLU(), nn.BatchNorm1d(256),
            nn.Conv1d(256,256,kernel_size=3,stride=1,padding=1), pooling, nn.ReLU(), nn.BatchNorm1d(256),
            nn.Conv1d(256,output_channels,kernel_size=3,stride=1,padding=1), nn.ReLU(),
            )
    def forward(self, x):
        return self.Network(x)

class PooledDecoder(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(PooledDecoder, self).__init__()
        upsample = nn.Upsample(scale_factor=2,mode='linear',align_corners=False)
        # upsample = SincInterpolation(scale_factor=2)
        self.Network = nn.Sequential(
            nn.ConvTranspose1d(input_channels,256,kernel_size=3,stride=1,padding=1), nn.ReLU(), nn.BatchNorm1d(256), upsample,
            nn.ConvTranspose1d(256,256,kernel_size=3,stride=1,padding=1), nn.ReLU(), nn.BatchNorm1d(256), upsample,
            nn.ConvTranspose1d(256,128,kernel_size=3,stride=1,padding=1), nn.ReLU(), nn.BatchNorm1d(128), upsample,
            nn.ConvTranspose1d(128,64,kernel_size=3,stride=1,padding=1), nn.ReLU(), upsample,
            nn.ConvTranspose1d(64,output_channels,kernel_size=3,stride=1,padding=1), nn.Sigmoid(),
            )
    def forward(self, x):
        return self.Network(x)

class SincInterpolation(nn.Module):
    def __init__(self,scale_factor=1):
        super(SincInterpolation, self).__init__()
        self.scale_factor = scale_factor

    def sinc(self,x):
        xp = math.pi*x
        sinc = torch.sin(xp) / xp
        sinc[x==0] = 1.0
        return sinc
        
    def forward(self, x):
        n = x.size(-1)
        nf = int(n*self.scale_factor)
        u = torch.linspace(0,nf-1,nf,device=x.device).view(1,1,-1)/self.scale_factor
        t = torch.linspace(0,n-1,n,device=x.device).view(1,-1,1)
        Phi = self.sinc(u-t)
        y = x @ Phi
        return y
'''
Auxiliary
'''

class Transpose(nn.Module):
    def __init__(self, dim1=-2,dim2=-1):
        super(Transpose, self).__init__()
        self.dim = (dim1,dim2)

    def forward(self, x):
        return x.transpose(self.dim[0],self.dim[1])

def KLD(mean_1, logvar_1, prior_mean=0.0, prior_logvar=0.0):
    var_1 = logvar_1.exp()
    var_2 = prior_logvar.exp()
    return  -0.5 * torch.sum(1 + logvar_1 - prior_logvar - (var_1 + (mean_1 - prior_mean).pow(2)) / var_2)

class NormalLikelihood(nn.Module):
    def __init__(self, variance=None, reduction='sum'):
        super(NormalLikelihood, self).__init__()
        if variance is None:
            variance = torch.tensor(1.0)
        self.register_buffer("variance",variance)

    def forward(self, x, mean, var=None):
        if var is None:
            var = self.variance
        return 0.5*torch.sum(var.log() + (x-mean).pow(2) / var)

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

# model = DSAE(dimx=32,dimz=10,num_sources=2,net_type='pooled')
# x = model.sample(N=4,M=10)[0].sum(1) / 2
# recon_x, recon_s, mu_q, logvar_q, mu_p, logvar_p, mu0_q, logvar0_q = model(x)
# print(x.size())
# print(recon_x.size())
