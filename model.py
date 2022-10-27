#%%
import torch
from torch import nn
import math
from utils.stft import STFT, iSTFT
import numpy as np
from lossfun import FreqSignalLoss
from dataset import get_data_loaders

class DSAE(nn.Module):
    def __init__(self, dimx, dimz, dimh=64, dimg=64, chans=64, num_sources=1, net_type='pooled', num_layers=1, scale_invariant=False):
        super(DSAE, self).__init__()
        self.dimx, self.dimz, self.dimh, self.dimg = dimx, dimz, dimh, dimg
        self.num_sources = num_sources
        self.num_layers = num_layers
        
        self.dec_chan = 128

        self.gru_generate = nn.LSTM(self.dimz,self.dimh,num_layers=self.num_layers,batch_first=True)
        self.gru_infer = nn.LSTM(dimg,dimg,num_layers=self.num_layers,bidirectional=True,batch_first=True)

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

        self.net_type = net_type
        
        if net_type == 'pooled':
            self.Encoder = PooledEncoder(dimx,dimg*num_sources)
            self.Decoder = PooledDecoder(dimz,dimx)
        else: 
            self.Encoder = StridedEncoder(dimx,self.dec_chan)
            self.Decoder = StridedDecoder(self.dec_chan,dimx)

        self.activation = nn.Softplus()
        self.activation = nn.Sigmoid()
        # self.activation = nn.Identity()

        self.likelihood = NormalLikelihood(scale_invariant=scale_invariant)
        self.criteria = nn.BCELoss(reduction='sum')
        self.register_buffer('variance',torch.tensor(1))
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
        z = z.transpose(-2,-1)
        if self.net_type == 'strided':
            z = z.view(-1,self.dec_chan,self.dimz//self.dec_chan,z.shape[-1])
        x = self.Decoder(z)
        x = self.activation(x)
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
    
    def sample_latent(self,N,M,device):
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
        return z, h

    def sample(self,N=10,M=1,device='cpu'):
        z, h = self.sample_latent(N=N,M=M,device=device)
        
        s = self.decode(z)
        s = s.view(-1,self.num_sources,s.size(-2),s.size(-1))
        return s, z, h

    def loss_function(self,x,mu_x,mu,logvar,mu_prior,logvar_prior, mu0, logvar0 ):
        n = x.size(0)
        # Si-SDR? scale the target... but it is the mixture.
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
        mu_s = mu_s.view(-1,self.num_sources,mu_s.size(-2),mu_s.size(-1))
        recon_s = mu_s
        recon_x = recon_s.sum(1)
        return recon_x, recon_s, mu_q, logvar_q, mu_p, logvar_p, mu0_q, logvar0_q


'''
Encoder/Decoders for Features.
1) Strided
2) Pooled
'''

class StridedEncoder(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size=(3,2),stride=(2,1),padding=(1,0)):
        super(StridedEncoder, self).__init__()
        self.Network = nn.Sequential(
            nn.Conv2d(input_channels,32,kernel_size=kernel_size,stride=stride,padding=padding), nn.LeakyReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32,64,kernel_size=kernel_size,stride=stride,padding=padding), nn.LeakyReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64,64,kernel_size=kernel_size,stride=stride,padding=padding), nn.LeakyReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64,128,kernel_size=kernel_size,stride=stride,padding=padding), nn.LeakyReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128,output_channels,kernel_size=kernel_size,stride=stride,padding=padding), nn.LeakyReLU(),
            )
    def forward(self, x):
        return self.Network(x)

class StridedDecoder(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size=(3,2),stride=(2,1),padding=(1,0)):
        super(StridedDecoder, self).__init__()
        self.Network = nn.Sequential(
            nn.ConvTranspose2d(input_channels,128,kernel_size=kernel_size,stride=stride,padding=padding), nn.LeakyReLU(), nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128,64,kernel_size=kernel_size,stride=stride,padding=padding), nn.LeakyReLU(), nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64,64,kernel_size=kernel_size,stride=stride,padding=padding), nn.LeakyReLU(), nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64,32,kernel_size=kernel_size,stride=stride,padding=padding), nn.LeakyReLU(), nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32,output_channels,kernel_size=kernel_size,stride=stride,padding=padding), 
            )
    def forward(self, x):
        return self.Network(x)

# Pooled
class PooledEncoder(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(PooledEncoder, self).__init__()
        pooling = nn.MaxPool1d(kernel_size=2,stride=2,padding=0)
        self.Network = nn.Sequential(
            nn.Conv1d(input_channels,64,kernel_size=3,stride=1,padding=1), pooling, nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64,128,kernel_size=3,stride=1,padding=1), pooling, nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128,output_channels,kernel_size=3,stride=1,padding=1), nn.ReLU(),
            )
    def forward(self, x):
        return self.Network(x)

class PooledDecoder(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(PooledDecoder, self).__init__()
        upsample = nn.Upsample(scale_factor=2,mode='linear',align_corners=False)
        self.Network = nn.Sequential(
            nn.ConvTranspose1d(input_channels,128,kernel_size=3,stride=1,padding=1), nn.ReLU(), nn.BatchNorm1d(128), upsample,
            nn.ConvTranspose1d(128,64,kernel_size=3,stride=1,padding=1), nn.ReLU(), nn.BatchNorm1d(64), upsample,
            nn.ConvTranspose1d(64,output_channels,kernel_size=3,stride=1,padding=1), 
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
    def __init__(self, variance=None, reduction='sum', scale_invariant=False):
        super(NormalLikelihood, self).__init__()
        if variance is None:
            variance = torch.tensor(1.0)
        self.scale_invariant = scale_invariant
        self.register_buffer("variance",variance)

    def forward(self, x, mean, var=None):
        if var is None:
            var = self.variance

        alpha = torch.ones(x.shape[0],1,1)
        # mu = mean.clone().detach()
        # if self.scale_invariant:
        #     for i in range(x.shape[0]):
        #         alpha[i] = (x[i] * mu[i]).sum() / (x[i] * x[i]).sum()
        # eps = x*alpha - mean

        mu = mean.clone().detach()
        if self.scale_invariant:
            for i in range(x.shape[0]):
                alpha[i] = (x[i] * mu[i]).sum() / (mu[i] * mu[i]).sum()
                
        eps = x - mean*alpha

        return 0.5*torch.sum(var.log() + (eps).pow(2) / var)

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

def output_size(H_in, kernel_size, stride=1,  dilation=1, padding=0):
    H_out = H_in + 2*padding - dilation * (kernel_size - 1) - 1
    return H_out // stride + 1


class DCVAE(DSAE):
    def __init__(self, fs, nfft, chans=64, num_sources=1, net_type='pooled', num_layers=1):
        dimx = nfft//2+1
        self.dimx = dimx * 2
        if net_type == 'strided':
            self.dimx = 2
            
        H_in = dimx
        for i in range(5):
            H_in = output_size(H_in,kernel_size=3,stride=2,padding=1)
            
        self.net_type = net_type
            
        dimg = 128
        dimz = dimg 
        dimh = dimz
        chans = dimh
        super().__init__(self.dimx, dimz, dimh, dimg, chans, num_sources, self.net_type, num_layers)
        self.stft = STFT(fs=fs,nfft=nfft,winlen=.02,hopfrac=.5)
        self.istft = iSTFT(fs=fs,nfft=nfft,winlen=.02,hopfrac=.5)

    def forward(self,x):
        X = self.stft(x)
        
        if self.net_type == 'pooled':
            X = X.view(X.shape[0], X.shape[1], -1).transpose(-2,-1)
        else:
            X = X.transpose(1,-1)
            
        k = X.shape[-1]
        dsfac = 2**2
        pad_ = k % dsfac
        if pad_ > 0:
            pad_ = dsfac - pad_

        X = torch.cat([X,torch.zeros(X.shape[0],X.shape[1],pad_)],-1)
            
        recon_X, recon_S, mu_q, logvar_q, mu_p, logvar_p, mu0_q, logvar0_q = super().forward(X)

        if pad_ > 0:
            recon_X = recon_X[:,:,:-pad_]
            recon_S = recon_S[:,:,:-pad_]
        
        if self.net_type == 'pooled':
            recon_X = recon_X.transpose(-2,-1)
            recon_X = recon_X.view(recon_X.shape[0],recon_X.shape[1],-1,2)
        else:
            recon_X = recon_X.transpose(1,-1)
            
        recon_x = self.istft(recon_X,self.stft.padder)
        return recon_x,mu_q, logvar_q, mu_p, logvar_p, mu0_q, logvar0_q


#%%
from torch import optim
from argparser import parser
import os
from dataset import WHAMR
import matplotlib.pyplot as plt
import soundfile as sf

args = parser.parse_args(args=[])
torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AUDIO
s,fs = sf.read('/Users/julian/Documents/MATLAB/sounds/clar.wav')
stft = STFT(fs=fs,winlen=.02,hopfrac=.5,nfft=256,normalize=False)
x = stft(torch.from_numpy(s).view(1,-1)).float()
# x = torch.concat([x[:,:,:,0],x[:,:,:,1]],-1)
x = (x**2.0).sum(-1) ** .5
x = x.transpose(-2,-1)
# x = x / x.max()
x = x[:,:,:32]
dimx = 129

kwds = {}
train_loader, test_loader = get_data_loaders('/Users/julian/Documents/Code/data/mnist',60,kwds)


dimx = 28

model = DSAE(dimx=dimx,dimz=32,num_sources=1,scale_invariant=True)
model = model.to(device)

#%%
optimizer = optim.Adam(params=model.parameters(),lr=2e-4)
model.train()
#%%
epochs = 5
for epoch in range(epochs):
    for k,batch_full in enumerate(train_loader):
        batch = batch_full[0]
        batch = batch.view(-1,6,batch.shape[-2],batch.shape[-1])
        x = torch.cat([batch[:,i] for i in range(batch.shape[1])],-1)
        x = x[:,:,:128]


        optimizer.zero_grad()
        recon_x, recon_s, mu_q, logvar_q, mu_p, logvar_p, mu0_q, logvar0_q = model(x)
        ell, kld = model.loss_function(x,recon_x, mu_q, logvar_q, mu_p, logvar_p, mu0_q, logvar0_q)
        loss = ell + kld
        loss.backward()
        optimizer.step()

        if k % 100 == 0:
            print('Epoch {} \t Batch {}  \t KLD = {:4.2f} \t ELL = {:4.2f} \t Loss = {:4.2f}'.format(epoch+1,k+1,kld.item(),ell.item(),loss.item()))

#%%

batch_full = next(iter(test_loader))
batch = batch_full[0]
batch = batch.view(-1,6,batch.shape[-2],batch.shape[-1])
x = torch.cat([batch[:,i] for i in range(batch.shape[1])],-1)
x = x[:,:,:128]


recon_x, recon_s, mu_q, logvar_q, mu_p, logvar_p, mu0_q, logvar0_q = model(x)
loss = ell + kld

print(' Test ==> \t KLD = {:4.2f} \t ELL = {:4.2f} \t Loss = {:4.2f}'.format(epoch+1,k+1,kld.item(),ell.item(),loss.item()))

    
X = x.detach().numpy()
# X = np.abs(X[:,:dimx] + X[:,dimx:]*1j)

X_hat = recon_x.detach().numpy()
S_hat = recon_s.detach().numpy()
# X_hat = np.abs(X_hat[:,:dimx] + X_hat[:,dimx:]*1j)
kwds = dict(cmap='Greys',interpolation='none')
plt.figure(figsize=(6,10))
plt.subplot(411)
plt.imshow(X[0],**kwds)
plt.subplot(412)
plt.imshow(X_hat[0],**kwds)
plt.subplot(413)
plt.imshow(S_hat[0,0],**kwds)
# plt.subplot(414)
# plt.imshow(S_hat[0,1],**kwds)s

# %%
s, z, h = model.sample(N=100)

S = s[0,:].detach().numpy()
# S = np.abs(S[:,:dimx] + S[:,dimx:]*1j)

plt.figure(figsize=(6,4))
plt.imshow(S[0],**kwds)
plt.savefig('mnist_sampled.pdf')

 # %%
