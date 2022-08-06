#%%
import torch
from torch import nn
import math
from utils.stft import STFT, iSTFT
import numpy as np

class DSAE(nn.Module):
    def __init__(self, dimx, dimz, dimh=64, dimg=64, chans=64, num_sources=1, net_type='pooled', num_layers=1):
        super(DSAE, self).__init__()
        self.dimx, self.dimz, self.dimh, self.dimg = dimx, dimz, dimh, dimg
        self.num_sources = num_sources
        self.num_layers = num_layers

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
        
        if net_type == 'pooled':
            self.Encoder = PooledEncoder(dimx,dimg*num_sources)
            self.Decoder = PooledDecoder(dimz,dimx)
        else: 
            self.Encoder = StridedEncoder(dimx,64)
            self.Decoder = StridedDecoder(64,dimx)

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
        z = z.transpose(-2,-1)
        z = z.view(-1,64,self.dimz//64,z.shape[-1])
        x = self.Decoder(z)
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
        recon_s = mu_s
        recon_x = recon_s
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
            nn.Conv2d(input_channels,32,kernel_size=kernel_size,stride=stride,padding=padding), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32,64,kernel_size=kernel_size,stride=stride,padding=padding), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64,64,kernel_size=kernel_size,stride=stride,padding=padding), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64,64,kernel_size=kernel_size,stride=stride,padding=padding), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64,output_channels,kernel_size=kernel_size,stride=stride,padding=padding), nn.ReLU(),
            )
    def forward(self, x):
        return self.Network(x)

class StridedDecoder(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size=(3,2),stride=(2,1),padding=(1,0)):
        super(StridedDecoder, self).__init__()
        self.Network = nn.Sequential(
            nn.ConvTranspose2d(input_channels,64,kernel_size=kernel_size,stride=stride,padding=padding), nn.ReLU(), nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64,64,kernel_size=kernel_size,stride=stride,padding=padding), nn.ReLU(), nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64,64,kernel_size=kernel_size,stride=stride,padding=padding), nn.ReLU(), nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64,32,kernel_size=kernel_size,stride=stride,padding=padding), nn.ReLU(), nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32,output_channels,kernel_size=kernel_size,stride=stride,padding=padding), 
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
            nn.Conv1d(256,512,kernel_size=3,stride=1,padding=1), pooling, nn.ReLU(), nn.BatchNorm1d(512),
            nn.Conv1d(512,output_channels,kernel_size=3,stride=1,padding=1), nn.ReLU(),
            )
    def forward(self, x):
        return self.Network(x)

class PooledDecoder(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(PooledDecoder, self).__init__()
        upsample = nn.Upsample(scale_factor=2,mode='linear',align_corners=False)
        # upsample = SincInterpolation(scale_factor=2)
        self.Network = nn.Sequential(
            nn.ConvTranspose1d(input_channels,512,kernel_size=3,stride=1,padding=1), nn.ReLU(), nn.BatchNorm1d(512), upsample,
            nn.ConvTranspose1d(512,256,kernel_size=3,stride=1,padding=1), nn.ReLU(), nn.BatchNorm1d(256), upsample,
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

def output_size(H_in, kernel_size, stride=1,  dilation=1, padding=0):
    H_out = H_in + 2*padding - dilation * (kernel_size - 1) - 1
    return H_out // stride + 1

class DCVAE(DSAE):
    def __init__(self, fs, nfft, dimz, dimh=64, dimg=64, chans=64, num_sources=1, net_type='pooled', num_layers=1):
        dimx = nfft//2+1
        H_in = dimx
        for i in range(5):
            H_in = output_size(H_in,kernel_size=3,stride=2,padding=1)
        dimg = 64*H_in
        dimz = dimg 
        super().__init__(2, dimz, dimh, dimg, chans, num_sources, net_type, num_layers)
        self.stft = STFT(fs=fs,nfft=nfft,winlen=.02,hopfrac=.5)
        self.istft = iSTFT(fs=fs,nfft=nfft,winlen=.02,hopfrac=.5)

    def forward(self,x):
        X = self.stft(x)
        X = X.transpose(1,-1)
        recon_X, recon_S, mu_q, logvar_q, mu_p, logvar_p, mu0_q, logvar0_q = super().forward(X)
        recon_X = recon_X.transpose(1,-1)
        recon_x = self.istft(recon_X,self.stft.padder)
        return recon_x,mu_q, logvar_q, mu_p, logvar_p, mu0_q, logvar0_q

# %%

#%%
from torch import nn, optim

model = DCVAE(fs=16e3,nfft=320,dimz=64,net_type='strided')
print(model.dimg)

#%%
optimizer = optim.Adam(params=model.parameters(),lr=1e-3)

lossfun = nn.MSELoss()

model.train()

#%%
for batch in range(1000):

    optimizer.zero_grad()

    x = torch.cos(2*np.pi*torch.arange(8e3).view(1,-1)/16e3*torch.rand(16,1)*5e2 + torch.randn(16,1)*4)

    y, mu_q, logvar_q, mu_p, logvar_p, mu0_q, logvar0_q = model(x)

    loss = lossfun(input=y,target=x)

    ell, kld = model.loss_function(x,y, mu_q, logvar_q, mu_p, logvar_p, mu0_q, logvar0_q)
    loss = ell + kld

    loss.backward()

    optimizer.step()

    if batch % 10 == 0:
        print("loss {:4.2f} \t ell {:4.2f} \t kld {:4.2f}".format(loss.item(),ell.item(),kld.item()))

# %%
import matplotlib.pyplot as plt
plt.plot(x[0])
plt.plot(y[0].detach().numpy())
plt.xlim(0,512)
