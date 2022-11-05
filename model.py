#%%
import torch
from torch import nn
import math
from utils.stft import STFT, iSTFT
import numpy as np
from lossfun import FreqSignalLoss
from dataset import get_data_loaders
from torch.nn import functional as F

class DSAE(nn.Module):
    def __init__(self, dimx, dimz, dimh=64, dimg=64, chans=64, num_sources=1, net_type='pooled', num_layers=1, scale_invariant=False, layer_channels=None, averaged_loss=True):
        super(DSAE, self).__init__()
        self.dimx, self.dimz, self.dimh, self.dimg = dimx, dimz, dimh, dimg
        self.num_sources = num_sources
        self.num_layers = num_layers
        self.averaged_loss = averaged_loss
        
        self.gru_generate = nn.LSTM(self.dimz,self.dimh,num_layers=self.num_layers,batch_first=True)
        self.gru_infer = nn.LSTM(dimg,dimg,num_layers=self.num_layers,bidirectional=True,batch_first=True)

        self.make_transition_nets(dimg,chans,dimz,dimh)

        self.net_type = net_type
        
        if layer_channels is None:
            layer_channels = (64,128)
        self.Encoder = PooledEncoder(dimx,dimg*num_sources,layer_channels=layer_channels)
        self.Decoder = PooledDecoder(dimz,dimx,layer_channels=layer_channels)

        self.activation = nn.Softplus()
        self.activation = nn.Sigmoid()
        # self.activation = nn.Identity()

        self.likelihood = NormalLikelihood(scale_invariant=scale_invariant)
        self.criteria = nn.BCELoss(reduction='sum')
        self.register_parameter('variance',nn.Parameter(torch.ones(1)*.1,requires_grad=False))
        self.register_parameter('smu',nn.Parameter(torch.zeros(1,num_sources,self.dimz)))
        self.register_parameter('smu0',nn.Parameter(torch.zeros(1,num_sources,self.dimz)))
        self.register_buffer('slogvar',torch.zeros(1,num_sources,self.dimz))
        self.register_buffer('slogvar0',torch.zeros(1,num_sources,self.dimz))

        self.varmin = 1e-6
        
    def make_transition_nets(self,dimg,chans,dimz,dimh):
        self.mu_q = nn.Sequential(
            nn.Linear(dimg,chans),
            nn.LeakyReLU(),
            nn.Linear(chans,dimz),
            )
        self.logvar_q = nn.Sequential(
            nn.Linear(dimg,chans),
            nn.LeakyReLU(),
            nn.Linear(chans,dimz),
            )

        self.mu0_q = nn.Sequential(
            nn.Linear(dimg,chans),
            nn.LeakyReLU(),
            nn.Linear(chans,dimz),
            )
        self.logvar0_q = nn.Sequential(
            nn.Linear(dimg,chans),
            nn.LeakyReLU(),
            nn.Linear(chans,dimz),
            )

        self.mu_p = nn.Sequential(
            nn.Linear(dimh,chans),
            nn.LeakyReLU(),
            nn.Linear(chans,dimz),
            )
        self.logvar_p = nn.Sequential(
            nn.Linear(dimh,chans),
            nn.LeakyReLU(),
            nn.Linear(chans,dimz),
            )

        self.transition_z = nn.Sequential(
            nn.Linear(self.dimz,self.dimh),
            nn.LeakyReLU(),
            )

    def get_variance(self):
        # If going to learn this, then there needs to be gibbs sampler, with prior to prevent it from going to zero.
        # return F.softplus(100.0*self.variance)/100.0+self.varmin
        return self.variance*1.0

    def feature_encoder(self,x):
        gx = self.Encoder(x)
        return gx.view(-1,self.dimg,gx.size(-1)).transpose(-2,-1)

    def feature_decoder(self,z):
        return self.Decoder(z)

    def encode(self,x):
        gx = self.feature_encoder(x)
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
        x = self.feature_decoder(z)
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

    def loss_function(self,x,output):
        mu_x,mu,logvar,mu_prior,logvar_prior, mu0, logvar0 = output['recon_x'], output['mu_q'], output['logvar_q'],output['mu_p'], output['logvar_p'], output['mu0_q'], output['logvar0_q']
        n = x.size(0)
        N = n + 0
        if self.averaged_loss is True:
            N = torch.numel(x)
        N = float(N)
            
        # Si-SDR? scale the target... but it is the mixture.
        ell = self.likelihood(x, mu_x, var=self.get_variance()) 
        kld = KLD(mu,logvar,prior_mean=mu_prior,prior_logvar=logvar_prior) 
        prior_mean = self.batch_and_source(self.smu0,n)
        prior_logvar = self.batch_and_source(self.slogvar0,n)
        kld += KLD(mu0,logvar0,prior_mean=prior_mean,prior_logvar=prior_logvar)
        return ell / N , kld / N

    def forward(self, x):
        mu_q, logvar_q, mu0_q, logvar0_q = self.encode(x)
        z = reparameterize(mu_q,logvar_q)
        z0 = reparameterize(mu0_q,logvar0_q)
        mu_p, logvar_p = self.prior(z,z0)
        mu_s = self.decode(z)
        mu_s = mu_s.view(-1,self.num_sources,mu_s.size(-2),mu_s.size(-1))
        recon_s = mu_s
        recon_x = recon_s.sum(1)
        
        n = x.shape[0]
        out = {
            'recon_x': recon_x, 
            'recon_s': recon_s, 
            'mu_q': mu_q, 
            'logvar_q': logvar_q, 
            'mu_p': mu_p, 
            'logvar_p': logvar_p, 
            'mu0_q': mu0_q, 
            'logvar0_q': logvar0_q,
            'smu0': self.batch_and_source(self.smu0,n),
            'slogvar0':  self.batch_and_source(self.slogvar0,n),
            'variance': self.get_variance(),
            }
        return out


class DCAE(DSAE):
    def __init__(self, dimx, dimz, dimh=64, dimg=64, chans=64, num_sources=1, net_type='pooled', num_layers=1, scale_invariant=False, layer_channels=None,kernel_size=(3,2),stride=(2,1),padding=(1,0)):
        super(DCAE, self).__init__(dimx=dimx,dimz=dimz)

        if layer_channels is None:
            layer_channels = (64,128)

        self.conv_channels = layer_channels[-1]

        n_feature_layers = len(layer_channels)
        self.freq_features = dimx // (2**(n_feature_layers+1)) + (dimx % 2) 
        self.encoded_features = self.freq_features * self.conv_channels
        self.n_feat_encoded = dimg * num_sources
        # dimh = dimg
        # dimz = dimh
        # chans = dimg
        chans = (dimz + dimg) // 2
        dimx = 1
        self.dimx, self.dimz, self.dimh, self.dimg = dimx, dimz, dimh, dimg
        self.num_sources = num_sources
        self.num_layers = num_layers
        
        self.gru_generate = nn.LSTM(self.dimz,self.dimh,num_layers=self.num_layers,batch_first=True)
        self.gru_infer = nn.LSTM(dimg,dimg,num_layers=self.num_layers,bidirectional=True,batch_first=True)
        
        self.gru_feature = nn.LSTM(self.n_feat_encoded,self.n_feat_encoded,num_layers=self.num_layers,bidirectional=False,batch_first=True)

        self.make_transition_nets(dimg,chans,dimz,dimh)
        self.fc_encoder = nn.Sequential(
            nn.Conv1d(self.encoded_features,self.dimg*self.num_sources,kernel_size=1,stride=1,padding=0),
            nn.LeakyReLU(),
        )
        self.fc_decoder = nn.Sequential(
            nn.ConvTranspose1d(self.dimz,self.encoded_features,kernel_size=1,stride=1,padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.encoded_features),
        )
        
        kwds_feature_conv = dict(kernel_size=kernel_size,stride=stride,padding=padding)
        self.Encoder = StridedEncoder(1,self.conv_channels,layer_channels=layer_channels,**kwds_feature_conv)
        self.Decoder = StridedDecoder(self.conv_channels,1,layer_channels=layer_channels,**kwds_feature_conv)

        self.activation = nn.Sigmoid()

        self.likelihood = NormalLikelihood(scale_invariant=scale_invariant)
        self.criteria = nn.BCELoss(reduction='sum')
        self.register_parameter('variance',nn.Parameter(torch.ones(1)*.1,requires_grad=False))
        self.register_parameter('smu',nn.Parameter(torch.zeros(1,num_sources,self.dimz)))
        self.register_parameter('smu0',nn.Parameter(torch.zeros(1,num_sources,self.dimz)))
        self.register_buffer('slogvar',torch.zeros(1,num_sources,self.dimz))
        self.register_buffer('slogvar0',torch.zeros(1,num_sources,self.dimz))

        self.varmin = 1e-6
        
    def feature_encoder(self,x):
        if x.ndim < 4:
            x = x.unsqueeze(1)
        gx = self.Encoder(x)
        gx = gx.view(-1,self.encoded_features,gx.shape[-1])
        gx = self.fc_encoder(gx)
        # gx = gx.view(-1,self.n_feat_encoded,gx.size(-1)).transpose(-2,-1)
        # g0 = torch.zeros(self.num_layers,gx.size(0),self.n_feat_encoded,device=x.device)
        # c0 = torch.zeros(self.num_layers,gx.size(0),self.n_feat_encoded,device=x.device)
        # gx,_ = self.gru_feature(gx,(g0,c0))
        # gx = gx.transpose(-2,-1)
        gx = gx.view(-1,self.dimg,gx.shape[-1]).transpose(-2,-1)
        return gx

    def feature_decoder(self,z):
        z = self.fc_decoder(z)
        z = z.view(z.shape[0],self.conv_channels,-1,z.shape[-1])
        y = self.Decoder(z)
        return y

'''
Encoder/Decoders for Features.
1) Strided
2) Pooled
'''

class Conv2dLayer(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size=(3,2),stride=(2,1),padding=(1,0),batchnorm=True,activation=True) -> None:
        super(Conv2dLayer,self).__init__()

        mods = nn.ModuleList()
        mods.append(nn.Conv2d(input_channels,output_channels,kernel_size=kernel_size,stride=stride,padding=padding))
        if activation:
            mods.append(nn.LeakyReLU())
        if batchnorm:
            mods.append(nn.BatchNorm2d(output_channels))
        self.net = nn.Sequential(*mods)

    def forward(self,x):
        return self.net(x)

class ConvTranspose2dLayer(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size=(3,2),stride=(2,1),padding=(1,0),batchnorm=True,activation=True) -> None:
        super(ConvTranspose2dLayer,self).__init__()

        mods = nn.ModuleList()
        mods.append(nn.ConvTranspose2d(input_channels,output_channels,kernel_size=kernel_size,stride=stride,padding=padding))
        if batchnorm:
            mods.append(nn.BatchNorm2d(output_channels))
        if activation:
            mods.append(nn.LeakyReLU())

        self.net = nn.Sequential(*mods)

    def forward(self,x):
        return self.net(x)

class StridedEncoder(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size=(3,2),stride=(2,1),padding=(1,0),layer_channels=None):
        super(StridedEncoder, self).__init__()
        if layer_channels is None: 
            layer_channels = (32,64,128)

        kwds = dict(kernel_size=kernel_size,stride=stride,padding=padding)

        mlist = nn.ModuleList()
        mlist.append(Conv2dLayer(input_channels,layer_channels[0],**kwds))
        for ii in range(len(layer_channels)-1):
            mlist.append(Conv2dLayer(layer_channels[ii],layer_channels[ii+1],**kwds))
        mlist.append(Conv2dLayer(layer_channels[-1],output_channels,batchnorm=True,**kwds))

        self.Network = nn.Sequential(*mlist)

    def forward(self, x):
        return self.Network(x)

class StridedDecoder(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size=(3,2),stride=(2,1),padding=(1,0),layer_channels=None):
        super(StridedDecoder, self).__init__()

        if layer_channels is None: 
            layer_channels = (32,64,128)

        kwds = dict(kernel_size=kernel_size,stride=stride,padding=padding)

        mlist = nn.ModuleList()
        mlist.append(ConvTranspose2dLayer(input_channels,layer_channels[-1],**kwds))
        for ii in range(len(layer_channels)-2,-1,-1):
            mlist.append(ConvTranspose2dLayer(layer_channels[ii+1],layer_channels[ii],**kwds))
        mlist.append(ConvTranspose2dLayer(layer_channels[0],output_channels,batchnorm=False,**kwds))

        self.Network = nn.Sequential(*mlist)
    def forward(self, x):
        return self.Network(x)

# Pooled
class PooledEncoder(nn.Module):
    def __init__(self,input_channels,output_channels,layer_channels=None):
        super(PooledEncoder, self).__init__()

        if layer_channels is None: 
            layer_channels = (64,128)
           
        # pooling = nn.MaxPool1d(kernel_size=2,stride=2,padding=0)
        pooling = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2),padding=(0,0))
        mlist = nn.ModuleList()
        mlist.append(nn.Sequential(
            nn.Conv1d(input_channels,layer_channels[0],kernel_size=3,stride=1,padding=1), pooling, nn.LeakyReLU(), nn.BatchNorm1d(layer_channels[0])))
        for ii in range(len(layer_channels)-1):
            mlist.append(nn.Sequential(
                nn.Conv1d(layer_channels[ii],layer_channels[ii+1],kernel_size=3,stride=1,padding=1), pooling, nn.LeakyReLU(), nn.BatchNorm1d(layer_channels[ii+1])))
        mlist.append(nn.Sequential(
            nn.Conv1d(layer_channels[-1],output_channels,kernel_size=3,stride=1,padding=1), nn.LeakyReLU()))
        
        self.Network = nn.Sequential(*mlist)
        
    def forward(self, x):
        return self.Network(x)

class PooledDecoder(nn.Module):
    def __init__(self,input_channels,output_channels,layer_channels=None):
        super(PooledDecoder, self).__init__()
        
        if layer_channels is None: 
            layer_channels = (64,128)
            
        upsample = nn.Upsample(scale_factor=2,mode='linear',align_corners=False)
            
        mlist = nn.ModuleList()
        mlist.append(nn.Sequential(
            nn.Conv1d(input_channels,layer_channels[-1],kernel_size=3,stride=1,padding=1), nn.LeakyReLU(), nn.BatchNorm1d(layer_channels[-1]), upsample))
        for ii in range(len(layer_channels)-2,-1,-1):
            mlist.append(nn.Sequential(
                nn.Conv1d(layer_channels[ii+1],layer_channels[ii],kernel_size=3,stride=1,padding=1), nn.LeakyReLU(), nn.BatchNorm1d(layer_channels[ii]), upsample))
        mlist.append(nn.Sequential(
            nn.Conv1d(layer_channels[0],output_channels,kernel_size=3,stride=1,padding=1)))
        
        self.Network = nn.Sequential(*mlist)
        
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

        alpha = torch.ones_like(mean)[:,0,0].view(-1,1,1)
        mu = mean.clone().detach()

        if self.scale_invariant:
            for i in range(x.shape[0]):
                d = (mu[i] * mu[i]).sum()
                if d == 0.0: d += 1.0
                alpha[i] = (x[i] * mu[i]).sum() / d
                
        eps = x - mean*alpha

        return 0.5*torch.sum(var.log() + (eps).pow(2) / var)

class LossFunction(nn.Module):
    def __init__(self,averaged=True,scale_invariant=True):
        super(LossFunction,self).__init__()
        
        self.averaged = averaged        
        self.likelihood = NormalLikelihood(scale_invariant=scale_invariant)
        
    def forward(self,x,output):
        mu_x,mu,logvar,mu_prior,logvar_prior, mu0, logvar0 = output['recon_x'], output['mu_q'], output['logvar_q'],output['mu_p'], output['logvar_p'], output['mu0_q'], output['logvar0_q']
        smu0, slogvar0, variance = output['smu0'], output['slogvar0'], output['variance']
        
        n = x.size(0)
        N = n + 0
        if self.averaged is True:
            N = torch.numel(x)
        N = float(N)
            
        # Si-SDR? scale the target... but it is the mixture.
        ell = self.likelihood(x, mu_x, var=variance) 
        kld = KLD(mu,logvar,prior_mean=mu_prior,prior_logvar=logvar_prior) 
        prior_mean = smu0
        prior_logvar = slogvar0
        kld += KLD(mu0,logvar0,prior_mean=prior_mean,prior_logvar=prior_logvar)
        
        return ell / N , kld / N
    
def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

def output_size(H_in, kernel_size, stride=1,  dilation=1, padding=0):
    H_out = H_in + 2*padding - dilation * (kernel_size - 1) - 1
    return H_out // stride + 1

def make_invariant(x,y):
    return  ((x * y).sum() + 1e-9) / ((x * x).sum() + 1e-9)



#%%
if __name__ == "__main__":
    enc = StridedEncoder(input_channels=1,output_channels=32)
    dec = StridedDecoder(input_channels=32,output_channels=1)
    
    dimx = 257
    x = torch.randn(4,1,dimx,129)
    z = enc(x)
    y = dec(z)

    model = DCAE(dimx=dimx,dimz=32,kernel_size=(3,2),stride=(2,1),padding=(1,0),layer_channels=(32,64,64,64,64))
    output = model(x)
    
    x_hat = output['recon_x']

    print(x.shape)
    print(x_hat.shape)

# %%
