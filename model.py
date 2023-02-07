#%%
import torch
from torch import nn
import math
from utils.stft import STFT, iSTFT, get_win, biorwin
import numpy as np
from lossfun import FreqSignalLoss
from dataset import get_data_loaders, AudioPipeline
from torch.nn import functional as F
import torchaudio

class DSAE(nn.Module):
    def __init__(self, dimx, dimz, dimh=64, dimg=64, chans=64, num_sources=1, net_type='pooled', num_layers=1, scale_invariant=False, layer_channels=None, averaged_loss=True,rnn_type='lstm',inference_rnn=True):
        super(DSAE, self).__init__()
        self.dimx, self.dimz, self.dimh, self.dimg = dimx, dimz, dimh, dimg
        self.num_sources = num_sources
        self.num_layers = num_layers
        self.averaged_loss = averaged_loss
        
        self.rnn_type = rnn_type
        self.inference_rnn = inference_rnn

        if rnn_type == 'gru':  
            self.gru_generate = nn.GRU(self.dimz,self.dimh,num_layers=self.num_layers,batch_first=True)
            if inference_rnn:
                self.gru_infer = nn.GRU(dimg,dimg,num_layers=self.num_layers,bidirectional=True,batch_first=True)
        else:
            self.gru_generate = nn.LSTM(self.dimz,self.dimh,num_layers=self.num_layers,batch_first=True)
            if inference_rnn:
                self.gru_infer = nn.LSTM(dimg,dimg,num_layers=self.num_layers,bidirectional=True,batch_first=True)

        if inference_rnn is False:
            dimg = dimz*2

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
        if self.inference_rnn:
            g0 = torch.zeros(2*self.num_layers,gx.size(0),self.dimg,device=x.device)
            if self.rnn_type == 'gru':
                g,_ = self.gru_infer(gx,g0)
            else:
                c0 = torch.zeros(2*self.num_layers,gx.size(0),self.dimg,device=x.device)
                g,_ = self.gru_infer(gx,(g0,c0))
            g = 1/2*(g[:,:,:self.dimg] + g[:,:,self.dimg:])
        else:
            g = gx
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
        zt = torch.cat([z0,z[:,:-1]],1)

        h0 = torch.zeros(self.num_layers,z.size(0),self.dimh,device=z.device)
        if self.rnn_type == 'gru':
            h,_ = self.gru_generate(zt,h0)
        else:
            c0 = torch.zeros(self.num_layers,z.size(0),self.dimh,device=z.device)
            h,_ = self.gru_generate(zt,(h0,c0))
        
        zh = self.combiner(zt,h)
        mu,logvar = self.prior_stats(zh)
        return mu, logvar
    
    def sample_latent(self,N,M,device):
        MK = M * self.num_sources
        z = torch.zeros(MK,N,self.dimz).to(device)
        h = torch.zeros(MK,N,self.dimh).to(device)
        h_t = torch.zeros(self.num_layers,MK,self.dimh).to(device)
        c_t = torch.zeros(self.num_layers,MK,self.dimh).to(device)
        prior_mean = self.batch_and_source(self.smu0,M).to(device)
        prior_logvar = self.batch_and_source(self.slogvar0,M).to(device)
        z_t = torch.randn(MK,1,self.dimz).to(device)*torch.exp(0.5*prior_logvar) + prior_mean
        for n in range(N):
            if self.rnn_type == 'gru':
                _,h_t = self.gru_generate(z_t,h_t)
            else:
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

    def forward(self, x, stochastic=True):
        mu_q, logvar_q, mu0_q, logvar0_q = self.encode(x)
        if stochastic is True:
            z = reparameterize(mu_q,logvar_q)
            z0 = reparameterize(mu0_q,logvar0_q)
        else:
            z = mu_q.clone()
            z0 = mu0_q.clone()
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
            'z': z,
            }
        return out


class DCAE(DSAE):
    def __init__(self, dimx, dimz, dimh=64, dimg=64, chans=64, num_sources=1, net_type='pooled', num_layers=1, scale_invariant=False, layer_channels=None,kernel_size=(3,2),stride=(2,1),padding=(1,0),dilation=(1,1),variance=.1,activate_final_layer=True,kernel_size_separator=1,rnn_type='lstm',inference_rnn=True):
        super(DCAE, self).__init__(dimx=dimx,dimz=dimz,rnn_type=rnn_type,inference_rnn=inference_rnn)

        if inference_rnn is False:
            dimg = dimz*2

        if layer_channels is None:
            layer_channels = (64,128)

        self.conv_channels = layer_channels[-1]

        n_feature_layers = len(layer_channels)
        self.freq_features = dimx // (2**(n_feature_layers)) + (dimx % 2) 
        self.encoded_features = self.freq_features * self.conv_channels
        self.n_feat_encoded = dimg * num_sources

        chans = (dimz + dimg) // 2
        dimx = 1
        self.dimx, self.dimz, self.dimh, self.dimg = dimx, dimz, dimh, dimg
        self.num_sources = num_sources
        self.num_layers = num_layers
        

        self.rnn_type = rnn_type
        if rnn_type == 'gru':  
            self.gru_generate = nn.GRU(self.dimz,self.dimh,num_layers=self.num_layers,batch_first=True)
            if inference_rnn:
                self.gru_infer = nn.GRU(dimg,dimg,num_layers=self.num_layers,bidirectional=True,batch_first=True)
        else:
            self.gru_generate = nn.LSTM(self.dimz,self.dimh,num_layers=self.num_layers,batch_first=True)
            if inference_rnn:
                self.gru_infer = nn.LSTM(dimg,dimg,num_layers=self.num_layers,bidirectional=True,batch_first=True)
        
        
        self.make_transition_nets(dimg,chans,dimz,dimh)
        self.fc_encoder = nn.Sequential(
            nn.Conv1d(self.encoded_features,self.dimg*self.num_sources,kernel_size=kernel_size_separator,stride=1,padding=kernel_size_separator//2),
            nn.LeakyReLU(),
        )
        self.fc_decoder = nn.Sequential(
            nn.ConvTranspose1d(self.dimz,self.encoded_features,kernel_size=1,stride=1,padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.encoded_features),
        )
        
        kwds_feature_conv = dict(kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        self.Encoder = StridedEncoder(1,self.conv_channels,layer_channels=layer_channels,**kwds_feature_conv)
        self.Decoder = StridedDecoder(self.conv_channels,1,layer_channels=layer_channels,activate_final_layer=activate_final_layer,**kwds_feature_conv)

        self.activation = nn.Sigmoid()

        self.likelihood = NormalLikelihood(scale_invariant=scale_invariant)
        self.criteria = nn.BCELoss(reduction='sum')
        self.register_parameter('variance',nn.Parameter(torch.ones(1)*variance,requires_grad=False))
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
        gx = gx.view(-1,self.dimg,gx.shape[-1]).transpose(-2,-1)
        return gx

    def feature_decoder(self,z):
        z = self.fc_decoder(z)
        z = z.view(z.shape[0],self.conv_channels,-1,z.shape[-1])
        y = self.Decoder(z)
        return y


class LinearBlock(nn.Module):
    def __init__(self, in_channels,out_channels,activation=True):
        super(LinearBlock, self).__init__()
        if activation is True:
            self.block = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                )
        else:
            self.block = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                )

    def forward(self, x):
        return self.block(x)

class VAE(nn.Module):
    def __init__(self, dimx, dimz, n_sources=1,variational=True):
        super(VAE, self).__init__()

        self.dimx = dimx
        self.dimz = dimz
        self.n_sources = n_sources
        self.variational = variational

        chans = (2560, 2048, 1536, 1024, 512)

        self.out_z = nn.Linear(chans[-1],2*self.n_sources*self.dimz)

        self.Encoder = nn.Sequential(
            LinearBlock(self.dimx, chans[0]),
            LinearBlock(chans[0],chans[1]),
            LinearBlock(chans[1],chans[2]),
            LinearBlock(chans[2],chans[3]),
            LinearBlock(chans[3],chans[4]),
            )

        self.Decoder = nn.Sequential(
            LinearBlock(self.dimz, chans[4]),
            LinearBlock(chans[4],chans[3]),
            LinearBlock(chans[3],chans[2]),
            LinearBlock(chans[2],chans[1]),
            LinearBlock(chans[1],chans[0]),
            LinearBlock(chans[0],self.dimx,activation=False),
            )

        self.register_parameter('variance',nn.Parameter(torch.ones(1)*.1,requires_grad=False))
        self.register_parameter('m0',nn.Parameter(torch.zeros(1,n_sources*self.dimz)))
        
    def get_variance(self):
        return self.variance*1.0

    def encode(self, x):
        d = self.Encoder(x)
        dz = self.out_z(d)
        mu = dz[:,::2]
        logvar = dz[:,1::2]
        return mu,logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        d = self.Decoder(z.view(-1,self.dimz))
        recon_separate = torch.sigmoid(d).view(-1,self.n_sources,self.dimx)
        recon_x = recon_separate.sum(1) 
        return recon_x, recon_separate

    def forward(self, x):
        xshape = x.shape
        mu, logvar = self.encode(x.reshape(-1, self.dimx))
        if self.variational is True:
            z = self.reparameterize(mu, logvar)
            recon_x, recons = self.decode(z)
        else:
            recon_x, recons = self.decode(mu)

        out = {
            'recon_x': recon_x.view(-1,xshape[-2],xshape[-1]), 
            'recon_s': recons.view(-1,recons.shape[1],xshape[-2],xshape[-1]), 
            'mu_q': mu, 
            'logvar_q': logvar,
            'mu_p': self.m0, 
            'logvar_p': torch.zeros_like(logvar), 
            'mu0_q': torch.zeros(1,device=mu.device), 
            'logvar0_q': torch.zeros(1,device=mu.device), 
            'smu0': torch.zeros(1,device=mu.device), 
            'slogvar0':  torch.zeros(1,device=mu.device), 
            'variance': self.get_variance(),
            }
        return out

'''
Encoder/Decoders for Features.
1) Strided
2) Pooled
'''

class Conv2dLayer(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size=(3,2),stride=(2,1),padding=(1,0),dilation=(1,1),batchnorm=True,activation=True,bias=True) -> None:
        super(Conv2dLayer,self).__init__()

        mods = nn.ModuleList()
        mods.append(nn.Conv2d(input_channels,output_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=bias))
        if activation:
            mods.append(nn.LeakyReLU())
        if batchnorm:
            mods.append(nn.BatchNorm2d(output_channels))
        self.net = nn.Sequential(*mods)

    def forward(self,x):
        return self.net(x)

class ConvTranspose2dLayer(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size=(3,2),stride=(2,1),padding=(1,0),dilation=(1,1),batchnorm=True,activation=True) -> None:
        super(ConvTranspose2dLayer,self).__init__()

        mods = nn.ModuleList()
        mods.append(nn.ConvTranspose2d(input_channels,output_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation))
        if batchnorm:
            mods.append(nn.BatchNorm2d(output_channels))
        if activation:
            mods.append(nn.LeakyReLU())

        self.net = nn.Sequential(*mods)

    def forward(self,x):
        return self.net(x)

class StridedEncoder(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size=(3,2),stride=(2,1),padding=(1,0),dilation=(1,1),layer_channels=None):
        super(StridedEncoder, self).__init__()
        if layer_channels is None: 
            layer_channels = (32,64,128)

        kwds = dict(kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)

        mlist = nn.ModuleList()
        mlist.append(Conv2dLayer(input_channels,layer_channels[0],**kwds))
        for ii in range(1,len(layer_channels)-1):
            mlist.append(Conv2dLayer(layer_channels[ii-1],layer_channels[ii],**kwds))
        mlist.append(Conv2dLayer(layer_channels[-1],output_channels,batchnorm=True,**kwds))

        self.Network = nn.Sequential(*mlist)

    def forward(self, x):
        return self.Network(x)

class StridedDecoder(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size=(3,2),stride=(2,1),padding=(1,0),dilation=(1,1),layer_channels=None,activate_final_layer=False):
        super(StridedDecoder, self).__init__()

        if layer_channels is None: 
            layer_channels = (32,64,128)

        kwds = dict(kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)

        mlist = nn.ModuleList()
        mlist.append(ConvTranspose2dLayer(input_channels,layer_channels[-1],**kwds))
        for ii in range(len(layer_channels)-3,-1,-1):
            mlist.append(ConvTranspose2dLayer(layer_channels[ii+1],layer_channels[ii],**kwds))

        mlist.append(ConvTranspose2dLayer(layer_channels[0],output_channels,activation=activate_final_layer,batchnorm=False,**kwds))

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
    def __init__(self, variance=None, reduction='sum', scale_invariant=False,complex=False):
        super(NormalLikelihood, self).__init__()
        if variance is None:
            variance = torch.tensor(1.0)
        self.scale_invariant = scale_invariant
        self.register_buffer("variance",variance)
        self.complex = complex

    def get_eps(self,x,mean):
        n = mean.shape[0]
        alpha = torch.ones(n,*([1]*(x.ndim-1)),device=mean.device)
        
        mu = mean.clone().detach()
        
        if self.scale_invariant:
            for i in range(n):
                d = (mu[i] * mu[i]).sum()
                if d == 0.0: 
                    d += 1.0
                alpha[i] = (x[i] * mu[i]).sum() / d
                
        eps = x - mean*alpha
        return eps

    def get_eps_complex(self,x,mean):
        n = mean.shape[0]
        x = torch.complex(x[:,0],x[:,1])
        mean = torch.complex(mean[:,0],mean[:,1])
        mu = mean.clone().detach()
        alpha = torch.ones(n,*([1]*(x.ndim-1)),device=mean.device,dtype=x.dtype)
        
        if self.scale_invariant:
            d = (mu.conj() * mu).view(n,-1).sum(-1)
            d[d==0] = 1.0
            alpha = (mu.conj() * x).view(n,-1).sum(-1) / d
            alpha = alpha.view(n,*([1]*(x.ndim-1)))
                
        eps = x - alpha*mean
        eps = torch.stack([eps.real,eps.imag],1)
        return eps

    def forward(self, x, mean, var=None):
        if var is None:
            var = self.variance

        if self.complex:
            eps = self.get_eps_complex(x,mean)
        else:
            eps = self.get_eps(x,mean)

        return 0.5*torch.sum(var.log() + (eps).pow(2) / var)

class LaplaceLikelihood(nn.Module):
    def __init__(self, variance=None, reduction='sum', scale_invariant=False):
        super(LaplaceLikelihood, self).__init__()
        if variance is None:
            variance = torch.tensor(1.0)
        self.scale_invariant = scale_invariant
        self.register_buffer("variance",variance)

        self.register_buffer("log2",torch.tensor(2.0).log())

        self.l1 = nn.L1Loss(reduction='none')


    def forward(self, x, mean, var=None):
        if var is None:
            var = self.variance
        scale = torch.sqrt(0.5*var)
        
        n = mean.shape[0]
        alpha = torch.ones(n,*([1]*(x.ndim-1)),device=mean.device)
        mu = mean.clone().detach()
        
        if self.scale_invariant:
            for i in range(n):
                d = (mu[i] * mu[i]).sum()
                if d == 0.0: 
                    d += 1.0
                alpha[i] = (x[i] * mu[i]).sum() / d
                
        l1_result = self.l1(mean*alpha,x)

        return torch.sum(self.log2 + scale.log() + l1_result / scale)


class LossFunction(nn.Module):
    def __init__(self,averaged=True,scale_invariant=True,complex_version=False,laplace=False):
        super(LossFunction,self).__init__()

        self.averaged = averaged  
        self.complex_version = complex_version

        if laplace:
            self.likelihood = LaplaceLikelihood(scale_invariant=scale_invariant)
        else:
            self.likelihood = NormalLikelihood(scale_invariant=scale_invariant,complex=self.complex_version)
        
    def forward(self,x,output):
        mu_x,mu,logvar,mu_prior,logvar_prior, mu0, logvar0 = output['recon_x'], output['mu_q'], output['logvar_q'],output['mu_p'], output['logvar_p'], output['mu0_q'], output['logvar0_q']
        smu0, slogvar0, variance = output['smu0'], output['slogvar0'], output['variance']
        
        n = x.size(0)
        N = n + 0
        if self.averaged is True:
            N = torch.numel(x)
        N = float(N)

        # if self.complex_version:
        #     if x.ndim == 4:
        #         x = (x**2.0).sum(1).sqrt()
            
        # Si-SDR? scale the target... but it is the mixture.
        ell = self.likelihood(x, mu_x, var=variance) 
        kld = KLD(mu,logvar,prior_mean=mu_prior,prior_logvar=logvar_prior)
        kld += KLD(mu0,logvar0,prior_mean=smu0,prior_logvar=slogvar0)
        
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

def create_masks(x,s_hat):
    return s_hat * (x /  s_hat.sum(1)).unsqueeze(1) 



class SVAE(DCAE):
    """
    Complex-valued input and output (2D input for Real and Imag or Mag and Phase)
    """
    def __init__(self, dimx, dimz, dimh=64, dimg=64, chans=64, num_sources=1, net_type='pooled', num_layers=1, scale_invariant=True, layer_channels=None, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), variance=0.1, activate_final_layer=False, mag_output=True):
        super().__init__(dimx, dimz, dimh, dimg, chans, num_sources, net_type, num_layers, scale_invariant, layer_channels, kernel_size, stride, padding, variance, activate_final_layer)

        kwds_feature_conv = dict(kernel_size=kernel_size,stride=stride,padding=padding)
        self.Encoder = StridedEncoder(2,self.conv_channels,layer_channels=layer_channels,**kwds_feature_conv)
        self.Decoder = StridedDecoder(self.conv_channels,2,layer_channels=layer_channels,activate_final_layer=activate_final_layer,**kwds_feature_conv)

        self.mag_output = mag_output
        self.phase_activation = nn.Tanh()

    def decode(self,z):        
        z = z.transpose(-2,-1)
        x = self.feature_decoder(z) # B*S X 2 X F X C
        mag = self.activation(x[:,0])
        phase = np.pi*self.phase_activation(x[:,1])

        
        if self.mag_output:
            return mag
        else:
            xr = mag * torch.cos(phase)
            xi = mag * torch.sin(phase)
            x = torch.stack([xr,xi],1)
            return x

    def forward(self, x):
        mu_q, logvar_q, mu0_q, logvar0_q = self.encode(x)
        z = reparameterize(mu_q,logvar_q)
        z0 = reparameterize(mu0_q,logvar0_q)
        mu_p, logvar_p = self.prior(z,z0)
        mu_s = self.decode(z)

        if self.mag_output:
            recon_s = mu_s.view(-1,self.num_sources,mu_s.size(-2),mu_s.size(-1))
            recon_x = recon_s.sum(1)
        else:
            recon_s = mu_s.view(-1,self.num_sources,2,mu_s.size(-2),mu_s.size(-1))
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

class Padder(nn.Module):
    """non-trainable padder module"""

    def __init__(self, win_length=512,hop_length=128,centered=True):
        super(Padder, self).__init__()
        """
        hop_length    hop length
        win_length      window length
        """
        self.hop_len = hop_length
        self.win_length = win_length
        self.centered = centered

    def pad(self,x):
        self.siglen = x.shape[-1]

        self.padding = np.zeros(2).astype(int)
        self.padding[0] = self.win_length-self.hop_len

        M = np.ceil((self.win_length+self.siglen - 2.0*self.hop_len-1)/self.hop_len + 1).astype(int)
        self.n_times = np.power(2,np.ceil(np.log2(M-1)))+1
        self.n_times = np.ceil(M/64)*64 + 1
        self.padding[1] = ((self.n_times-1)*self.hop_len + self.win_length) - (self.padding[0] + self.siglen)

        if self.centered == True:
            pad_total = self.padding.sum()
            self.padding[0] = pad_total // 2
            self.padding[1] = pad_total - self.padding[0]

        xpad = F.pad(x,tuple(self.padding))
        self.L = xpad.shape[-1]
        return xpad

    def unpad(self,x):
        return x.transpose(0,-1)[self.padding[0]:-self.padding[1]].transpose(0,-1)

    def forward(self, x, remove_pad=False):
        if remove_pad is True:
            return self.unpad(x)
        else:
            return self.pad(x)


class Separator(nn.Module):
    def __init__(self,model:'DCAE',compression=.5,n_fft=512,hop_length=128,mask=True) -> None:
        super(Separator,self).__init__()

        self.win_length = n_fft
        self.hop_length = hop_length
        self.n_fft = n_fft

        self.padder = Padder(win_length=self.win_length,hop_length=self.hop_length,centered=True)
        self.model = model
        self.stochastic = False

        self.compression = compression

        self.win = torch.hann_window(self.win_length) 
        self.normalizer = self.win.pow(2.0).sum().sqrt()

        self.mask = mask

        
    def preprocess(self,input:torch.Tensor):
        # pad
        input = self.padder(input)
        # stft
        self.n_times = self.padder.n_times
        slices = torch.arange(self.win_length).view(-1,1) + torch.arange(self.n_times).view(1,-1)*self.hop_length
        sig_slices = input[:,slices.long()] * self.win.view(1,-1,1) / self.normalizer
        output = torch.fft.rfft(sig_slices,n=self.n_fft,dim=-2) 
        # compress
        output = output.abs().pow(self.compression) * torch.exp(1j*output.angle())
        return output

    def postprocess(self,input:torch.Tensor):
        # decompress
        input = input.abs().pow(1.0/self.compression) * torch.exp(1j*input.angle())
        # istft
        ypad_slices = torch.fft.irfft(input,n=self.n_fft,dim=-2,norm=None)[:,:self.win_length] * self.win.view(1,-1,1) * self.normalizer
        fold = torch.nn.Fold(output_size=(1,self.padder.L),kernel_size=(1,self.win_length),stride=(1,self.hop_length))
        ypad = fold(ypad_slices).view(-1,self.padder.L) * 2/3
        # unpad
        output = self.padder.unpad(ypad)
        return output

    def forward(self,input:torch.tensor):
        Y_stft = self.preprocess(input)
        Y = Y_stft.abs()
        output = self.model(Y,stochastic=False)

        Y_hat = output['recon_x']
        S_hat = output['recon_s']

        if self.mask:
            S_hat = S_hat * (Y/Y_hat).unsqueeze(1)
        
        S_hat_stft = S_hat * torch.exp(1j*Y_stft.angle()).unsqueeze(1)

        dims = S_hat_stft.shape
        S_hat_stft = S_hat_stft.view(-1,dims[-2],dims[-1])
        s_hat = self.postprocess(input=S_hat_stft)
        s_hat = s_hat.view(dims[0],dims[1],-1)
        return s_hat

class Conv1dLayer(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size=3,stride=1,padding=0,dilation=1,batchnorm=True,activation=True,bias=True) -> None:
        super(Conv1dLayer,self).__init__()

        mods = nn.ModuleList()
        mods.append(nn.Conv1d(input_channels,output_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=bias))
        if activation:
            mods.append(nn.LeakyReLU())
        if batchnorm:
            mods.append(nn.BatchNorm1d(output_channels))
        self.net = nn.Sequential(*mods)

    def forward(self,x):
        return self.net(x)

class Conv1dSeparator(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size=1,stride=1,padding=0,dilation=1,layers=1,layer_channels=32):
        super(Conv1dSeparator, self).__init__()

        kwds = dict(kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        
        mlist = nn.ModuleList()
        mlist.append(Conv1dLayer(input_channels,layer_channels,**kwds))
        for ii in range(layers-1):
            mlist.append(Conv1dLayer(layer_channels,layer_channels,**kwds))
        mlist.append(Conv1dLayer(layer_channels,output_channels,batchnorm=False,**kwds))

        self.Network = nn.Sequential(*mlist)

    def forward(self, x):
        return self.Network(x)

class TConvSeparator(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size=2,stride=1,padding=0,dilation=1,layers=1,layer_channels=32,starting_layer=0,bias=True):
        super(TConvSeparator, self).__init__()

        dilations = 2**(np.arange(layers)+starting_layer)
        fields = (kernel_size-1)*dilations+1
        paddings = fields//2

        mlist = nn.ModuleList()
        if layers == 1:
            kwds = dict(kernel_size=kernel_size,stride=stride,padding=paddings[0],dilation=dilations[0])
            mlist.append(Conv1dLayer(input_channels,output_channels,batchnorm=True,bias=bias,**kwds))
        else:
            mlist.append(Conv1dLayer(input_channels,layer_channels,kernel_size=kernel_size,stride=stride,padding=paddings[0],dilation=dilations[0],batchnorm=True,bias=bias))
            for ii in range(1,layers-1):
                mlist.append(Conv1dLayer(layer_channels,layer_channels,kernel_size=kernel_size,stride=stride,padding=paddings[ii],dilation=dilations[ii],batchnorm=True,bias=bias))
            mlist.append(Conv1dLayer(layer_channels,output_channels,kernel_size=kernel_size,stride=stride,padding=paddings[-1],dilation=dilations[-1],batchnorm=False,bias=bias))

        self.Network = nn.Sequential(*mlist)

    def forward(self, x):
        return self.Network(x)

class StridedDilatedEncoder(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size=(3,3),stride=(2,1),padding=(1,1),dilation=(1,1),layer_channels=None,bias=True):
        super(StridedDilatedEncoder, self).__init__()
        if layer_channels is None: 
            layer_channels = (32,64,128)

        layers = len(layer_channels)
        dilations = 2**np.arange(layers)
        fields = (kernel_size[-1]-1)*dilations+1
        padding_1 = padding[0]
        padding_2 = fields//2
        padding = [(padding_1,k) for k in padding_2]
        dilation = [(1,k) for k in dilations]

        mlist = nn.ModuleList()
        mlist.append(Conv2dLayer(input_channels,layer_channels[0],kernel_size=kernel_size,stride=stride,padding=padding[0],dilation=dilation[0],bias=bias))
        for ii in range(1,len(layer_channels)-1):
            mlist.append(Conv2dLayer(layer_channels[ii-1],layer_channels[ii],kernel_size=kernel_size,stride=stride,padding=padding[ii],dilation=dilation[ii],bias=bias))
        mlist.append(Conv2dLayer(layer_channels[-1],output_channels,kernel_size=kernel_size,stride=stride,padding=padding[-1],dilation=dilation[-1],bias=bias))

        self.Network = nn.Sequential(*mlist)

    def forward(self, x):
        return self.Network(x)
    
class DSVAE(DCAE):
    def __init__(self, dimx, dimz, dimh=64, dimg=64, chans=64, num_sources=1, net_type='strided', num_layers=1, scale_invariant=False, layer_channels=None, kernel_size=(3, 2), stride=(2, 1), padding=(1, 0), dilation=(1, 1), variance=0.1, activate_final_layer=True, separator_kernel_size=1,separator_layers=1,separator_dilation=1,separator_channels=32,rnn_type='gru',inference_rnn=True):
        if layer_channels is None:
            layer_channels = (32,64,64,64)

        super().__init__(dimx, dimz, dimh, dimg, chans, num_sources, net_type, num_layers, scale_invariant, layer_channels, kernel_size, stride, padding, dilation, variance, activate_final_layer, separator_kernel_size,rnn_type=rnn_type,inference_rnn=inference_rnn)

        kwds_feature_conv = dict(kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        self.Encoder = StridedDilatedEncoder(1,self.conv_channels,layer_channels=layer_channels,**kwds_feature_conv)

        self.fc_encoder = TConvSeparator(input_channels=self.encoded_features,output_channels=self.dimg*self.num_sources,kernel_size=separator_kernel_size,stride=1,padding=1,dilation=1,layers=separator_layers,layer_channels=separator_channels,starting_layer=len(layer_channels))



#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils.utils import receptive_field

    layers = 5
    dilation = 2**np.arange(layers)
    kernel_size = 3

    field = (kernel_size-1)*dilation+1

    print(receptive_field(field,field*0+1))
    layer_channels = tuple([32]*layers)
     
    T = 16000*4
    input = torch.randn(2,T)
    input*= 0
    input[:,T//2] = 1
    model = DSVAE(num_sources=2,dimx=257,dimz=16,dimh=256,kernel_size=(3,3),stride=(2,1),padding=(1,1),dilation=(1,1),layer_channels=layer_channels,activate_final_layer=True,separator_kernel_size=kernel_size,separator_layers=2,separator_dilation=1,rnn_type='gru',separator_channels=64,inference_rnn=False)

    separator = Separator(model=model,compression=.5)
    with torch.no_grad():
        output = separator(input)
        data = separator.preprocess(input).abs().unsqueeze(1)
        data = torch.zeros_like(data)
        data[...,data.shape[-1]//2] = 1
        features = model.Encoder(x=data)
        print(features.shape)
        feats = model.feature_encoder(x=data)


    # plt.plot((input[0]-output[0,0]))
    # plt.figure()

    # net = TConvSeparator(input_channels=320,output_channels=128,kernel_size=2,stride=1,padding=0,dilation=1,layers=3)
    # print(net)
    
    # # %%
    # features_time = features.view(-1,features.shape[-1]).T
    # plt.subplot(211)
    # plt.imshow(data[0,0],aspect='auto')
    # plt.subplot(212)
    # plt.imshow(features_time.T,aspect='auto')
    # # %%
    # plt.plot(features_time.mean(-1))
    # # %%
    # temp = features_time.mean(-1)
    # (temp > 0).sum()
    # # %%
    # temp2 = feats.transpose(-2,-1)
    # temp2 = temp2.view(-1,temp2.shape[-1]).mean(0)
    # plt.plot(temp2)
    # print((temp2 > 0).sum())
    # # %%
