#%%
import torch
import torchaudio
import os
import matplotlib.pyplot as plt
from model import *
from dataset import *
from argparser import parser
from utils.utils import optimal_permute
from collections import OrderedDict
from utils.utils import torchToNumpy, mag2db
import glob
import re
import samplerate 

#* Make a class / module to process the bss eval measures.

def load_state_dict(path):
    # original saved file with DataParallel
    state_dict = torch.load(path,map_location=torch.device(device))
    # create new OrderedDict that does not contain `module.`
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if 'module.' == name[:7]:
            name = name[7:] # remove `module.`
        new_state_dict[name] = v
    
    return new_state_dict

def make_scale_invariant(estimates,targets):
    k = targets.shape[1]
    estimates = estimates[:,:k]
    Rss = (targets * targets).sum(-1)
    alpha = (estimates * targets).sum(-1) / Rss
    return targets * alpha.unsqueeze(-1)


def mp2comp(magnitude,phase):
    Zc = magnitude*torch.exp(1j*phase)
    Z = torch.view_as_real(Zc)
    return Z

def pad_full(Z,nfft):
    out = torch.zeros(Z.size(0),nfft//2+1,Z.size(2),2)
    out[:,:Z.size(1)] = Z
    return out

# Batch Synthesize
def synthesize(magnitude,phase,istft):
    X = mp2comp(magnitude=magnitude,phase=phase)
    X = pad_full(X) 
    x = istft(X)
    return x


#%%
args = parser.parse_args(args=[])
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Check the parameters of the model saved in that directory.
root = "/Users/julian/Desktop/svae-eval/svae-s5/"
args.model_directory = root + "models"
os.chdir(args.model_directory)
model_name = glob.glob("*.pt")[0]
print(model_name)
match = re.match(r'svae_([^_]+)_S(\d+)_K(\d+)_L1_pooled_D(\d+)_C(\d+)-10.pt', model_name)

args.num_targets = int(match.group(2))
args.num_sources = int(match.group(3))
args.latent_dim = int(match.group(4))
args.compression = float(match.group(5)) * .1
args.conv2d = True
args.rnn_dim = 1024

print(args.num_targets)
print(args.num_sources)
print(args.latent_dim)
print(args.compression)

args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

nfft = 512
averaged = False
dim_data = (nfft//2+1,args.duration)

args.num_targets=1
mixer = Mixer(n_sources=args.num_targets,average=averaged)
args.n_time = 257+128
pipeline = AudioPipeline(n_fft=nfft,hop_length=128,compression=args.compression,n_time=args.n_time)
mixer = mixer.to(device)
pipeline = pipeline.to(device)

#%%

filename = "trombone_violin"
input_file = "/Users/julian/Documents/MATLAB/sounds/{}.wav".format(filename)
audio,sr = torchaudio.load(input_file)


# input_file = "/Users/julian/Documents/MATLAB/sounds/example16k_source_1.wav".format(filename)
# audio1,sr = torchaudio.load(input_file)
# input_file = "/Users/julian/Documents/MATLAB/sounds/example16k_source_2.wav".format(filename)
# audio2,sr = torchaudio.load(input_file)
# audio = torch.cat([audio1,audio2],0)
# audio += torch.randn(audio.shape)*1e-5

audio = torchaudio.functional.resample(audio,orig_freq=sr,new_freq=16000.0,resampling_method='kaiser_window')
# audio = samplerate.resample(audio.numpy().ravel(),16000.0/sr)
# audio = torch.from_numpy(audio).view(1,-1)

#%% LOAD MODEL

dimx = dim_data[0]
dimz = args.latent_dim
dimh = args.rnn_dim
dimg = args.rnn_dim
kernel_size=(3,3)
stride=(2,2)
padding=(1,1)

model = DCAE(
        dimx=dimx,
        dimz=dimz,
        dimh=dimh,
        dimg=dimg,
        num_sources=args.num_sources,
        net_type=args.net_type,
        num_layers=args.num_layers,
        scale_invariant=True,
        layer_channels=(32,64,128,128),
        kernel_size=kernel_size,stride=stride,padding=padding,
        )

model.load_state_dict(load_state_dict(path=args.model_directory + "/" + model_name))
model.to(device)
model.eval()
pipeline.compression = args.compression

with torch.no_grad():
    mix, sources = mixer(waveform=audio.to(device))
    mix_stft = pipeline.forward_complex(mix)
    source_stft = pipeline.forward_complex(sources)

    data = mix_stft.abs()
    output = model(data)

    s_hat = output['recon_s']
    m_hat = create_masks(data,s_hat)
    estimates_stft = m_hat * torch.exp(1j*mix_stft.angle().unsqueeze(1))

    mix = pipeline.invert(mix_stft)
    estimates = pipeline.invert(estimates_stft)
    sources = pipeline.invert(source_stft)

    estimates = optimal_permute(estimates,sources)


# %%
args.output_directory = root + filename
os.makedirs(args.output_directory,exist_ok=True)

kwds = dict(cmap='jet',origin='lower',aspect='auto',interpolation='none',vmin=-160,vmax=2)
pipeline.compression = 1.0
n = 0
fig,ax = plt.subplots(args.num_sources,3,figsize=(9,3*args.num_sources))
ax[0,0].imshow(mag2db(pipeline(mix[n])+1e-10).cpu(),**kwds)
for i in range(args.num_targets):
    ax[i,1].imshow(mag2db(pipeline(sources[n,i])+1e-9).cpu(),**kwds)
for i in range(args.num_sources):
    ax[i,2].imshow(mag2db(pipeline(estimates[n,i])+1e-9).cpu(),**kwds)

ax[0,0].set_title('Input Mixture')
ax[0,1].set_title('Target')
ax[0,2].set_title('Estimates')

for i in range(1,args.num_sources):
    ax[i,0].remove()
for i in range(args.num_targets,args.num_sources):
    ax[i,1].remove()

plt.tight_layout()
plt.savefig(args.output_directory + "/evaluation_spec.png")
# plt.close()

plt.savefig(args.output_directory + "/spectrograms.pdf")
# %%

torchaudio.save(args.output_directory + "/mix.wav",mix[n].unsqueeze(0).cpu() / 2,16000)
for i in range(estimates.shape[1]):
    torchaudio.save(args.output_directory + "/estimate_{}.wav".format(i+1),estimates[n,i].unsqueeze(0).cpu() / 2,16000)
    
# %%
mix_down = torch.zeros((2,mix[n].shape[0]))
nsrc = estimates.shape[1]
for i in range(nsrc):
    panr = i/(nsrc-1.0)
    panl = 1-panr
    pan = torch.tensor([panl,panr])
    mix_down += estimates[n,i].view(1,-1).cpu() * pan.view(-1,1)

torchaudio.save(args.output_directory + "/mixdown_estimates.wav",mix_down.cpu() / 2,16000)
# %%
