#%%
import torch
from torchaudio.transforms import Spectrogram
import os
import matplotlib.pyplot as plt
from model import *
from dataset import *
from argparser import parser

from utils.utils import load_state_dict
import json
import soundfile as sf
# import samplerate
import sines
from pedalboard.io import AudioFile
import glob
from utils.utils import optimal_permute, compute_measures_batch,make_scale_invariant

sines.style(plt)

#%%
K = 5
model_path = Path("output/svae/RWC/targets/231017-22161498")

files = glob.glob(str(model_path / "*.pt"))
model_filename = files[0]

path_out = model_path / "inference"

fname = model_path / "configuration.txt"
with open(fname) as f:
    data = f.read()
args = json.loads(data)
parser.set_defaults(**args)
args = parser.parse_args(args=[])

args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

nfft = 512
dimx = nfft//2+1
dimz = args.latent_dim
dimh = args.rnn_dim
dimg = args.rnn_dim
kernel_size=(3,3)
stride=(2,1)
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
        activate_final_layer=True,
        inference_rnn=False,
        rnn_type=args.rnn_type,
        separator_channels=args.separator_channels,
        # separator_channels=args.separator_channels,
        separator_layers=args.separator_layers,
        separator_kernel_size=args.separator_kernel_size
        )
                

model = DCAE(num_sources=K,dimx=dimx,dimz=dimz,dimh=dimh,dimg=dimg,kernel_size=(3,3),stride=(2,2),padding=(1,1),layer_channels=(32,64,128,128,128),activate_final_layer=True)
# model = DSVAE(**kwds_model)
model.load_state_dict(load_state_dict(path=model_filename))

model = Separator(model,compression=args.compression,mask=True)
model.to(device)
model.eval()

#%%

filename = "mixture"
audio_root = "data/violin_scale/"
audio_format = "wav"


path_out = path_out / filename
os.makedirs(path_out,exist_ok=True)


stem_files = sorted(glob.glob(audio_root + "*.wav"))

stems = []
for s in stem_files:
    _temp, sr = sf.read(s)
    stems.append(_temp)
sources = torch.from_numpy(np.stack(stems,0)).float().unsqueeze(0)
input = sources.sum(1)
normalize = input.abs().max()
input /= normalize
sources /= normalize


# duration = 30
# with AudioFile("{}/{}.{}".format(audio_root,filename,audio_format)) as f:
#     sr = f.samplerate
#     input = f.read(int(sr * duration))

# input = torch.from_numpy(input)
# input = torchaudio.functional.resample(input,sr,16000)
# input = input.sum(0).view(1,-1)
# dset = Audioset(root="data/wsj0")
# dload = torch.utils.data.DataLoader(dataset=dset,batch_size=8,shuffle=True)
# mixer = Mixer(n_sources=2,random_num=True)
# batch = next(iter(dload))[0]
# mix,source = mixer(input)
# input = mix[[0]]

with torch.no_grad():
    estimates = model(input)


estimates, perm = optimal_permute(estimates,sources)
sources_invariant = make_scale_invariant(estimates=estimates,targets=sources)

SDR, SIR, SAR = compute_measures_batch(estimates=estimates,targets=sources_invariant)


#%%
to_plot = np.arange(args.num_sources)
fig,ax = plt.subplots(1+len(to_plot),2,figsize=(5,6.0),sharex=True,sharey=True)
S,t,f = sines.stft(input[0].numpy())
sines.plot_spectrogram(S,t,f,ax=ax[0,0],kHz=True,**dict(vmin=-160,vmax=-25))
# 
ax[0,0].set_ylabel('freq. (kHz)')
ax[0,0].set_xlim(0,7.0)
for m,k in enumerate(to_plot):
    i = m + 1

    col = 0
    S,t,f = sines.stft(sources[0,k].numpy())
    sines.plot_spectrogram(S,t,f,ax=ax[i,col],kHz=True,**dict(vmin=-160,vmax=-25))
    ax[i,col].set_xlim(0,7.0)
    sf.write(file=path_out/ "targets_{}.wav".format(i),data=sources[0,k],samplerate=sr)

    col = 1
    S,t,f = sines.stft(estimates[0,k].numpy())
    sines.plot_spectrogram(S,t,f,ax=ax[i,col],kHz=True,**dict(vmin=-160,vmax=-25))
    sf.write(file=path_out/ "estimates_{}.wav".format(i),data=estimates[0,k],samplerate=sr)
    ax[i,col].set_xlim(0,7.0)

    ax[i,0].set_ylabel(r'$f$ (kHz)')
    
ax[-1,0].set_xlabel('time (s)')
ax[-1,1].set_xlabel('time (s)')
ax[0,1].remove()
# ax[1,0].set_title("Targets",y=.8)
# ax[1,1].set_title("Estimates",y=.8)
fig.subplots_adjust(wspace=0, hspace=0)

# plt.tight_layout()
plt.savefig(path_out/"spectrograms.pdf".format(filename),bbox_inches="tight")


left_mix = torch.linspace(0,1,K)
lr = torch.stack([left_mix,1-left_mix],-1)
estimates_mixed = lr.T @ estimates[0]
sf.write(file=path_out / "{}.wav".format("bounced"),data=estimates_mixed.T,samplerate=sr)

sf.write(file=path_out/ "{}.wav".format("input"),data=input[0],samplerate=sr)
