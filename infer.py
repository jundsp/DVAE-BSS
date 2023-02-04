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
import samplerate
import sines

plt.rcParams['figure.dpi'] = 150
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times",
    "axes.titlesize": "medium",
    "axes.labelsize": "medium",
})

#%%

model_path = Path("/Users/julian/Documents/Code/Python/Results/SVAE/RWC/compression/K5/230122-13058078/")
model_filename = model_path / "svae_RWC_S2_K5_L1_pooled_D256_C5-10.pt"

path_out = Path("/Users/julian/Documents/Code/Python/Results/SVAE/music")
os.makedirs(path_out,exist_ok=True)

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
        activate_final_layer=True
        )
                
model = DCAE(**kwds_model)
model.load_state_dict(load_state_dict(path=model_filename))

model = Separator(model,compression=args.compression)
model.to(device)
model.eval()

#%%
filename = "trombone_violin"
audio,sr = torchaudio.load("/Users/julian/Documents/MATLAB/sounds/{}.wav".format(filename))
input = audio[[0]]
input = torchaudio.functional.resample(input,sr,16000)
sr = 16000

with torch.no_grad():
    estimates = model(input)

# %%
spec = Spectrogram(n_fft=512,hop_length=64,power=1,normalized=True)
S = spec(estimates).abs().log10()*20

#%%
to_plot = [0,3,4]
to_plot = np.arange(5)
fig,ax = plt.subplots(len(to_plot),figsize=(4,4),sharex=True)
ax = ax.ravel()
for i,k in enumerate(to_plot):
    ax[i].imshow(S[0,k],**sines.plot_spectrogram_kwds(),cmap='jet',vmin=-150,vmax=0)
    sf.write(file=path_out/ "{}_{}.wav".format(filename,i+1),data=estimates[0,k],samplerate=sr)
plt.tight_layout()
# %%
