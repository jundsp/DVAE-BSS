#%%
import torch
import torchaudio
import os
import matplotlib.pyplot as plt
from model import *
from dataset import *
from argparser import parser
from utils.utils import optimal_permute, ideal_masks, load_state_dict, compute_measures_batch, make_scale_invariant
import seaborn as sns
import json
from pathlib import Path

plt.style.use("gibbs.mplstyles.latex")

args = parser.parse_args(args=[])

args.model_directory = "/Users/julian/Documents/Code/output/svae/RWC/mixit"
args.data_directory = "/Users/julian/Documents/Code/data/TwoNote"
args.output_directory = "/Users/julian/Documents/Code/data/TwoNote/baselines/mixit"

args.job_id = "231106-22715625"
path_data = Path(args.data_directory) 
path_model = Path(args.model_directory) / args.job_id
path_out = Path(args.output_directory) / args.job_id

#%%
model_path = list(path_model.glob("*.pt"))
if len(model_path) == 0:
    raise ValueError("no models found in path")
else:
    model_path = model_path[0]

path_audio_out = path_out / "audio"
os.makedirs(path_out,exist_ok=True)

fname = path_model / "configuration.txt"
with open(fname) as f:
    config = f.read()
config = json.loads(config)
with open(path_out / 'configuration.txt', 'w') as f:
    json.dump(config, f, indent=2)
parser.set_defaults(**config)

#%%
args = parser.parse_args(args=[])
args.batch_size = 16

args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

eval_ideal_masks= False
if eval_ideal_masks:
    args.compression = 1

save_audio = False

if save_audio:
    os.makedirs(path_audio_out,exist_ok=True)

normalize = False
nfft = 512
averaged = False
dim_data = nfft//2+1
layer_channels=tuple(args.cnn_channels)

test_set = NSYNTH(root=path_data)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,shuffle=True)

#%%
# LOAD MODEL
print(args)

kernel_size=(3,3)
stride=(2,2)
padding=(1,1)

kwds_model = dict(
    dimx=dim_data,
    dimz=args.latent_dim,
    dimh=args.rnn_dim,
    dimg=args.rnn_dim,
    num_sources=args.num_sources,
    net_type=args.net_type,
    num_layers=args.num_layers,
    scale_invariant=True,
    layer_channels=layer_channels,
    kernel_size=kernel_size,stride=stride,padding=padding,
    variance=args.variance,
    activate_final_layer=args.activate_final_layer,
    rnn_type=args.rnn_type,
    inference_rnn=(not args.inference_rnn_off),
    autoencoder=args.autoencoder
    )
                

model = DCAE_MixIT(**kwds_model)
model.load_state_dict(load_state_dict(path=model_path))
model.eval()
model = model.to(device)

separator = Separator(model=None,compression=args.compression)
separator = separator.to(device)

mixer = Mixer(n_sources=args.num_targets,average=args.average_mix,random_num=False)
pipeline = AudioPipeline(n_fft=nfft,hop_length=128,compression=args.compression,n_time=args.n_time,output_realimag=args.complex_version)
# mixer = mixer.to(device)
pipeline = pipeline.to(device)

# %%
data = next(iter(test_loader))
data = data['stems']
data = data.view(-1,1,data.shape[-1])
mix, stems = mixer(data.squeeze(1))
mix = pipeline(mix.to(device))
inputs, targets = mixer(mix)

#%%
with torch.no_grad():
    estim_sources = model(inputs.unsqueeze(1))
# %%
num_plot = 4
fig,ax = plt.subplots(estim_sources.shape[1],num_plot)
for i in range(ax.shape[0]):
    for k in range(num_plot):
        ax[i,k].imshow(estim_sources[k,i].log10()*20,cmap='jet')
        ax[i,k].axis("off")
plt.tight_layout()

# %%
