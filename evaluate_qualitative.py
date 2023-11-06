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

args = parser.parse_args(args=[])

args.model_directory = "/Volumes/JulianSSD/research/svae-bss/output/RWC/saves/ablation_targets"
args.output_directory = "/Volumes/JulianSSD/research/svae-bss/evaluation/NSYNTH_test/ablation_targets"
args.data_directory = "/Volumes/JulianSSD/data/NSYNTH_test"
args.job_id = "231017-22161497"
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

normalize = False
nfft = 512
averaged = False
dim_data = nfft//2+1
layer_channels=tuple(args.cnn_channels)

test_set = NSYNTH(root=path_data)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,shuffle=False)

#%%
# LOAD MODEL
print(args)

if args.separator_version:
    kernel_size=(3,3)
    stride=(2,1)
    padding=(1,1)
else:
    kernel_size=(3,3)
    stride=(2,2)
    padding=(1,1)
    # layer_channels = layer_channels + tuple([layer_channels[-1]])

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
                
if args.separator_version:
    model = DSVAE(separator_layers=args.separator_layers,separator_kernel_size=args.separator_kernel_size,separator_channels=args.separator_channels,**kwds_model)
else:
    model = DCAE(**kwds_model)

model.load_state_dict(load_state_dict(path=model_path))

model = Separator(model,compression=args.compression)
model.to(device)
model.eval()

#%%

save_audio = True

if save_audio:
    os.makedirs(path_audio_out,exist_ok=True)

for (i,data) in enumerate(test_loader):
    if i == 0:
        break



with torch.no_grad():
        mix = data['mix'][:,0].to(device)
        sources = data['stems'].to(device)
        # Inference
        estimates = model(mix)

        estimates, perm = optimal_permute(estimates,sources)
        sources_invariant = make_scale_invariant(estimates=estimates,targets=sources)

        SDR, SIR, SAR = compute_measures_batch(estimates=estimates,targets=sources_invariant)

        if save_audio:
            for i in range(estimates.shape[0]):
                path_out_i = path_audio_out / "test_{}".format(i)
                os.makedirs(path_out_i,exist_ok=True)
                info = dict(SDR=list(SDR[i].numpy()),SIR=list(SIR[i].numpy()),SAR=list(SAR[i].numpy()))
                info['index'] = [data['index1'][i].item(),data['index2'][i].item()]
                info['instrument'] = [data['instrument1'][i],data['instrument2'][i]]
                pd.DataFrame(info).to_csv(path_out_i/'info.csv')
                torchaudio.save(path_out_i / "mixture.wav",mix[[i]],16000)
                torchaudio.save(path_out_i / "targets.wav",sources[i,:2],16000)
                torchaudio.save(path_out_i / "estimates.wav",estimates[i,:2],16000)

# %%
