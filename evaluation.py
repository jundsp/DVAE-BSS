#%%
import torch
import torchaudio
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from model import *
from dataset import *
from argparser import parser
from utils.utils import optimal_permute, ideal_masks, load_state_dict, compute_measures_batch, make_scale_invariant
from collections import OrderedDict
import seaborn as sns
from utils.utils import torchToNumpy, mag2db, panning
import glob
import re
import shutil
import json

model_path = Path("/Users/julian/Documents/Code/Python/Results/SVAE/RWC/compression/K5/230122-13058078/")
model_filename = model_path / "svae_RWC_S2_K5_L1_pooled_D256_C5-10.pt"

path_out = Path("/Users/julian/Documents/Code/Python/Results/SVAE/ASA")
path_audio_out = Path("/Volumes/JulianSSD/research/svae-bss/twotone") / "audio"
os.makedirs(path_out,exist_ok=True)
os.makedirs(path_audio_out,exist_ok=True)

fname = model_path / "configuration.txt"
with open(fname) as f:
    data = f.read()
args = json.loads(data)
parser.set_defaults(**args)
args = parser.parse_args(args=[])

args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

eval_ideal_masks= False
if eval_ideal_masks:
    args.compression = 1

save_audio = False
normalize = False
nfft = 512
averaged = False
dim_data = nfft//2+1

test_set = AuditoryStreams(root="/Users/julian/Documents/Code/data/dataset_harmonic/")
test_loader = torch.utils.data.DataLoader(test_set, batch_size=16,shuffle=False)

#%% LOAD MODEL
dimx = dim_data
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

#%%
model.eval()
print("evaluating {} examples.".format(len(test_set)))

results = dict(sdr=[],sir=[],sar=[],delta_time=[],delta_semitone=[],physics1=[],physics2=[])

count = 0
with torch.no_grad():
    for batch_idx, (data) in enumerate(test_loader):
        mix = data['mix'][:,0]
        sources = data['stems']
        estimates = model(mix)

        estimates = optimal_permute(estimates,sources)
        sources_invariant = make_scale_invariant(estimates=estimates,targets=sources)

        SDR, SIR, SAR = compute_measures_batch(estimates=estimates,targets=sources_invariant)

        
        if eval_ideal_masks:
            IRM,IBM = ideal_masks(stft_sources=sources_stft)
            irm = pipeline.invert(IRM)
            ibm = pipeline.invert(IBM)
            baseline_mix = mix.unsqueeze(1).repeat(1,2,1)
            baseline_mix =  irm
            sources_invariant = make_scale_invariant(estimates=baseline_mix,targets=sources)
            SDR, SIR, SAR = compute_measures_batch(estimates=baseline_mix,targets=sources_invariant)

        _,idx = torch.sort(SDR)
        
        results['sdr'].append(SDR.cpu().numpy().mean(-1))
        results['sar'].append(SAR.cpu().numpy().mean(-1))
        results['sir'].append(SIR.cpu().numpy().mean(-1))
        for key in ['delta_time','delta_semitone','physics1','physics2']:
            results[key].append(data[key])

        if save_audio:
            for i in range(estimates.shape[0]):
                torchaudio.save(path_audio_out / "estimates_{}.wav".format(count),estimates[i,:2],16000)
                count += 1

        print("Batch {}/{}".format(batch_idx+1,len(test_loader)))
        # if batch_idx > 1:
        #     break

        

#%%
for k in results:
    results[k] = np.concatenate(results[k])

# %%
df = pd.DataFrame(results)
df['physics1'].replace(['forced','forced_vibrato'],['sustain','vibrato'] , inplace=True)
df['physics2'].replace(['forced','forced_vibrato'],['sustain','vibrato'] , inplace=True)
df['physics'] = df['physics1'] + '-' + df['physics2']
# df.to_json(path_out / 'results.json',orient='index', indent=2)

# %%
sns.lineplot(data=df,x='delta_time',y='sdr',hue='physics')
plt.figure()
sns.lineplot(data=df,x='delta_semitone',y='sdr',hue='physics')

# %%
# sns.heatmap(data=df,x='delta_time',y='delta_semitone',hue='sdr')
# %%
sns.lineplot(data=df,x='delta_time',y='sdr')
plt.figure()
sns.lineplot(data=df,x='delta_semitone',y='sdr')
# %%
