
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

args = parser.parse_args()

args.model_directory = "/Users/julian/Documents/Code/Python/Results/SVAE/RWC/dilated/"
args.output_directory = "/Users/julian/Documents/Code/Python/Results/SVAE/RWC/dilated"
args.data_directory = "/Volumes/JulianSSD/data/NSYNTH_test"
args.job_id = "230208-13605818"
path_data = Path(args.data_directory) 
path_model = Path(args.model_directory) / args.job_id
path_out = Path(args.output_directory) / args.job_id

model_path = list(path_model.glob("*.pt"))
if len(model_path) == 0:
    raise ValueError("no models found in path")
else:
    model_path = model_path[0]

path_audio_out = path_out / "audio"
os.makedirs(path_out,exist_ok=True)
os.makedirs(path_audio_out,exist_ok=True)

fname = path_model / "configuration.txt"
with open(fname) as f:
    config = f.read()
config = json.loads(config)
with open(path_out / 'configuration.txt', 'w') as f:
    json.dump(config, f, indent=2)
parser.set_defaults(**config)
args = parser.parse_args()

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

test_set = NSYNTH(root=path_data)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,shuffle=False)

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

kwds_model = dict(
    dimx=dim_data,
    dimz=args.latent_dim,
    dimh=args.rnn_dim,
    dimg=args.rnn_dim,
    num_sources=args.num_sources,
    net_type=args.net_type,
    num_layers=args.num_layers,
    scale_invariant=True,
    layer_channels=tuple(args.cnn_channels),
    kernel_size=kernel_size,stride=stride,padding=padding,
    variance=args.variance,
    activate_final_layer=args.activate_final_layer,
    rnn_type=args.rnn_type,
    inference_rnn=(not args.inference_rnn_off),
    )
                
if args.separator_version:
    model = DSVAE(separator_layers=args.separator_layers,separator_kernel_size=args.separator_kernel_size,separator_channels=args.separator_channels,**kwds_model)
else:
    model = DCAE(**kwds_model)

model.load_state_dict(load_state_dict(path=model_path))

model = Separator(model,compression=args.compression)
model.to(device)
model.eval()

print("evaluating {} examples.".format(len(test_set)))

results = dict(sdr=[],sir=[],sar=[],place=[],instrument=[],instrument1=[],instrument2=[],delta_time=[],pitch=[],delta_pitch=[])
count = 0
with torch.no_grad():
    for batch_idx, (data) in enumerate(test_loader):
        mix = data['mix'][:,0].to(device)
        sources = data['stems'].to(device)
        # Inference
        estimates = model(mix)

        estimates, perm = optimal_permute(estimates,sources)
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

        _,place = torch.sort(SDR)
        
        results['sdr'].append(SDR.cpu().numpy().ravel())
        results['sar'].append(SAR.cpu().numpy().ravel())
        results['sir'].append(SIR.cpu().numpy().ravel())
        results['place'].append(place.cpu().numpy().ravel())
        results['instrument'].append(np.stack([data['instrument1'],data['instrument2']],-1).ravel())
        for key in ['instrument1','instrument2']:
            results[key].append(np.array(data[key])[:,None].repeat(2,1).ravel())

        if 'delta_time' in data:
            results['delta_time'].append(np.array(data['delta_time'])[:,None].repeat(2,1).ravel())

        if 'pitch1' in data:
            results['pitch'].append(np.stack([data['pitch1'],data['pitch2']],-1).ravel())
            results['delta_pitch'].append((np.array(data['pitch1'])-np.array(data['pitch2']))[:,None].repeat(2,1).ravel())

        if save_audio:
            for i in range(estimates.shape[0]):
                torchaudio.save(path_audio_out / "estimates_{}.wav".format(count),estimates[i,:2],16000)
                count += 1

        print("Batch {}/{}".format(batch_idx+1,len(test_loader)))
        if batch_idx >= 5: 
            break

for k in list(results):
    if len(results[k])  > 0:
        results[k] = np.concatenate(results[k])
    else:
        results.pop(k)

df = pd.DataFrame(results)
df.to_json(path_out / 'results.json',orient='index', indent=2)


plt.figure(figsize=(5,8))
sns.boxplot(data=df)
plt.savefig(path_out / "boxplot.png",dpi=200)

print(df.mean())
