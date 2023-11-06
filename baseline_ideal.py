import torch
import os
import matplotlib.pyplot as plt
from model import *
from dataset import *
from argparser import parser
from utils.utils import optimal_permute, ideal_masks, compute_measures_batch, make_scale_invariant
import seaborn as sns
from pathlib import Path

args = parser.parse_args()

args.data_directory = "/Users/julian/Documents/Code/data/TwoNote"
args.output_directory = "/Users/julian/Documents/Code/data/TwoNote/baselines"
path_data = Path(args.data_directory) 
path_out = Path(args.output_directory)

os.makedirs(path_out,exist_ok=True)

args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

eval_ideal_masks= True
if eval_ideal_masks:
    args.compression = 1

save_audio = False
normalize = False
nfft = 512
averaged = False
dim_data = nfft//2+1

test_set = NSYNTH(root=path_data)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,shuffle=False)
separator = Separator(model=None,compression=1.0).to(device)

# LOAD MODEL
print(args)
print("evaluating {} examples.".format(len(test_set)))

results_mix = dict(sdr=[],sir=[],sar=[],instrument=[],instrument1=[],instrument2=[],delta_time=[],pitch=[],delta_pitch=[])
results_irm = results_mix
results_ibm = results_mix
results = dict(mix=results_mix,irm=results_irm,ibm=results_ibm)

count = 0
with torch.no_grad():
    for batch_idx, (data) in enumerate(test_loader):
        mix = data['mix'][:,0].to(device)
        sources = data['stems'].to(device)
        
        # MIX
        estimates = torch.stack([mix]*2,1)
        estimates, perm = optimal_permute(estimates,sources)
        sources = make_scale_invariant(estimates=estimates,targets=sources)
        SDR_mix, SIR_mix, SAR_mix = compute_measures_batch(estimates=estimates,targets=sources)

        # IRM / IBM
        sources_stft = separator.preprocess(sources.reshape(-1,sources.shape[-1]))
        sources_stft = sources_stft.reshape(-1,2,*(sources_stft.shape[1:]))
        IRM,IBM = ideal_masks(stft_sources=sources_stft)
        IRM = IRM.reshape(-1,*(IRM.shape[-2:]))
        irm = separator.postprocess(IRM)
        irm = irm.reshape(sources.shape)
        sources_invariant = make_scale_invariant(estimates=irm,targets=sources)
        SDR_irm, SIR_irm, SAR_irm = compute_measures_batch(estimates=irm,targets=sources_invariant)

        IBM = IBM.reshape(-1,*(IBM.shape[-2:]))
        ibm = separator.postprocess(IBM)
        ibm = ibm.reshape(sources.shape)
        sources_invariant = make_scale_invariant(estimates=ibm,targets=sources)
        SDR_ibm, SIR_ibm, SAR_ibm = compute_measures_batch(estimates=ibm,targets=sources_invariant)
        
        # ICA
        
        # NMF
        
        results['mix']['sdr'].append(SDR_mix.cpu().numpy().ravel())
        results['mix']['sar'].append(SAR_mix.cpu().numpy().ravel())
        results['mix']['sir'].append(SIR_mix.cpu().numpy().ravel())

        results['irm']['sdr'].append(SDR_irm.cpu().numpy().ravel())
        results['irm']['sar'].append(SAR_irm.cpu().numpy().ravel())
        results['irm']['sir'].append(SIR_irm.cpu().numpy().ravel())

        results['ibm']['sdr'].append(SDR_ibm.cpu().numpy().ravel())
        results['ibm']['sar'].append(SAR_ibm.cpu().numpy().ravel())
        results['ibm']['sir'].append(SIR_ibm.cpu().numpy().ravel())

        for key in results:
            r = results[key]
            r['instrument'].append(np.stack([data['instrument1'],data['instrument2']],-1).ravel())
            for key in ['instrument1','instrument2']:
                r[key].append(np.array(data[key])[:,None].repeat(2,1).ravel())

            if 'delta_time' in data:
                r['delta_time'].append(np.array(data['delta_time'])[:,None].repeat(2,1).ravel())

            if 'pitch1' in data:
                r['pitch'].append(np.stack([data['pitch1'],data['pitch2']],-1).ravel())
                r['delta_pitch'].append((np.array(data['pitch1'])-np.array(data['pitch2']))[:,None].repeat(2,1).ravel())

        print("Batch {}/{}".format(batch_idx+1,len(test_loader)))
        # if batch_idx >= 5: 
            # break

for key in results:
    r = results[key]
    for k in list(r):
        if len(r[k])  > 0:
            r[k] = np.concatenate(r[k])
        else:
            r.pop(k)

    df = pd.DataFrame(r)
    df.to_json(path_out / 'results_{}.json'.format(key),orient='index', indent=2)

    plt.figure(figsize=(5,8))
    sns.boxplot(data=df)
    plt.savefig(path_out / "boxplot_{}.png".format(key),dpi=200)
