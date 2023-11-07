import torch
import os
import matplotlib.pyplot as plt
from model import *
from dataset import *
from argparser import parser
from utils.utils import optimal_permute, ideal_masks, compute_measures_batch, make_scale_invariant
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import FastICA
from NMFtoolbox.initTemplates import initTemplates
from NMFtoolbox.initActivations import initActivations
from NMFtoolbox.NMFD import NMFD
from NMFtoolbox.alphaWienerFilter import alphaWienerFilter

def compute_measures(estims,targets,separator):
    estims = separator.postprocess(estims)
    estims = estims.reshape(targets.shape)
    estims,_ = optimal_permute(estims,targets)
    targets_invariant = make_scale_invariant(estimates=estims,targets=targets)
    sdr,sir,sar = compute_measures_batch(estimates=estims,targets=targets_invariant)
    return dict(sdr=sdr,sir=sir,sar=sar), estims

class NMF():
    def __init__(self, num_bins,num_frames,num_comp=2, num_templates=8, num_iter=50,n_fft=2048, hop_length=512):
        # generate initial guess for templates
        paramTemplates = dict()
        paramTemplates['deltaF'] = 44100/n_fft
        paramTemplates['numComp'] = num_comp
        paramTemplates['numBins'] = num_bins
        paramTemplates['numTemplateFrames'] = num_templates
        initW = initTemplates(paramTemplates,'random')

        # generate initial activations
        paramActivations = dict()
        paramActivations['numComp'] = num_comp
        paramActivations['numFrames'] = num_frames
        initH = initActivations(paramActivations,'uniform')

        # NMFD parameters
        paramNMFD = dict()
        paramNMFD['numComp'] = num_comp
        paramNMFD['numFrames'] = num_frames
        paramNMFD['numIter'] = num_iter
        paramNMFD['numTemplateFrames'] = num_templates
        paramNMFD['initW'] = initW
        paramNMFD['initH'] = initH
        self.parameters = paramNMFD

    def __call__(self,x):
        return self.forward(x)

    def __repr__(self):
        string = 'NMF object \n Components ==> {}\n Templates ==> {}\n Iterations ==> {}\n'.format(self.parameters['numComp'],self.parameters['numTemplateFrames'],self.parameters['numIter'])
        return string

    def forward(self,A):
        nmfdW, nmfdH, nmfdV, divKL, _ = NMFD(A, self.parameters)
        nmfdA, _ = alphaWienerFilter(A, nmfdV, 1.0)
        return nmfdA,nmfdW, nmfdH, nmfdV, divKL

def get_separate(W,H):
    W = W.unsqueeze(-1).permute(1,2,0)
    H = H.unsqueeze(-1)
    X_sep = (H @ W).transpose(-2,-1)
    return X_sep



args = parser.parse_args(args=[])

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

# torch.manual_seed(123)

test_set = NSYNTH(root=path_data)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64,shuffle=False)
separator = Separator(model=None,compression=1.0).to(device)

# LOAD MODEL
print(args)
print("evaluating {} examples.".format(len(test_set)))

data = test_set.__getitem__(0)
mix = data['mix'][[0]]
mix_stft = separator.preprocess(mix)

# NMF
nmf = NMF(num_bins=mix_stft[0].shape[0],num_frames=mix_stft[0].shape[-1],num_comp=2,num_iter=30)
ica = FastICA(n_components=2)


results = dict(
        NMF=dict(sdr=[],sir=[],sar=[],instrument=[],instrument1=[],instrument2=[],delta_time=[],pitch=[],delta_pitch=[]),
        ICA=dict(sdr=[],sir=[],sar=[],instrument=[],instrument1=[],instrument2=[],delta_time=[],pitch=[],delta_pitch=[])
    )

for batch_idx, batch in enumerate(test_loader):
    mix = batch['mix'][:,0]
    sources = batch['stems']

    mix_stft = separator.preprocess(mix)
    for i in range(mix.shape[0]):
        data = mix_stft[i].abs().numpy()
        phase_part = mix_stft[i].angle()
        source = sources[[i]]
        
        A = nmf(data)[0]
        recons_nmf = torch.tensor(A).float()
        recons_nmf = recons_nmf * torch.exp(1j*phase_part.unsqueeze(0))

        bss_nmf, estims_nmf = compute_measures(recons_nmf,source,separator)
        for k in bss_nmf:
            results['NMF'][k].append(bss_nmf[k].cpu().numpy().ravel())
        
        # ICA
        W = ica.fit_transform(data)
        H = ica.mixing_
        W = torch.tensor(W)
        H = torch.tensor(H).T
        recons_ica = get_separate(W,H).float()
        recons_ica = recons_ica * torch.exp(1j*phase_part.unsqueeze(0))
        
        bss_ica, estims_ica = compute_measures(recons_ica,source,separator)
        for k in bss_ica:
            results['ICA'][k].append(bss_ica[k].cpu().numpy().ravel())
            
    for key in results:
        r = results[key]
        r['instrument'].append(np.stack([batch['instrument1'],batch['instrument2']],-1).ravel())
        for key in ['instrument1','instrument2']:
            r[key].append(np.array(batch[key])[:,None].repeat(2,1).ravel())

        if 'delta_time' in batch:
            r['delta_time'].append(np.array(batch['delta_time'])[:,None].repeat(2,1).ravel())

        if 'pitch1' in batch:
            r['pitch'].append(np.stack([batch['pitch1'],batch['pitch2']],-1).ravel())
            r['delta_pitch'].append((np.array(batch['pitch1'])-np.array(batch['pitch2']))[:,None].repeat(2,1).ravel())
        
    print("Batch {}/{}".format(batch_idx+1,len(test_loader)))


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