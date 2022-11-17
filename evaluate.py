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
from utils.utils import optimal_permute
from collections import OrderedDict
import seaborn as sns
from utils.utils import torchToNumpy, mag2db
import glob
import re

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

def compute_measures_batch(estimates, targets, salt=1e-14):

    n = targets.shape[0]
    k = targets.shape[1]
    
    SDR = torch.zeros(n,k,device=estimates.device)
    SIR = torch.zeros(n,k,device=estimates.device)
    SAR = torch.zeros(n,k,device=estimates.device)

    Rss = torch.zeros(n,k,k,device=estimates.device)
    for ii in range(n):
        Rss[ii] = targets[ii] @ targets[ii].T
    
    for jj in range(k):
        SDR[:,jj], SIR[:,jj], SAR[:,jj] = compute_measures_batch_one(estimates=estimates[:,jj],targets=targets,Rss=Rss,j=jj,salt=salt)

    return SDR, SIR, SAR

def compute_measures_batch_one(estimates, targets, Rss, j, salt=1e-14):
    e_true = targets[:,j] 
    # Get the SIR
    e_res= estimates - e_true
    Rsr = (e_res.unsqueeze(1) * targets).sum(-1)
    b = torch.linalg.solve(Rss,Rsr)

    e_interf = (targets * b.unsqueeze(-1)).sum(1)
    e_artif = e_res - e_interf

    E_target = (e_true*e_true).sum(-1) + salt
    E_residual = (e_res*e_res).sum(-1) + salt
    E_interf = (e_interf*e_interf).sum(-1) + salt
    E_artif = (e_artif*e_artif).sum(-1) + salt

    SDR = 10*(E_target.log10() - E_residual.log10())
    SIR = 10*(E_target.log10() - E_interf.log10())
    SAR = 10*(E_target.log10() - E_artif.log10())
    return SDR, SIR, SAR

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

def dict_average(input):
    avg = {}
    for k,v in input.items():
        avg[k] = sum(v) / float(len(v))
    return avg

#%%
args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

#%%
output_path = os.path.join(args.output_directory,args.job_id)
output_audio_path = os.path.join(output_path,"audio")
os.makedirs(output_path,exist_ok=True)
os.makedirs(output_audio_path,exist_ok=True)

# Check the parameters of the model saved in that directory.
model_path = os.path.join(args.model_directory,args.job_id)

os.chdir(model_path)
model_name = glob.glob("*.pt")[0]
match = re.match(r'svae_([^_]+)_K(\d+)_L1_pooled_D(\d+)_C(\d+)-10.pt', model_name)

args.num_sources = int(match.group(2))
args.latent_dim = int(match.group(3))
args.compression = float(match.group(4)) * .1

print(args.num_sources)
print(args.latent_dim)
print(args.compression)

args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

nfft = 512
averaged = False
dim_data = (nfft//2+1,args.duration)

data_root = args.data_directory + '/RWC_dataset'
path_csv = 'test.csv'

if args.dataset == 'composite':
    data_root = args.data_directory
    path_csv = 'composite/test.csv'
if args.dataset == 'MUMS':
    data_root = args.data_directory + "/MUMS"
    path_csv = None

test_set = Audioset(root=data_root,path_csv=path_csv,augment_levels=args.augment_levels,duration=float(args.duration),silence=.5,starting=0,max_files=args.max_files)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,shuffle=True)

print("evaluating {} examples.".format(len(test_set)))

mixer = Mixer(n_sources=args.num_targets,average=averaged)
pipeline = AudioPipeline(n_fft=nfft,hop_length=128,compression=args.compression,n_time=args.n_time)
mixer = mixer.to(device)
pipeline = pipeline.to(device)

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

model.load_state_dict(load_state_dict(path=os.path.join(args.model_directory,args.job_id) + "/" + model_name))
model.to(device)
model.eval()

#%% EVALUATE
torch.manual_seed(args.seed)
np.random.seed(args.seed)
SiSDR = []
SiSIR = []
SiSAR = []
Idx = []

with torch.no_grad():
    for batch_idx, (data,_) in enumerate(test_loader):
        mix, sources = mixer(data.to(device))
        mix_stft = pipeline.forward_complex(mix)
        source_stft = pipeline.forward_complex(sources)

        data = mix_stft.abs()
        output = model(data)

        s_hat = output['recon_s']
        m_hat = create_masks(data,s_hat)
        estimates_stft = m_hat * torch.exp(1j*mix_stft.angle().unsqueeze(1))

        mix = pipeline.invert(mix_stft)
        sources = pipeline.invert(source_stft)
        estimates = pipeline.invert(estimates_stft)

        estimates = optimal_permute(estimates,sources)
        sources_invariant = make_scale_invariant(estimates=estimates,targets=sources)

        SDR, SIR, SAR = compute_measures_batch(estimates=estimates,targets=sources_invariant)

        _,idx = torch.sort(SDR)
        
        SiSDR.append(SDR)
        SiSIR.append(SIR)
        SiSAR.append(SAR)
        Idx.append(idx)

        print("Evaluating {} / {} ==> ".format(batch_idx+1,len(test_loader)))
    
#%%
SI_SDR = torch.cat(SiSDR).cpu()
SI_SIR = torch.cat(SiSIR).cpu()
SI_SAR = torch.cat(SiSAR).cpu()
Index = torch.cat(Idx).cpu()
print("SI-SDR = {:4.2f}".format(SI_SDR.mean()))
print("SI-SIR = {:4.2f}".format(SI_SIR.mean()))
print("SI-SAR = {:4.2f}".format(SI_SAR.mean()))

df = pd.DataFrame(dict(sdr=SI_SDR.mean().item(),sir=SI_SIR.mean().item(),sar=SI_SAR.mean().item()),index=[0])
df.to_csv(output_path + '/bss_summary.csv',index=False) 

df = pd.DataFrame(dict(SDR=SI_SDR.ravel(),SIR=SI_SIR.ravel(),SAR=SI_SAR.ravel(),Source=Index.ravel()))
df.to_csv(output_path + '/bss.csv',index=False) 
dfm = df.melt(id_vars=['Source'],value_vars=['SDR','SIR','SAR'])
sns.boxplot(data=dfm,x='variable',y='value',hue='Source',showfliers=False)
plt.grid()
plt.tight_layout()
plt.savefig(output_path + "/bss.png")
plt.close()
# %%
jet = cm.get_cmap('jet', 256)
newcolors = jet(np.linspace(0, 1, 256))
newcolors[0] = (0.0,0.0,0.0,1.0)
mycmp = ListedColormap(newcolors)

#%%
kwds = dict(cmap=mycmp,origin='lower',aspect='auto',interpolation='none',vmax=15)
pipeline.compression = 1.0
np.random.seed(1)
print(mix.shape)
idx = np.random.permutation(mix.shape[0])[:8]
for ii,n in enumerate(idx):
    output_audio_path_n = output_audio_path + "/ex_{}".format(ii+1)
    os.makedirs(output_audio_path_n,exist_ok=True)

    fig,ax = plt.subplots(args.num_sources,3,figsize=(9,3*args.num_sources))
    ax[0,0].imshow(mag2db(pipeline(mix[n])+1e-9).cpu(),**kwds)
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


    plt.savefig(output_audio_path_n + "/evaluation_spec.png")
    plt.tight_layout()
    plt.close()

    torchaudio.save(output_audio_path_n + "/mix.wav",mix[n].unsqueeze(0).cpu() / 2,16000)
    for i in range(sources.shape[1]):
        torchaudio.save(output_audio_path_n + "/source_{}.wav".format(i+1),sources[n,i].unsqueeze(0).cpu() / 2,16000)
    for i in range(estimates.shape[1]):
        torchaudio.save(output_audio_path_n + "/estimate_{}.wav".format(i+1),estimates[n,i].unsqueeze(0).cpu() / 2,16000)

    mix_down = torch.zeros((2,mix[n].shape[0]))
    nsrc = estimates.shape[1]
    for i in range(nsrc):
        panr = i/(nsrc-1.0)
        panl = 1-panr
        pan = torch.tensor([panl,panr])
        mix_down += estimates[n,i].view(1,-1).cpu() * pan.view(-1,1)

    torchaudio.save(output_audio_path_n + "/mixdown_estimates.wav",mix_down.cpu() / 2,16000)
