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
from utils.utils import torchToNumpy, mag2db, panning
import glob
import re
import shutil

#* Make a class / module to process the bss eval measures.

def save_audio_batch(path,mix,sources,estimates,starting_index):
    N = mix.shape[0]
    for n in range(N):
        output_audio_path_n = path + "/audio_{}".format(starting_index)
        os.makedirs(output_audio_path_n,exist_ok=True)

        torchaudio.save(output_audio_path_n + "/mix.wav",mix[n].unsqueeze(0).cpu() / 2,16000)
        for i in range(sources.shape[1]):
            torchaudio.save(output_audio_path_n + "/source_{}.wav".format(i+1),sources[n,i].unsqueeze(0).cpu(),16000)
        for i in range(estimates.shape[1]):
            torchaudio.save(output_audio_path_n + "/estimate_{}.wav".format(i+1),estimates[n,i].unsqueeze(0).cpu(),16000)

        starting_index += 1
    return starting_index

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
    I = torch.eye(k,device=estimates.device)*1e-12
    for ii in range(n):
        Rss[ii] = targets[ii] @ targets[ii].T + I
    
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

if __name__ == "__main__":
    #%%
    save_audio_files = False

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #%%
    output_path = os.path.join(args.output_directory,args.job_id)
    output_path = os.path.join(output_path,"t{}".format(args.num_targets))
    output_audio_path = os.path.join(output_path,"audio")
    os.makedirs(output_path,exist_ok=True)
    os.makedirs(output_audio_path,exist_ok=True)

    # Check the parameters of the model saved in that directory.
    model_path = os.path.join(args.model_directory,args.job_id)

    try:
        shutil.copyfile(model_path + '/configuration.txt', output_path+'/configuration.txt')
    except:
        print("No config file found in model directory, or problem copying.")

    os.chdir(model_path)
    model_name = glob.glob("*.pt")[0]
    match = re.match(args.model_type+r'_([^_]+)_S(\d+)_K(\d+)_L1_pooled_D(\d+)_C(\d+)-10.pt', model_name)

    num_train_targets = int(match.group(2))
    args.num_sources = int(match.group(3))
    args.latent_dim = int(match.group(4))
    args.compression = int(match.group(5)) * .1

    print(args.num_sources)
    print(args.latent_dim)
    print(args.compression)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

    nfft = 512
    averaged = False
    dim_data = (nfft//2+1,int(args.n_time))

    data_root = args.data_directory + '/RWC_dataset'
    path_csv = 'test.csv'

    if args.dataset == 'composite':
        data_root = args.data_directory
        path_csv = 'composite/test.csv'
    if args.dataset == 'MUMS':
        data_root = args.data_directory + "/MUMS"
        path_csv = None
    if args.dataset == 'nsynth':
        data_root = args.data_directory + "/nsynth-test/audio"
        path_csv = None
    if args.dataset == 'MUSDB':
        data_root = args.data_directory + "/MUSDB"
        path_csv = 'test.csv'

    test_set = Audioset(root=data_root,path_csv=path_csv,augment_levels=args.augment_levels,duration=float(args.duration),silence=.5,starting=0,max_files=args.max_files)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,shuffle=True)

    print("evaluating {} examples.".format(len(test_set)))

    mixer = Mixer(n_sources=args.num_targets,average=averaged)
    pipeline = AudioPipeline(n_fft=nfft,hop_length=128,compression=args.compression,n_time=args.n_time,output_realimag=args.complex_version)
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

    if args.model_type == 'vae':
        dimx = int(dim_data[0]*dim_data[1])
        model = VAE(dimx=dimx,dimz=int(args.latent_dim),n_sources=int(args.num_sources))
    else:
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
                activate_final_layer=args.activate_final_layer
                )
                    
        if args.complex_version:
            model = SVAE(mag_output=True,**kwds_model)
        else:
            model = DCAE(**kwds_model)

    model.load_state_dict(load_state_dict(path=os.path.join(args.model_directory,args.job_id) + "/" + model_name))
    model.to(device)
    model.eval()

    #%% EVALUATE
    #TODO: SAVE INDIV. SOURCE FILES.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    SiSDR = []
    SiSIR = []
    SiSAR = []
    Idx = []

    starting_index = 0

    spectral_averages = 0.0


    with torch.no_grad():
        for batch_idx, (data,_) in enumerate(test_loader):
            mix, sources = mixer(data.to(device))
            mix_stft = pipeline.forward_complex(mix)
            source_stft = pipeline.forward_complex(sources)

            data = mix_stft.abs()
            if args.complex_version:
                input_data = torch.stack([mix_stft.real, mix_stft.imag],1)
            else:
                input_data = data
            output = model(input_data)

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

            spectral_averages += s_hat.sum(-1).mean(0)

            print("Evaluating {} / {} ==> ".format(batch_idx+1,len(test_loader)))

            if save_audio_files:
                starting_index = save_audio_batch(path=output_audio_path,estimates=estimates,sources=sources,mix=mix,starting_index=starting_index)

    spectral_averages = spectral_averages.cpu().numpy()
    spectral_averages /= len(test_loader)
    #%%
    SI_SDR = torch.cat(SiSDR).cpu()
    SI_SIR = torch.cat(SiSIR).cpu()
    SI_SAR = torch.cat(SiSAR).cpu()
    Index = torch.cat(Idx).cpu()
    print("SI-SDR = {:4.2f}".format(SI_SDR.mean()))
    print("SIR = {:4.2f}".format(SI_SIR.mean()))
    print("SAR = {:4.2f}".format(SI_SAR.mean()))

    df = pd.DataFrame(dict(compression=args.compression,sources=args.num_sources,latent_dim=args.latent_dim,targets=args.num_targets),index=[0])
    df.to_csv(output_path + '/config.csv',index=False) 

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

    kwds = dict(cmap=mycmp,origin='lower',aspect='auto',interpolation='none',vmax=15)
    pipeline.compression = 1.0
    np.random.seed(1)
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

        plt.tight_layout()
        plt.savefig(output_audio_path_n + "/evaluation_spec.png")
        plt.close()

        torchaudio.save(output_audio_path_n + "/mix.wav",mix[n].unsqueeze(0).cpu() / 2,16000)
        for i in range(sources.shape[1]):
            torchaudio.save(output_audio_path_n + "/source_{}.wav".format(i+1),sources[n,i].unsqueeze(0).cpu() / 2,16000)
        for i in range(estimates.shape[1]):
            torchaudio.save(output_audio_path_n + "/estimate_{}.wav".format(i+1),estimates[n,i].unsqueeze(0).cpu() / 2,16000)

        mix_down = torch.zeros((2,mix[n].shape[0]))
        nsrc = estimates.shape[1]
        pans  = panning(nsrc)

        print(pans)

        for i in range(nsrc):
            panr = i/(nsrc-1.0)
            panr = pans[i]
            panl = 1-panr
            pan = torch.tensor([panl,panr])
            mix_down += estimates[n,i].view(1,-1).cpu() * pan.view(-1,1)

        torchaudio.save(output_audio_path_n + "/mixdown_estimates.wav",mix_down.cpu() / 2,16000)


    np.save(file=output_path + "/spectral_averages",arr=spectral_averages)

    fig,ax = plt.subplots(args.num_sources,figsize=(4,2*args.num_sources))
    for i in range(args.num_sources):
        ax[i].plot(spectral_averages[i])
    plt.tight_layout()
    plt.savefig(output_path + "/spectral_averages.png")
    plt.close()