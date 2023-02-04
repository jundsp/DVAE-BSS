
import torch
import torchaudio
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from model import *
from dataset import *
from argparser import parser
from utils.utils import optimal_permute, ideal_masks, make_scale_invariant, compute_measures_batch
from collections import OrderedDict
import seaborn as sns
from utils.utils import torchToNumpy, mag2db, panning
import glob
import re
import shutil
import json

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

def dict_average(input):
    avg = {}
    for k,v in input.items():
        avg[k] = sum(v) / float(len(v))
    return avg

if __name__ == "__main__":
  
    save_audio_files = False

    args = parser.parse_args(args=[])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Check the parameters of the model saved in that directory.
    model_path = Path(os.path.join(args.model_directory,args.job_id))

    output_path = model_path / "results"
    # output_path = os.path.join(output_path,"t{}".format(args.num_targets))
    output_audio_path = os.path.join(output_path,"audio")
    os.makedirs(output_path,exist_ok=True)
    os.makedirs(output_audio_path,exist_ok=True)

    fname = model_path / "configuration.txt"
    with open(fname) as f:
        data = f.read()
    args = json.loads(data)
    entries_to_remove = ('data_directory', 'model_directory', 'job_id', 'batch_size', 'output_directory')
    for k in entries_to_remove:
        args.pop(k, None)
    parser.set_defaults(**args)
    args = parser.parse_args(args=[])
    args.batch_size = 8

    model_name = "svae_RWC_S2_K5_L1_pooled_D256_C5-10.pt"

    print(args.num_sources)
    print(args.latent_dim)
    print(args.compression)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

    nfft = 512
    averaged = False

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
    mixer = mixer.to(device)

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
                activate_final_layer=args.activate_final_layer
                )
                    
    model = DCAE(**kwds_model)

    model.load_state_dict(load_state_dict(path=os.path.join(args.model_directory,args.job_id) + "/" + model_name))
    model = Separator(model,compression=args.compression)
    model.to(device)
    model.eval()

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
            estimates = model(mix)

            estimates = optimal_permute(estimates,sources)
            sources_invariant = make_scale_invariant(estimates=estimates,targets=sources)

            SDR, SIR, SAR = compute_measures_batch(estimates=estimates,targets=sources_invariant)
            _,idx = torch.sort(SDR)
            
            SiSDR.append(SDR)
            SiSIR.append(SIR)
            SiSAR.append(SAR)
            Idx.append(idx)

            print("Evaluating {} / {} ==> ".format(batch_idx+1,len(test_loader)))

            if save_audio_files:
                starting_index = save_audio_batch(path=output_audio_path,estimates=estimates,sources=sources,mix=mix,starting_index=starting_index)
            if batch_idx > 1:
                break


    SI_SDR = torch.cat(SiSDR).cpu()
    SI_SIR = torch.cat(SiSIR).cpu()
    SI_SAR = torch.cat(SiSAR).cpu()
    Index = torch.cat(Idx).cpu()
    print("SI-SDR = {:4.2f}".format(SI_SDR.mean()))
    print("SIR = {:4.2f}".format(SI_SIR.mean()))
    print("SAR = {:4.2f}".format(SI_SAR.mean()))

    df = pd.DataFrame(dict(compression=args.compression,sources=args.num_sources,latent_dim=args.latent_dim,targets=args.num_targets),index=[0])
    df.to_csv(output_path / 'config.csv',index=False) 

    df = pd.DataFrame(dict(sdr=SI_SDR.mean().item(),sir=SI_SIR.mean().item(),sar=SI_SAR.mean().item()),index=[0])
    df.to_csv(output_path / 'bss_summary.csv',index=False) 

    df = pd.DataFrame(dict(SDR=SI_SDR.ravel(),SIR=SI_SIR.ravel(),SAR=SI_SAR.ravel(),Source=Index.ravel()))
    df.to_csv(output_path / 'bss.csv',index=False)