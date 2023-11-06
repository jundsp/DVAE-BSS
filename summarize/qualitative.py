#%%
import torch
import torchaudio
import os
import matplotlib.pyplot as plt
from argparser import parser
from pathlib import Path
import glob
import sines
import numpy as np

plt.style.use("gibbs.mplstyles.latex")

def save_spectrogram(filename,x,sr,vmin=-140,vmax=-20):
    S,t,f = sines.stft(x,sr=sr)
    # S_abs = np.abs(S)
    # S[S_abs==0] = S_abs[S_abs>0].min()

    fig,ax = plt.subplots(1,figsize=(3,2))
    sines.plot_spectrogram(S,t,f,ax=ax,vmin=vmin,vmax=vmax)
    ax.set_xlabel(r"time (s)")
    ax.set_ylabel(r"frequency (Hz)")
    plt.tight_layout()
    plt.savefig(filename,bbox_inches="tight")
    plt.close()

def save_spectrogram_grid(filename,mix,targets,estims,vmin=-140,vmax=-20):
    fig,ax = plt.subplots(3,2,figsize=(5.5,5.0),sharex=True,sharey=True)
    S,t,f = sines.stft(mix[0],sr=sr)
    is_on = np.nonzero(np.sum(np.abs(S),0)>0)[0]
    t_start = np.min(is_on)
    t_end = np.max(is_on)
    S = S[:,t_start:t_end]
    t = t[t_start:t_end]
    t -= t.min()
    sines.plot_spectrogram(S,t,f/1e3,ax=ax[0,0],vmin=vmin,vmax=vmax)
    

    for i in range(targets.shape[0]):
        S,t,f = sines.stft(targets[i],sr=sr)
        S = S[:,t_start:t_end]
        t = t[t_start:t_end]
        t -= t.min()
        sines.plot_spectrogram(S,t,f/1e3,ax=ax[1,i],vmin=vmin,vmax=vmax)
        
        

    for i in range(targets.shape[0]):
        S,t,f = sines.stft(estims[i],sr=sr)
        S = S[:,t_start:t_end]
        t = t[t_start:t_end]
        t -= t.min()
        sines.plot_spectrogram(S,t,f/1e3,ax=ax[2,i],vmin=vmin,vmax=vmax)

    ax[0,0].set_title("Mixture",y=.97)
    ax[0,1].remove()
    for i in range(3):
        ax[i,0].set_ylabel('frequency (kHz)')
    for i in range(2):
        ax[-1,i].set_xlabel('time (s)')
        ax[1,i].set_title("Target {}".format(i+1),y=.97)
        ax[2,i].set_title("Estimate {}".format(i+1),y=.97)
    plt.tight_layout()
    plt.savefig(filename,bbox_inches="tight")
    plt.close()
    
args = parser.parse_args(args=[])

args.output_directory = "/Users/julian/Documents/Code/output/svae/RWC_test/ablation_targets"
args.output_directory = "/Volumes/JulianSSD/research/svae-bss/evaluation/NSYNTH_test/ablation_targets"
args.output_directory = "/Users/julian/Documents/PhD/2023_IEEE_DVAE_BSS_Neri/imgs/results/svae_qualitative_composite"
args.job_id = "231017-22161497"
path_out = Path(args.output_directory) / args.job_id

#%%
path_audio_out = path_out / "audio"

folders = sorted(glob.glob(str(path_audio_out / "*"),recursive=True))


#%%
for f in folders:
    print(f)
    path_out_i = Path(f)
    path_imgs_i = path_out_i / "imgs"
    os.makedirs(path_imgs_i,exist_ok=True)

    mix, sr = torchaudio.load(path_out_i / "mixture.wav")
    targets, sr = torchaudio.load(path_out_i / "targets.wav")
    estims, sr =torchaudio.load(path_out_i / "estimates.wav")

    # save_spectrogram(path_imgs_i / "mix.pdf",mix[0],sr)
    # for i in range(targets.shape[0]):
    #     save_spectrogram(path_imgs_i / "target_{}.pdf".format(i+1),targets[i],sr)
    #     save_spectrogram(path_imgs_i / "estimate_{}.pdf".format(i+1),estims[i],sr)

    save_spectrogram_grid(path_imgs_i/"grid.pdf",mix,targets,estims)

# %%
