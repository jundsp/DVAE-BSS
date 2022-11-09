#%%
import numpy as np
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
# import scipy.signal as sig
import os
import argparse
# import pandas as pd
import torchaudio
import torch
# import samplerate

def walk_files(root, suffix, prefix=False, remove_suffix=False):
    root = os.path.expanduser(root)
    for dirName, subdirList, fileList in os.walk(root):
        for fname in fileList:
            if fname.lower().endswith(suffix):
                descr = fname[: -len(suffix)]
                if remove_suffix:
                    fname = fname[: -len(suffix)]
                if prefix:
                    fname = os.path.join(root, dirName,fname)
                yield fname, descr

data_path = Path('/home/jneri/projects/def-depalle/jneri/data/VCTK')
audio_path = 'wav48_silence_trimmed'
input_path = data_path.joinpath(audio_path)
output_path =data_path.joinpath("wav16k")
os.makedirs(output_path,exist_ok=True)

input_suffix = '.flac'
output_suffix = '.wav'

walker = walk_files(input_path, suffix=input_suffix, prefix=True)
_walker = list(walker)
n_files = len(_walker)

filepath = _walker[0][0]
s,sr = torchaudio.load(filepath)

orig_freq = sr
new_freq = 16000
resampler = torchaudio.transforms.Resample(orig_freq=orig_freq,new_freq=new_freq,resampling_method='kaiser_window')

for n in range(n_files):
    filepath = _walker[n][0]
    s,sr = torchaudio.load(filepath)
    # s = s[0]
    # x = samplerate.resample(s.numpy(),ratio=16000/44100,converter_type='sinc_medium')
    # x = torch.from_numpy(x).view(1,-1)
    x = resampler(s)

    path_name = filepath.split(str(input_path))[-1]
    path_name = path_name.split(input_suffix)[0]
    path_name = path_name + output_suffix
    output_name = str(output_path) + path_name
    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    torchaudio.save(output_name,x,new_freq)
 
    if n % 100 == 0:
        print("File {} / {}".format(n+1,n_files))
        print(output_name)

# %%
