#%%
import numpy as np
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
import os
import argparse
import torchaudio
import torch
import argparse
import librosa

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

parser = argparse.ArgumentParser(description='DSAE Example')
parser.add_argument('--data-path', type=str, default='data/nsynth/nsynth-test', metavar='N',
                    help='input batch size for training (default: 512)')
args = parser.parse_args()

data_path = Path(args.data_path)
audio_path = 'audio'
input_path = data_path.joinpath(audio_path)
output_path =data_path.joinpath("wav16")
os.makedirs(output_path,exist_ok=True)

input_suffix = '.wav'
output_suffix = '.wav'

walker = walk_files(input_path, suffix=input_suffix, prefix=True)
_walker = list(walker)
n_files = len(_walker)
for n in range(n_files):
    filepath = _walker[n][0]
    s,sr = torchaudio.load(filepath)
    if s.shape[0] == 1:
        s = s[0]
    x = librosa.effects.trim(s.numpy(),top_db=90)[0]
    x = torch.from_numpy(x)
    if x.ndim ==1:
        x = x.view(1,-1)

    path_name = filepath.split(str(input_path))[-1]
    path_name = path_name.split(input_suffix)[0]
    path_name = path_name + output_suffix
    output_name = str(output_path) + path_name
    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    torchaudio.save(output_name,x,sr)
 
    if n % 100 == 0:
        print("File {} / {}".format(n+1,n_files))
        print(output_name)

# %%
