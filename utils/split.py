#%%
import numpy as np
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.signal as sig
import os
import argparse

parser = argparse.ArgumentParser(description='DSAE Example')
parser.add_argument('--starting', type=int, default=0, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('--nfiles', type=int, default=4000, metavar='N',
                    help='number of epochs to train (default: 10)')

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

def split_wav(x, fs, min_dur=.1, thresh=-100,L=500):

    y = np.abs(x)
    y /= y.max()

    b = np.ones(L) / L
    y = sig.lfilter(b,1, y)

    min_length = min_dur * fs

    ydb = 20*np.log10(y)
    low = (ydb < thresh).astype(int)
    low[0] = 1
    low[-1] = 1
    changepoints = np.diff(low)
    turnon = np.nonzero(changepoints == -1)[0]
    turnoff = np.nonzero(changepoints == 1)[0]

    split = []
    for i in range(len(turnon)):
        if turnoff[i] - turnon[i] > min_length:
            stem = x[turnon[i]:turnoff[i]]
            split.append(stem)

    return split

# #%%
args = parser.parse_args()

data_path = Path('data').resolve()
dataset = 'RWC'
input_path = data_path.joinpath(dataset)
output_path =data_path.joinpath(dataset + '_stems')

walker = walk_files(input_path, suffix='.wav', prefix=True)
_walker = list(walker)[args.starting:args.starting+args.nfiles]
n_files = len(_walker)

output_path.mkdir(parents=True, exist_ok=True)

#%%

# for jj, (path,fname) in enumerate(_walker):
#     print("{} / {}".format(jj+1+args.starting, n_files))
#     x,fs = sf.read(path)
#     split = split_wav(x,fs)

#     folders = path.split("RWC")[-1][:-4][1:]

#     folder_save = output_path.joinpath(folders)

#     os.makedirs(folder_save,exist_ok=True)

#     for i,stem in enumerate(split):
#         sf.write(file=folder_save.joinpath(fname + '_{}'.format(i+1) + '.wav'),data=stem,samplerate=fs)
        
# %%
for jj, (path,fname) in enumerate(_walker):
    # print("{} / {}".format(jj+1+args.starting, n_files))
    
    folders = path.split("RWC")[-1][:-4][1:]
    
    if folders.split('/')[0] == '372':
        print(folders)
        folder_save = output_path.joinpath(folders)
        os.makedirs(folder_save,exist_ok=True)
        
        x,fs = sf.read(path)
        split = split_wav(x,fs)
        for i,stem in enumerate(split):
            sf.write(file=folder_save.joinpath(fname + '_{}'.format(i+1) + '.wav'),data=stem,samplerate=fs)