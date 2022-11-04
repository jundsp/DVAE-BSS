#%%
import numpy as np
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.signal as sig
import os
import argparse
import pandas as pd
import torchaudio

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

data_path = Path('/home/jneri/projects/def-depalle/jneri/data').resolve()
dataset = 'RWCS'
input_path = data_path.joinpath(dataset)
output_path =data_path.joinpath("rwc_dataset")


walker = walk_files(input_path, suffix='.wav', prefix=True)
_walker = list(walker)
n_files = len(_walker)

# for n in range(n_files):
#     filepath = _walker[n]
#     s,sr = torchaudio.read(filepath)
    
#     print(n)

for n in range(n_files):
    filepath = _walker[n][0]
    s,sr = torchaudio.load(filepath)
        
    if n % 200 == 0:
        print(n)

# #%%
# test = 5000
# validation = 10000
# train = n_files - (validation+test)
# idx = np.random.permutation(n_files)

# splits = np.cumsum([train,validation,test])

# idx_train = idx[:splits[0]]
# idx_validate = idx[splits[0]:splits[1]]
# idx_test = idx[splits[1]:splits[2]]

# idxs = {'train': idx_train, 'validate': idx_validate,'test':idx_test}

# os.makedirs(output_path,exist_ok=True)

# for key in idxs:
#     files = []
#     for n in np.sort(idxs[key]):
#         path = _walker[n][0]
#         folders = path.split("data")[-1][1:]
#         files.append(folders)

#     df = pd.DataFrame({'path':files})
#     df.to_csv(output_path.joinpath(key + ".csv"), sep='\t',index=False,header=False)
# # %%
