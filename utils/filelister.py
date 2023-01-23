#%%
import numpy as np
from pathlib import Path
import os
import pandas as pd

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

data_path = Path('/home/jneri/projects/def-depalle/jneri/data/MUSDB')
audio_path = 'train'
input_path = data_path.joinpath(audio_path)
output_path = data_path


walker = walk_files(input_path, suffix='.wav', prefix=True)
_walker = list(walker)
n_files = len(_walker)

# #%%
test = 0
validation = 0
train = n_files - (validation+test)
idx = np.random.permutation(n_files)


splits = np.cumsum([train,validation,test])

idx_train = idx[:splits[0]]
idx_validate = idx[splits[0]:splits[1]]
idx_test = idx[splits[1]:splits[2]]

idxs = {'train': idx_train, 'validate': idx_validate,'test':idx_test}

for key in idxs:
    files = []
    for n in np.sort(idxs[key]):
        path = _walker[n][0]
        folders = path.split(str(data_path))[-1][1:]
        files.append(folders)

    df = pd.DataFrame({'path':files})
    df.to_csv(output_path.joinpath(key + ".csv"), sep='\t',index=False,header=False)
    
# %%


# data_path = Path('/home/jneri/projects/def-depalle/jneri/data/MUMS')
# audio_path = 'wav16'
# input_path = data_path.joinpath(audio_path)
# output_path = data_path


# root = os.path.expanduser(input_path)
# for dirName, subdirList, fileList in os.walk(root,topdown=False):
#     for f in fileList:
#         old_path = os.path.join(dirName,f)
#         new_name = f.replace(" ", "_")
#         new_path = os.path.join(dirName,new_name)
#         os.rename(old_path, new_path)

#     roots = os.path.dirname(dirName)
#     temp = os.path.basename(dirName)
#     new_name = temp.replace(" ", "_")
#     new_path = os.path.join(roots,new_name)
#     os.rename(dirName, new_path)