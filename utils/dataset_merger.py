#%%
import numpy as np
from pathlib import Path
import os
import argparse
import pandas as pd
import torchaudio


dataset = ['RWC','VCTK']
idxs = {'train': [], 'validate': [],'test': []}

data_path = Path('/home/jneri/projects/def-depalle/jneri/data')
output_path = data_path.joinpath("composite")
os.makedirs(output_path,exist_ok=True)

for key in idxs:
    for d in dataset:
        dataset_dir = data_path.joinpath(d)
    
        df = pd.read_csv(dataset_dir.joinpath(key+".csv"),sep='\t',header=None)
        df[0] = d + '/' + df[0].astype(str)
        idxs[key].extend(df[0].values.tolist())

        # df = pd.DataFrame({'path':files})
    df = pd.DataFrame(idxs[key])
    df.to_csv(output_path.joinpath(key + ".csv"), sep='\t',index=False,header=False)
    
# %%
