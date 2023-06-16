#%%
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd
import numpy as np
import sines
from pathlib import Path
import json

plt.style.use('gibbs.mplstyles.latex')

path = Path("/Users/julian/Documents/Code/Python/Results/SVAE/ASA")
fname = path / 'results.json'
path_out = path / "imgs"
os.makedirs(path_out,exist_ok=True)

df = pd.read_json(fname,orient='index')

df.rename(columns={'sdr': 'SI-SDR', 'sir': 'SIR', 'sar': 'SAR'},inplace=True)

#%%
plt.figure(figsize=(3.5,2.5))
sns.boxplot(data=df,x='SI-SDR',y='physics',showfliers=False,whis=0,notch=True)
plt.ylabel(None)
plt.tight_layout()
plt.savefig(path_out / "physics_sdr.png")
#%%
plt.figure(figsize=(4,2.5))
ax = sns.lineplot(data=df,x='delta_time',y='SI-SDR',hue='physics1')
plt.xlabel(r"$\Delta$ time (s)")
plt.xlim(0,1)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.legend_.set_title(None)
plt.tight_layout()
plt.savefig(path_out / "time_sdr.png")
# %%
plt.figure(figsize=(4,2.5))
ax = sns.lineplot(data=df,x='delta_semitone',y='SI-SDR',hue='physics1')
plt.xlabel(r"$\Delta$ semitone")
plt.xlim(0,12)
plt.xticks(np.arange(13))
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.legend_.set_title(None)
plt.tight_layout()
plt.savefig(path_out / "semitone_sdr.png")

# %%
df_mean = df.filter(['delta_semitone','delta_time','SI-SDR','SIR','SAR'])
df_mean = df_mean.groupby(['delta_semitone','delta_time']).mean().reset_index()

#* Plot also for sustain vs vibrato, maybe the time diff doesn't matter as much?
# %%
x = df_mean['delta_time'].values
y = df_mean['delta_semitone'].values
semi = np.arange(y.min(),y.max()+1)
# %%
for measure in ['SI-SDR','SIR','SAR']:
    temp = df_mean[measure].values.reshape(13,-1)
    plt.figure(figsize=(3.5,2.5))
    plt.imshow(temp,aspect='auto',origin='lower',extent=[x.min(),x.max(),y.min(),y.max()+1],cmap='plasma',interpolation='none')
    # plt.xticks(time+.5,time)
    plt.yticks(semi+.5,semi)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(measure)
    plt.xlabel(r"$\Delta$ time (s)")
    plt.ylabel(r"$\Delta$ semitone")
    plt.tight_layout()
    plt.savefig(path_out / "time_semitone_{}.png".format(measure))
# %%
