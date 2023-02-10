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

plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times",
    "axes.titlesize": "medium",
    "axes.labelsize": "medium",
})

path = Path("/Users/julian/Documents/Code/Python/Results/SVAE/RWC_test/sources/230122-13058067")
fname = path / 'results.json'
path_out = path / "imgs"
os.makedirs(path_out,exist_ok=True)

df = pd.read_json(fname,orient='index')

df.rename(columns={'sdr': 'SI-SDR', 'sir': 'SIR', 'sar': 'SAR'},inplace=True)

#%%
plt.figure(figsize=(3.5,2.5))
sns.boxplot(data=df,x='place',y='SI-SDR',showfliers=False,whis=0)
plt.ylabel(None)
plt.tight_layout()
plt.savefig(path_out / "place_sdr.png")
#%%
plt.figure(figsize=(4,2.5))
ax = sns.lineplot(data=df,x='delta_time',y='SI-SDR',hue='instrument1')
plt.xlabel(r"$\Delta$ time (s)")
plt.xlim(0,1)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.legend_.set_title(None)
plt.tight_layout()
plt.savefig(path_out / "time_sdr.png")
# %%
plt.figure(figsize=(4,2.5))
ax = sns.lineplot(data=df,x='delta_semitone',y='SI-SDR',hue='instrument1')
plt.xlabel(r"$\Delta$ semitone")
plt.xlim(0,12)
plt.xticks(np.arange(13))
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.legend_.set_title(None)
plt.tight_layout()
plt.savefig(path_out / "semitone_sdr.png")

# %%
df_mean = df.groupby(['instrument']).mean().reset_index()

#* Plot also for sustain vs vibrato, maybe the time diff doesn't matter as much?
# %%
x = df_mean['instrument1'].values
y = df_mean['instrument2'].values
# semi = np.arange(y.min(),y.max()+1)
# %%
for measure in ['SI-SDR','SIR','SAR']:
    temp = df_mean[measure].values
    plt.figure(figsize=(3.5,2.5))
    plt.imshow(temp,aspect='auto',origin='lower',cmap='plasma',interpolation='none')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(measure)
    plt.xlabel(r"instrument 1")
    plt.ylabel(r"instrument 2")
    plt.tight_layout()
    plt.savefig(path_out / "instrument_{}.png".format(measure))

# %%
