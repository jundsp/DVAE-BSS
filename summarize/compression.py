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
plt.rcParams['figure.dpi'] = 150
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times",
    "axes.titlesize": "medium",
    "axes.labelsize": "medium",
})
# sns.set()

root = "/Users/julian/Documents/Code/Python/Results/SVAE/RWC/compression/"
output_directory = root + "/results"
os.makedirs(output_directory,exist_ok=True)

csv_name = glob.glob(root + "*/*/*/bss.csv")
spec_avg = []
comp = []
DF = []
for ii,cc in enumerate(csv_name):
    config = os.path.dirname(cc) + "/config.csv"
    df_1 = pd.read_csv(config)
    df = pd.read_csv(cc)
    df['compression'] = round(df_1['compression'].item(),1)
    df['sources'] = round(df_1['sources'].item(),1)
    df.rename(columns={'Source': 'estimate'},inplace=True)
    df['estimate'] +=1

    DF.append(df)

    fname = os.path.dirname(cc) + "/spectral_averages.npy"
    if df_1['sources'].item() ==5:
        spec_avg.append(np.load(fname))
        comp.append(df_1['compression'].item())

spec_avg = np.stack(spec_avg,0)
comp = np.stack(comp,0)


#%%
df = pd.concat(DF)
df = df.reset_index()
df.rename(columns={'SDR': 'SI-SDR'},inplace=True)

dfm = df.melt(['sources','compression','index','estimate'], var_name='metric', value_name='vals')

#%%
cvals = [1, .5, .3]
fig,ax = plt.subplots(3,spec_avg.shape[1],figsize=(10,3),sharex=True,sharey=False)
for jj,c in enumerate(cvals):
    Sdb = sines.mag2db(spec_avg[comp==c][0])
    sorted = np.argsort(Sdb.sum(-1))
    Sdb = Sdb[sorted]
    f = np.arange(Sdb.shape[-1]) / 512 * 16000 / 1e3
    for i in range(Sdb.shape[0]):
        ax[jj,i].plot(f,Sdb[i],'k',linewidth=1)
        ax[jj,i].set_xlim(0,f.max())
        ax[jj,i].set_ylim(Sdb.min(),Sdb.max())
        if jj == 0:
            ax[jj,i].set_title(r"$S_{}$".format(i+1))
        if jj == 2:
            ax[-1,i].set_xlabel('frequency (kHz)')
    ax[jj,0].set_ylabel(r"$c={}$".format(c))
plt.tight_layout()
plt.savefig(output_directory + "/spectrum_averages.pdf")
plt.savefig(output_directory + "/spectrum_averages.png",dpi=200)
#%%
fig,ax = plt.subplots(1,3,figsize=(10,3))
PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'grey'},
    'medianprops':{'color':'blue'},
    'whiskerprops':{'color':'blue'},
    'capprops':{'color':'black'}
}
kwds_box = dict(showcaps=False,showfliers=False,whis=0,**PROPS)
sns.boxplot(data=df,x="compression",y='SI-SDR',ax=ax[0],**kwds_box)
sns.boxplot(data=df,x="compression",y='SIR',ax=ax[1],**kwds_box)
sns.boxplot(data=df,x="compression",y='SAR',ax=ax[2],**kwds_box)
plt.tight_layout()
plt.savefig(output_directory + "/boxplot_nofill.pdf")
plt.savefig(output_directory + "/boxplot_nofill.png",dpi=200)
#%%
fig,ax = plt.subplots(1,3,figsize=(10,3))
kwds_box = dict(showfliers=False)
sns.boxplot(data=df,x="compression",y='SI-SDR',ax=ax[0],**kwds_box)
sns.boxplot(data=df,x="compression",y='SIR',ax=ax[1],**kwds_box)
sns.boxplot(data=df,x="compression",y='SAR',ax=ax[2],**kwds_box)
plt.tight_layout()
plt.savefig(output_directory + "/boxplot.pdf")

# %%
fig,ax = plt.subplots(1,3,figsize=(10,3))
sns.boxplot(data=df,hue="compression",y='SI-SDR',ax=ax[0],showfliers=False,x='estimate')
sns.boxplot(data=df,hue="compression",y='SIR',ax=ax[1],showfliers=False,x='estimate')
sns.boxplot(data=df,hue="compression",y='SAR',ax=ax[2],showfliers=False,x='estimate')
ax[0].legend([],[], frameon=False)
ax[1].legend([],[], frameon=False)

plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
plt.tight_layout()
plt.savefig(output_directory + "/boxplot_estims.pdf")

# %%
fig,ax = plt.subplots(1,3,figsize=(10,3))
sns.boxplot(data=df,x="compression",y='SI-SDR',ax=ax[0],showfliers=False,hue='estimate')
sns.boxplot(data=df,x="compression",y='SIR',ax=ax[1],showfliers=False,hue='estimate')
sns.boxplot(data=df,x="compression",y='SAR',ax=ax[2],showfliers=False,hue='estimate')
ax[0].legend([],[], frameon=False)
ax[1].legend([],[], frameon=False)

plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
plt.tight_layout()
plt.savefig(output_directory + "/boxplot_compression.pdf")

# %%
fig,ax = plt.subplots(1,3,figsize=(10,3))
kwds_line = dict(estimator='mean',errorbar=('sd', .5),color='k')
sns.pointplot(data=df,x="compression",y='SI-SDR',ax=ax[0],**kwds_line)
sns.pointplot(data=df,x="compression",y='SIR',ax=ax[1],**kwds_line)
sns.pointplot(data=df,x="compression",y='SAR',ax=ax[2],**kwds_line)
plt.tight_layout()
ax[1].set_title('.5 Standard Deviation')
plt.savefig(output_directory + "/lineplot_sd.pdf")


#%%
fig,ax = plt.subplots(1,3,figsize=(10,3))
kwds_line = dict(estimator='mean',errorbar=('ci', 95),hue='sources')
sns.pointplot(data=df,x="compression",y='SI-SDR',ax=ax[0],**kwds_line)
sns.pointplot(data=df,x="compression",y='SIR',ax=ax[1],**kwds_line)
sns.pointplot(data=df,x="compression",y='SAR',ax=ax[2],**kwds_line)
# ax[1].set_title(' 95% confidence interval')
plt.tight_layout()
plt.savefig(output_directory + "/lineplot_ci.pdf")
plt.savefig(output_directory + "/lineplot_ci.png",dpi=200)
# %%
