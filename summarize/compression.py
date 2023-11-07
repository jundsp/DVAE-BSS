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
import json

from pathlib import Path

plt.style.use('gibbs.mplstyles.latex')

# sns.set()

root = Path("/Users/julian/Documents/Code/Python/Results/SVAE/RWC_test/compression/")
output_directory = root / "results"
os.makedirs(output_directory,exist_ok=True)
  
compare_along = "compression"
#%%
json_name = root.glob("*/results.json")
spec_avg = []
comp = []
DF = []
for ii,cc in enumerate(json_name):
    df = pd.read_json(cc,orient="index")
    df['place'] = 2-df['place']

    fname = cc.parents[0] / "configuration.txt"
    with open(fname) as f:
        data = f.read()
    js = json.loads(data)
    df['compression'] = round(js['compression'],2)
    df['variance'] = round(js['variance'],3)
    df['sources'] = round(js['num_sources'],1)
    df['targets'] = round(js['num_targets'],1)

    df = df[(df['instrument1'] != 'glockenspiel') & (df['instrument2'] != 'glockenspiel')]

    DF.append(df)

#%%
df = pd.concat(DF)
df = df.reset_index()
df.rename(columns={'sdr': 'SI-SDR','sir': 'SIR','sar': 'SAR'},inplace=True)

dfm = df.melt(['sources','compression','index','place'], var_name='metric', value_name='vals')

#%%
# cvals = [1, .5, .3]
# fig,ax = plt.subplots(3,spec_avg.shape[1],figsize=(10,3),sharex=True,sharey=False)
# for jj,c in enumerate(cvals):
#     Sdb = sines.mag2db(spec_avg[comp==c][0])
#     sorted = np.argsort(Sdb.sum(-1))
#     Sdb = Sdb[sorted]
#     f = np.arange(Sdb.shape[-1]) / 512 * 16000 / 1e3
#     for i in range(Sdb.shape[0]):
#         ax[jj,i].plot(f,Sdb[i],'k',linewidth=1)
#         ax[jj,i].set_xlim(0,f.max())
#         ax[jj,i].set_ylim(Sdb.min(),Sdb.max())
#         if jj == 0:
#             ax[jj,i].set_title(r"$S_{}$".format(i+1))
#         if jj == 2:
#             ax[-1,i].set_xlabel('frequency (kHz)')
#     ax[jj,0].set_ylabel(r"$c={}$".format(c))
# plt.tight_layout()
# plt.savefig(output_directory + "/spectrum_averages.pdf")
# plt.savefig(output_directory + "/spectrum_averages.png",dpi=200)
#%%
fig,ax = plt.subplots(1,3,figsize=(7,2.5))
PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'grey'},
    'medianprops':{'color':'blue'},
    'whiskerprops':{'color':'blue'},
    'capprops':{'color':'black'},
}
kwds_box = dict(showcaps=False,showfliers=False,whis=0,**PROPS)
sns.boxplot(data=df,x=compare_along,y='SI-SDR',ax=ax[0],**kwds_box)
sns.boxplot(data=df,x=compare_along,y='SIR',ax=ax[1],**kwds_box)
sns.boxplot(data=df,x=compare_along,y='SAR',ax=ax[2],**kwds_box)

lims = []
for _ax in ax:
    _ax.set_xlabel(r"compression $c$")
    lims.append(_ax.get_ylim())
    _ax.set_xticklabels(_ax.get_xticklabels(), rotation = 90)


# lims = np.array(lims)
# max_range = np.diff(lims).max()/2
# for _ax in ax:
#     lims = np.array(_ax.get_ylim())
#     mean = lims.mean()
#     lims[0] = mean - max_range
#     lims[1] = mean + max_range
#     _ax.set_ylim(*lims)
#     _ax.set_ylabel(None)
# ax[0].set_title('SI-SDR')
# ax[1].set_title('SIR')
# ax[2].set_title('SAR')
plt.tight_layout()
plt.savefig(output_directory / "boxplot_nofill.pdf",bbox_inches='tight')
# plt.close()

#%%
fig,ax = plt.subplots(1,3,figsize=(8.5,3))
kwds_box = dict(showfliers=False,whis=0)
sns.boxplot(data=df,x=compare_along,y='SI-SDR',ax=ax[0],**kwds_box)
sns.boxplot(data=df,x=compare_along,y='SIR',ax=ax[1],**kwds_box)
sns.boxplot(data=df,x=compare_along,y='SAR',ax=ax[2],**kwds_box)
plt.tight_layout()
plt.savefig(output_directory / "boxplot.pdf",bbox_inches='tight')

# %%
fig,ax = plt.subplots(1,3,figsize=(8.5,3))
sns.boxplot(data=df,hue=compare_along,y='SI-SDR',ax=ax[0],showfliers=False,x='place')
sns.boxplot(data=df,hue=compare_along,y='SIR',ax=ax[1],showfliers=False,x='place')
sns.boxplot(data=df,hue=compare_along,y='SAR',ax=ax[2],showfliers=False,x='place')
ax[0].legend([],[], frameon=False)
ax[1].legend([],[], frameon=False)

plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
plt.tight_layout()
plt.savefig(output_directory / "boxplot_estims.pdf",bbox_inches='tight')

# %%
fig,ax = plt.subplots(1,3,figsize=(8.5,3))
sns.boxplot(data=df,x=compare_along,y='SI-SDR',ax=ax[0],showfliers=False,hue='place')
sns.boxplot(data=df,x=compare_along,y='SIR',ax=ax[1],showfliers=False,hue='place')
sns.boxplot(data=df,x=compare_along,y='SAR',ax=ax[2],showfliers=False,hue='place')
ax[0].legend([],[], frameon=False)
ax[1].legend([],[], frameon=False)

plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
plt.tight_layout()
plt.savefig(output_directory / "boxplot_variance.pdf",bbox_inches='tight')

# %%
fig,ax = plt.subplots(1,3,figsize=(8.5,3))
kwds_line = dict(estimator='mean',errorbar=('sd', .5),color='k')
sns.pointplot(data=df,x=compare_along,y='SI-SDR',ax=ax[0],**kwds_line)
sns.pointplot(data=df,x=compare_along,y='SIR',ax=ax[1],**kwds_line)
sns.pointplot(data=df,x=compare_along,y='SAR',ax=ax[2],**kwds_line)
plt.tight_layout()
ax[1].set_title('.5 Standard Deviation')
plt.savefig(output_directory / "lineplot_sd.pdf",bbox_inches='tight')


#%%
fig,ax = plt.subplots(1,3,figsize=(8.5,3))
kwds_line = dict(estimator='median',hue='place')
sns.pointplot(data=df,x=compare_along,y='SI-SDR',ax=ax[0],**kwds_line)
sns.pointplot(data=df,x=compare_along,y='SIR',ax=ax[1],**kwds_line)
sns.pointplot(data=df,x=compare_along,y='SAR',ax=ax[2],**kwds_line)
# ax[1].set_title(' 95% confidence interval')
for _ax in ax:
    # _ax.set_xlim(2,6)
    _ax.set_xlabel(r'model sources $K$')
plt.tight_layout()
plt.savefig(output_directory / "lineplot_ci.pdf",bbox_inches='tight')


# %%
