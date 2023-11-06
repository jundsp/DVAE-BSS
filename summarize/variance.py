#%%
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd
import json
import torch
from pathlib import Path

plt.style.use('gibbs.mplstyles.latex')

# sns.set()

root = Path("/Users/julian/Documents/Code/Python/Results/SVAE/RWC_test/variance/")
output_directory = root / "results"
os.makedirs(output_directory,exist_ok=True)
  
compare_along = "variance"

# %%
json_name = root.glob("*/results.json")
DF = []
for ii,cc in enumerate(json_name):
    df = pd.read_json(cc,orient="index")
    df['place'] = 2-df['place']

    fname = cc.parents[0] / "configuration.txt"
    with open(fname) as f:
        data = f.read()
    js = json.loads(data)
    df['variance'] = round(js['variance'],3)
    df['sources'] = round(js['num_sources'],1)
    df['targets'] = round(js['num_targets'],1)

    df = df[(df['instrument1'] != 'glockenspiel') & (df['instrument2'] != 'glockenspiel')]

    DF.append(df)

#%%
df = pd.concat(DF)
df = df.reset_index()
df.rename(columns={'sdr': 'SI-SDR','sir': 'SIR','sar': 'SAR'},inplace=True)

dfm = df.melt(['sources','variance','index','place'], var_name='metric', value_name='vals')

#%%
fig,ax = plt.subplots(1,3,figsize=(7,2.5))
PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'grey'},
    'medianprops':{'color':'blue'},
    'whiskerprops':{'color':'blue'},
    'capprops':{'color':'black'}
}
kwds_box = dict(showcaps=False,showfliers=False,whis=0,**PROPS)
sns.boxplot(data=df,x=compare_along,y='SI-SDR',ax=ax[0],**kwds_box)
sns.boxplot(data=df,x=compare_along,y='SIR',ax=ax[1],**kwds_box)
sns.boxplot(data=df,x=compare_along,y='SAR',ax=ax[2],**kwds_box)

lims = []
for _ax in ax:
    _ax.set_xlabel(r"variance $\sigma^2$")
    _ax.set_xticklabels(_ax.get_xticklabels(), rotation = 45)
    lims.append(_ax.get_ylim())
    _ax.set_ylim(max(0,_ax.get_ylim()[0]))

plt.tight_layout()
plt.savefig(output_directory / "boxplot_nofill.pdf",bbox_inches='tight')


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
