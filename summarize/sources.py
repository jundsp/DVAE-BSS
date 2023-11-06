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
import numpy as np
from pathlib import Path

plt.style.use('gibbs.mplstyles.latex')

root = Path("/Users/julian/Documents/Code/Python/Results/SVAE/RWC_test/sources/")
output_directory = root / "results"
os.makedirs(output_directory,exist_ok=True)
  
compare_along = "sources"
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
    _ax.set_xlabel(r"sources $K$")
    lims.append(_ax.get_ylim())
    # _ax.set_xticklabels(_ax.get_xticklabels(), rotation = 90)

plt.tight_layout()
plt.savefig(output_directory / "boxplot_nofill.pdf")

#%%
fig,ax = plt.subplots(1,3,figsize=(8.5,3))
kwds_box = dict(showfliers=False,whis=0)
sns.boxplot(data=df,x=compare_along,y='SI-SDR',ax=ax[0],**kwds_box)
sns.boxplot(data=df,x=compare_along,y='SIR',ax=ax[1],**kwds_box)
sns.boxplot(data=df,x=compare_along,y='SAR',ax=ax[2],**kwds_box)
plt.tight_layout()
plt.savefig(output_directory / "boxplot.pdf")

# %%
fig,ax = plt.subplots(1,3,figsize=(8.5,3))
sns.boxplot(data=df,hue=compare_along,y='SI-SDR',ax=ax[0],showfliers=False,x='place')
sns.boxplot(data=df,hue=compare_along,y='SIR',ax=ax[1],showfliers=False,x='place')
sns.boxplot(data=df,hue=compare_along,y='SAR',ax=ax[2],showfliers=False,x='place')
ax[0].legend([],[], frameon=False)
ax[1].legend([],[], frameon=False)

plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
plt.tight_layout()
plt.savefig(output_directory / "boxplot_estims.pdf")

# %%
fig,ax = plt.subplots(1,3,figsize=(8.5,3))
sns.boxplot(data=df,x=compare_along,y='SI-SDR',ax=ax[0],showfliers=False,hue='place')
sns.boxplot(data=df,x=compare_along,y='SIR',ax=ax[1],showfliers=False,hue='place')
sns.boxplot(data=df,x=compare_along,y='SAR',ax=ax[2],showfliers=False,hue='place')
ax[0].legend([],[], frameon=False)
ax[1].legend([],[], frameon=False)

plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
plt.tight_layout()
plt.savefig(output_directory / "boxplot_variance.pdf")

# %%
fig,ax = plt.subplots(1,3,figsize=(8.5,3))
kwds_line = dict(estimator='mean',errorbar=('sd', .5),color='k')
sns.pointplot(data=df,x=compare_along,y='SI-SDR',ax=ax[0],**kwds_line)
sns.pointplot(data=df,x=compare_along,y='SIR',ax=ax[1],**kwds_line)
sns.pointplot(data=df,x=compare_along,y='SAR',ax=ax[2],**kwds_line)
plt.tight_layout()
ax[1].set_title('.5 Standard Deviation')
plt.savefig(output_directory / "lineplot_sd.pdf")


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
plt.savefig(output_directory / "lineplot_ci.pdf")

# %%

df5 = df[df['sources']==6]


order = df5.groupby(by=["instrument1"])["SI-SDR"].median().sort_values().index

plt.figure(figsize=(2.5,3))
sns.boxplot(data=df5,x='instrument1',y='SI-SDR',showfliers=False,order=order,whis=0,notch=True)
plt.xticks(rotation=90)
plt.ylim(0)
plt.xlabel(None)
plt.tight_layout()
plt.savefig(output_directory/'boxplot_instrument_sisdr.pdf',bbox_inches='tight')

#%%

i1 = df5['instrument1'].values
i2 = df5['instrument2'].values
mix = np.stack([i1,i2],-1)
mixi = np.zeros_like(mix).astype(int)

temp = np.unique(mix[:,0])
for i,j in enumerate(temp):
    mixi[mix==j] = i
inds = mixi.argsort(-1)
for i in range(inds.shape[0]):
    mix[i] = mix[i,inds[i]]

#%%
df5['instrument1'] = mix[:,0]
df5['instrument2'] = mix[:,1]
df5 = df5.drop(['instrument'],axis=1)
df_instr = df5.groupby(['instrument1','instrument2']).mean().reset_index()
mtx = df_instr.pivot(index='instrument2', columns='instrument1', values='SI-SDR')

# %%
plt.figure(figsize=(4,3))
ax = sns.heatmap(data=mtx,square=True,annot=True,cmap='plasma',fmt="2.0f",cbar_kws={'label': 'SI-SDR'})
ax.set(xlabel="", ylabel="")
# ax.xaxis.tick_top()
# plt.xticks(rotation=50,ha='right')
plt.tight_layout()
plt.savefig(output_directory/"heatmap_instr.pdf")
# %%
plt.figure(figsize=(3,2))
df5=df5.round({'delta_time':1})
sns.lineplot(data=df5,x='delta_time',y='SI-SDR')
plt.xlabel(r"$|\Delta|$ time (s)")
plt.xlim(0,.9)
plt.ylim(5,15)
plt.tight_layout()
plt.savefig(output_directory/"time_sdr.pdf")
# %%
df6 = df[df['sources']==6]
plt.figure(figsize=(4,2.5))
sns.lmplot(data=df6[df6['delta_time']<=.9],x='delta_time',y='SI-SDR',order=3,scatter=None)
# plt.ylim(4,10)
plt.xlabel(r"$\Delta$ time (s)")
plt.xlim(0,.9)
plt.tight_layout()
# %%
df_pitch = df[df['sources']==6]
df_pitch['delta_pitch'] = np.abs(df_pitch['delta_pitch'])
df_pitch = df_pitch[df_pitch['delta_pitch']<(5*12)]
plt.figure(figsize=(3,2))
sns.lineplot(data=df_pitch,x='delta_pitch',y='SI-SDR')
plt.xlabel(r"$| \Delta |$ semitones")
plt.xlim(0,5*12)
plt.ylim(0,20)
plt.xticks(np.arange(0,5*12,12))
plt.tight_layout()
plt.savefig(output_directory/"pitch_sdr.pdf")

# %%
