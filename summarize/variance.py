#%%
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd
import json
plt.rcParams['figure.dpi'] = 150
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times",
    "axes.titlesize": "medium",
    "axes.labelsize": "medium",
})
# sns.set()

root = "/Users/julian/Documents/Code/Python/Results/SVAE/RWC/variance/"
output_directory = root + "/results"
os.makedirs(output_directory,exist_ok=True)
  

# %%
csv_name = glob.glob(root + "*/*/*/bss.csv")
DF = []
for ii,cc in enumerate(csv_name):
    df = pd.read_csv(cc)
    df.rename(columns={'Source': 'estimate'},inplace=True)
    df['estimate'] +=1


    fname = os.path.dirname(cc) + "/configuration.txt"
    with open(fname) as f:
        data = f.read()
    js = json.loads(data)
    df['variance'] = round(js['variance'],3)
    df['sources'] = round(js['num_sources'],1)
    df['targets'] = round(js['num_targets'],1)

    DF.append(df)
#%%
df = pd.concat(DF)
df = df.reset_index()
df.rename(columns={'SDR': 'SI-SDR'},inplace=True)

dfm = df.melt(['sources','variance','index','estimate'], var_name='metric', value_name='vals')

#%%
fig,ax = plt.subplots(1,3,figsize=(10,3))
PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'grey'},
    'medianprops':{'color':'blue'},
    'whiskerprops':{'color':'blue'},
    'capprops':{'color':'black'}
}
kwds_box = dict(showcaps=False,showfliers=False,whis=0,**PROPS)
sns.boxplot(data=df,x="variance",y='SI-SDR',ax=ax[0],**kwds_box)
sns.boxplot(data=df,x="variance",y='SIR',ax=ax[1],**kwds_box)
sns.boxplot(data=df,x="variance",y='SAR',ax=ax[2],**kwds_box)
plt.tight_layout()
plt.savefig(output_directory + "/boxplot_nofill.pdf")

#%%
fig,ax = plt.subplots(1,3,figsize=(10,3))
kwds_box = dict(showfliers=False)
sns.boxplot(data=df,x="variance",y='SI-SDR',ax=ax[0],**kwds_box)
sns.boxplot(data=df,x="variance",y='SIR',ax=ax[1],**kwds_box)
sns.boxplot(data=df,x="variance",y='SAR',ax=ax[2],**kwds_box)
plt.tight_layout()
plt.savefig(output_directory + "/boxplot.pdf")

# %%
fig,ax = plt.subplots(1,3,figsize=(10,3))
sns.boxplot(data=df,hue="variance",y='SI-SDR',ax=ax[0],showfliers=False,x='estimate')
sns.boxplot(data=df,hue="variance",y='SIR',ax=ax[1],showfliers=False,x='estimate')
sns.boxplot(data=df,hue="variance",y='SAR',ax=ax[2],showfliers=False,x='estimate')
ax[0].legend([],[], frameon=False)
ax[1].legend([],[], frameon=False)

plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
plt.tight_layout()
plt.savefig(output_directory + "/boxplot_estims.pdf")

# %%
fig,ax = plt.subplots(1,3,figsize=(10,3))
sns.boxplot(data=df,x="variance",y='SI-SDR',ax=ax[0],showfliers=False,hue='estimate')
sns.boxplot(data=df,x="variance",y='SIR',ax=ax[1],showfliers=False,hue='estimate')
sns.boxplot(data=df,x="variance",y='SAR',ax=ax[2],showfliers=False,hue='estimate')
ax[0].legend([],[], frameon=False)
ax[1].legend([],[], frameon=False)

plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
plt.tight_layout()
plt.savefig(output_directory + "/boxplot_variance.pdf")

# %%
fig,ax = plt.subplots(1,3,figsize=(10,3))
kwds_line = dict(estimator='mean',errorbar=('sd', .5),color='k')
sns.pointplot(data=df,x="variance",y='SI-SDR',ax=ax[0],**kwds_line)
sns.pointplot(data=df,x="variance",y='SIR',ax=ax[1],**kwds_line)
sns.pointplot(data=df,x="variance",y='SAR',ax=ax[2],**kwds_line)
plt.tight_layout()
ax[1].set_title('.5 Standard Deviation')
plt.savefig(output_directory + "/lineplot_sd.pdf")


#%%
fig,ax = plt.subplots(1,3,figsize=(10,3))
kwds_line = dict(estimator='median',errorbar=('ci', 95),hue='sources')
sns.pointplot(data=df,x="variance",y='SI-SDR',ax=ax[0],**kwds_line)
sns.pointplot(data=df,x="variance",y='SIR',ax=ax[1],**kwds_line)
sns.pointplot(data=df,x="variance",y='SAR',ax=ax[2],**kwds_line)
# ax[1].set_title(' 95% confidence interval')
plt.tight_layout()
plt.savefig(output_directory + "/lineplot_ci.pdf")
# %%
