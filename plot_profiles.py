# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 21:43:00 2015

@author: Yujie
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import D14Cpreprocess as prep
import pylab
import C14tools
import matplotlib

#%% fill Layer_top, Layer_bottom using Layer_top_norm Layer_bottom_norm
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
newdata = prep.get_originalLayerdepth(data)
newdata.to_csv('Non_peat_data_synthesis2.csv',encoding='iso-8859-1')
#%% plot 14C and SOC profile at each site
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
pid = data[data['Start_of_Profile']==1].index # index of profile start
d14C = prep.getvarxls(data,'D14C_BulkLayer', pid, ':')
sampleyr = prep.getvarxls(data, 'SampleYear', pid, ':')
tau, cost = C14tools.cal_tau(d14C, sampleyr)
data['tau'] = pd.Series(tau[:,0], index=data.index)

#%% plot individual profile
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16,10))
for fign in range(15):
    i = fign + 135
    ax1 = fig.axes[fign]
    plt.gca().invert_yaxis()
    ax2 = ax1.twiny()
    Y = data.loc[pid[i]:pid[i+1]-1,['Layer_bottom']]
    X1 = data.loc[pid[i]:pid[i+1]-1,['D14C_BulkLayer']]
    # total SOC
    X2 = np.array(data.loc[pid[i]:pid[i+1]-1,['BulkDensity']]).astype(float)* \
                    (np.array(data.loc[pid[i]:pid[i+1]-1,['Layer_bottom']].astype(float)) \
                    -np.array(data.loc[pid[i]:pid[i+1]-1,['Layer_top']].astype(float))) * \
                    np.array(data.loc[pid[i]:pid[i+1]-1,['pct_C']].astype(float))/100.0 # total SOC g/cm2
    # tau
    X2 = data.loc[pid[i]:pid[i+1]-1,['tau']]
    h1 = ax1.plot(np.array(X1.astype(float)),np.array(Y.astype(float)),color='g')
    ax1.set_xlabel(r"$\Delta14C$ $(permil)$",color='g')
    start, end = ax1.get_xlim()
    ax1.xaxis.set_ticks(np.arange(start, end,(end-start)/6))
    for tl in ax1.get_xticklabels():
        tl.set_color('g')
    h2 = ax2.plot(np.array(X2.astype(float)),np.array(Y.astype(float)))
    #ax2.set_xlabel(r"SOC content (g cm^{-2})")
    ax2.set_xlabel(r"tau (yr)")
fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
#plt.close()

#%% plot 14C profile of different biomes
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID')
biome = {1:'Boreal Forest',2:'Temperate Forest',3:'Tropical Forest',4:'Grassland', \
         5:'Cropland',6:'Shrublands',7:'Peatland',8:'Savannas',9:'Tundra',10:'Desert'}
var = ['D14C_BulkLayer','BulkDensity','pct_C','totalSOCgcm3', 'tau']
varlabel = [r"$\Delta14C$ ("+ u"\u2030)",
            r"Bulk Density $(g\ cm^{-3})$",
            r"percent C (%)",
            r"SOC content $(gC\ cm^{-3})$",
            r"turnover time (yr)"]
# customize
pltbiome = biome.keys()[8:10]
var1 = 1 # plot on the top x axis
var2 = 2 # plot on the bottom x axis

dum = 0
fig, axes = plt.subplots(nrows=1, ncols=len(set(pltbiome)), figsize=(12,8))
for i in set(pltbiome): # loop over biomes
    biomedataid = data[data.VegTypeCode_Local==i].index
    ax1 = fig.axes[dum] # bottom
    ax2 = ax1.twiny() # top
    cm = plt.get_cmap('Set1')
    numcolr = len(set(biomedataid)) # no repeat in color
    ax1.set_color_cycle([cm(1.*jj/numcolr) for jj in range(numcolr)])
    ax2.set_color_cycle([cm(1.*jj/numcolr) for jj in range(numcolr)])
    for kk in set(biomedataid): # loop over profiles in current biome
        Y = (data[data.index==kk]['Layer_top_norm'].values.astype(float)+\
            data[data.index==kk]['Layer_bottom_norm'].values.astype(float))/2.0
        X1 = data[data.index==kk][var[var1]].astype(float)
        if var2 == 3:
            X2 = data[data.index==kk]['BulkDensity'].astype(float) * \
                 data[data.index==kk]['pct_C'].astype(float)/100.0 # total SOC g/cm3
        else:
            X2 = data[data.index==kk][var[var2]].astype(float) 
        h1 = ax1.plot(X2,Y,'-.',lw=3,label='%s,%s,%s' % (data[data.index==kk].Site.values[0],\
            data[data.index==kk].Country.values[0],data[data.index==kk].State_City.values[0]))        
        ax1.set_xlabel(varlabel[var2]+'(dashed line)',color='g')
        ax1.set_ylabel(r"Depth $(cm)$")
        for tl in ax1.get_xticklabels():
                tl.set_color('g')
        h2 = ax2.plot(X1,Y)
        ax2.set_xlabel(varlabel[var1]+'(solid line)')
    plt.gca().invert_yaxis()
    pylab.text(0.8, 0.1,biome[i]+"\n"+"N = "+str(len(set(biomedataid))),
               horizontalalignment='center',verticalalignment='center',
               transform = ax1.transAxes,fontsize=16)
    dum = dum + 1 
fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
matplotlib.rcParams.update({'font.size': 10}) 
fig.savefig('../figures/biome_profiles/withGCBdata/BD_pctC/%s_%s.png'%(biome[pltbiome[0]],biome[pltbiome[1]]))

#%% plot 14C profile of different soil orders
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID')  
orderlabel = {1:'Alfisol',2:'Andisol',3:'Aridisol',4:'Entisol', \
         5:'Gelisol',6:'Histosol',7:'Inceptisol',8:'Mollisol',9:'Oxisol',10:'Spodosol',
         11:'Ultisol',12:'Vertisol'}
order = {1:'Alf',2:'And',3:'Ard',4:'Ent', \
         5:'Gel',6:'His',7:'Ept',8:'Oll',9:'Ox',10:'Spo',
         11:'Ult',12:'Ert'}
pltorder = order.keys()[10:12]
fig, axes = plt.subplots(nrows=1, ncols=len(set(pltbiome)), figsize=(12,8))
dum = 0
var = ['pct_C','totalSOCgcm3', 'tau']
varlabel = ['r"percent C $(%)$ (dashed line)"','r"SOC content $(g\ cm^{-3})$ (dashed line)"', 
            'r"turnover time (yr) (dashed line)"']
pltvar = 0 # plot on the bottom x axis
for i in pltorder: # loop over biomes
    biomedataid = data[data.SoilOrder_LEN_USDA==order[i][:3]].index
    ax1 = fig.axes[dum]
    ax2 = ax1.twiny()
    cm = plt.get_cmap('gist_rainbow')
    numcolr = len(set(biomedataid)) # no repeat in color
    ax1.set_color_cycle([cm(1.*jj/numcolr) for jj in range(numcolr)])
    ax2.set_color_cycle([cm(1.*jj/numcolr) for jj in range(numcolr)])
    for kk in set(biomedataid): # loop over profiles in current biome
        Y = (data[data.index==kk]['Layer_top_norm'].values.astype(float)+\
            data[data.index==kk]['Layer_bottom_norm'].values.astype(float))/2.0
        X1 = data[data.index==kk]['D14C_BulkLayer'].astype(float)
        if var[pltvar] == 'pct_C':    
            X2 = data[data.index==kk]['pct_C'].astype(float)/100.0      
        elif var[pltvar] == 'totalSOCgcm3':
            X2 = data[data.index==kk]['BulkDensity'].astype(float) * \
                 data[data.index==kk]['pct_C'].astype(float)/100.0 # total SOC g/cm3
        elif var[pltvar] == 'tau':
            X2 = data[data.index==kk]['tau'].astype(float) # tau (yr)
        h1 = ax1.plot(X2,Y,':',lw=3,label='%s,%s,%s' % (data[data.index==kk].Site.values[0],\
            data[data.index==kk].Country.values[0],data[data.index==kk].State_City.values[0]))        
        ax1.set_xlabel(eval(varlabel[pltvar]),color='g')
        ax1.set_ylabel(r"Depth $(cm)$")
        for tl in ax1.get_xticklabels():
                tl.set_color('g')
        h2 = ax2.plot(X1,Y)
        ax2.set_xlabel(r"$\Delta14C$ ("+ u"\u2030)"+ "(solid line)")
    plt.gca().invert_yaxis()
    pylab.text(0.8, 0.1,orderlabel[i]+"\n"+"N = "+str(len(set(biomedataid))),
               horizontalalignment='center',verticalalignment='center',
               transform = ax1.transAxes,fontsize=16)
    dum = dum + 1 
fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
matplotlib.rcParams.update({'font.size': 14}) 
#fig.savefig('../figures/biome_profiles/withGCBdata/%s_%s.png'%(order[pltorder[0]],order[pltorder[1]]))
#%% plot C-averaged D14C and depth/SOC content
filename = 'Non_peat_data_synthesis.csv'
Cave14C = prep.getCweightedD14C2(filename)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
nsmp = Cave14C[~np.isnan(Cave14C[:,4]),4].shape[0]
axes[0].scatter(Cave14C[:,5],Cave14C[:,4])
axes[0].set_xlabel(r'Total C Content$(kg  \ m^{-2})$')
axes[0].set_ylabel(r'SOC averaged $\Delta 14C$ ('+ u"\u2030)")
ylimm = axes[0].get_ylim()
axes[0].plot(np.array([7.02,7.02]),ylimm,'r:',lw=2)
axes[0].set_ylim(ylimm)
axes[0].set_xlim(0,axes[0].get_xlim()[1])
axes[0].annotate('HadGEM equilibrium global mean',xy=(7.3,-750),xytext=(15,-750),\
                 arrowprops=dict(facecolor='black',shrink=0.05)) # global average SOC stock in HadGEM is 7.02 kg/m2
axes[0].text(100,0.8,'n = %d'%nsmp)
axes[1].scatter(Cave14C[:,2],Cave14C[:,4])    
axes[1].set_xlabel(r'Depth (cm)')
axes[1].set_ylabel(r'SOC averaged $\Delta 14C$ ('+ u"\u2030)")
axes[1].set_xlim(0,axes[1].get_xlim()[1])
fig.tight_layout() 

fig.savefig('./figures/Cave_D14CvsDepthandSOC.tif')


#%% use increment approach, plot 14C profile of different biomes
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
pid = data[data['Start_of_Profile']==1].index # index of profile start
colname = ['Layer_top_norm','Layer_bottom_norm','Layer_top','Layer_bottom','D14C_BulkLayer','pct_C','BulkDensity','VegTypeCode_Local']
newdf = data[colname]
incre_data = prep.prep_increment(newdf, colname)

biome = {1:'Boreal Forest',2:'Temperate Forest',3:'Tropical Forest',4:'Grassland', \
         5:'Cropland',6:'Shrublands',7:'Peatland',8:'Savannas',9:'Tundra',10:'Desert'}
pltbiome = biome.keys()[8:10]
fig, axes = plt.subplots(nrows=1, ncols=len(set(pltbiome)), figsize=(12,8))
dum = 0
var = ['pct_C','totalSOCgcm3', 'tau']
varlabel = ['r"percent C $(%)$ (dashed line)"','r"SOC content $(g\ cm^{-3})$ (dashed line)"', 
            'r"turnover time (yr) (dashed line)"']
pltvar = 1 # plot on the bottom x axis
for i in set(pltbiome): # loop over biomes
    biomedataid = incre_data[incre_data.VegTypeCode_Local==i].index
    ax1 = fig.axes[dum]
    ax2 = ax1.twiny()
    cm = plt.get_cmap('gist_rainbow')
    numcolr = len(set(biomedataid)) # no repeat in color
    ax1.set_color_cycle([cm(1.*jj/numcolr) for jj in range(numcolr)])
    ax2.set_color_cycle([cm(1.*jj/numcolr) for jj in range(numcolr)])
    for kk in set(biomedataid): # loop over profiles in current biome
        Y = incre_data[incre_data.index==kk]['cont_depth'].values.astype(float)
        X1 = incre_data[incre_data.index==kk]['D14C_BulkLayer'].astype(float)
        if var[pltvar] == 'pct_C':    
            X2 = incre_data[incre_data.index==kk]['pct_C'].astype(float)/100.0      
        elif var[pltvar] == 'totalSOCgcm3':
            X2 = incre_data[incre_data.index==kk]['BulkDensity'].astype(float) * \
                 incre_data[incre_data.index==kk]['pct_C'].astype(float)/100.0 # total SOC g/cm3
        elif var[pltvar] == 'tau':
            X2 = incre_data[incre_data.index==kk]['tau'].astype(float) # tau (yr)
        h1 = ax1.plot(X2,Y,':',lw=3)        
        ax1.set_xlabel(eval(varlabel[pltvar]),color='g')
        ax1.set_ylabel(r"Depth $(cm)$")
        for tl in ax1.get_xticklabels():
                tl.set_color('g')
        h2 = ax2.plot(X1,Y)
        ax2.set_xlabel(r"$\Delta14C$ ("+ u"\u2030)"+ "(solid line)")
    plt.gca().invert_yaxis()
    pylab.text(0.8, 0.1,biome[i]+"\n"+"N = "+str(len(set(biomedataid))),
               horizontalalignment='center',verticalalignment='center',
               transform = ax1.transAxes,fontsize=16)
    dum = dum + 1 
fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
matplotlib.rcParams.update({'font.size': 14}) 
fig.savefig('../figures/biome_profiles/incre_1cm/%s_%s.png'%(biome[pltbiome[0]],biome[pltbiome[1]]))

#%% plot 14C~cumC profile of different biomes
filename = 'Non_peat_data_synthesis.csv'
data = prep.getD14C_cumC(filename)
biome = {1:'Boreal Forest',2:'Temperate Forest',3:'Tropical Forest',4:'Grassland', \
         5:'Cropland',6:'Shrublands',7:'Peatland',8:'Savannas',9:'Tundra',10:'Desert'}
pltbiome = biome.keys()[0:2]
fig, axes = plt.subplots(nrows=1, ncols=len(set(pltbiome)), figsize=(12,8))
dum = 0

for i in set(pltbiome): # loop over biomes
    biomedataid = data[data.VegTypeCode_Local==i].index
    ax1 = fig.axes[dum]
    cm = plt.get_cmap('Paired')
    numcolr = len(set(biomedataid)) # no repeat in color
    ax1.set_color_cycle([cm(1.*jj/numcolr) for jj in range(numcolr)])
    n = 0
    for kk in set(biomedataid): # loop over profiles in current biome
        if np.any(np.isnan(data.loc[kk:kk, 'cumC']),axis=0):
            continue
        Y = data.loc[kk:kk, 'D14C_BulkLayer'].astype(float)
        X1 = data.loc[kk:kk, 'cumC'].astype(float) # tau (yr)
        h1 = ax1.plot(X1,Y,'-',lw=1,label='%s,%s,%s' % (data[data.index==kk].Site.values[0],\
            data[data.index==kk].Country.values[0],data[data.index==kk].State_City.values[0]))        
        ax1.set_xlabel(r"cumulative C (kgC/m2)")
        ax1.set_ylabel(r"$\Delta14C$ ("+ u"\u2030)")
        if X1.values[-1] > 70:
            print 'kk is ',kk
            print 'cumC is ', data.loc[kk:kk, 'cumC']
        n += 1
    pylab.text(0.8, 0.1,biome[i]+"\n"+"N = "+str(n),
               horizontalalignment='center',verticalalignment='center',
               transform = ax1.transAxes,fontsize=16)
    dum = dum + 1 
fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
matplotlib.rcParams.update({'font.size': 14}) 
fig.savefig('../figures/biome_profiles/D14CvscumCofbiome/%s_%s.png'%(biome[pltbiome[0]],biome[pltbiome[1]]))

#%% get C total that has 14C > 0 and 14C < -500 in top 1m soil of different biomes
# do not fill missing pctC, use profiles that has complete pctC and Bulk Density info and are <1m deep
filename = 'Non_peat_data_synthesis.csv'
data = prep.getD14C_cumC(filename)
pid = data.index.unique() # index of profile start
biome = {1:'Boreal Forest',2:'Temperate Forest',3:'Tropical Forest',4:'Grassland', \
         5:'Cropland',6:'Shrublands',7:'Peatland',8:'Savannas',9:'Tundra',10:'Desert'}
cfastfrac = [] # row is each biome, cols are >0 and <-500 total absolute C kg/m2
cpasvfrac = []
for i in data.VegTypeCode_Local.unique():
    biomeprofid = data[data.VegTypeCode_Local==i].index.unique()
    n = 0
    cfastfrac += [],
    for kk in biomeprofid: # loop over all profiles in a biome
        if data.loc[kk:kk, 'Layer_bottom'].values[-1] < 100.:
            continue
        wr_smlzn1m = np.where(data.loc[kk:kk, 'Layer_bottom'].values<100.)[-1]
        is_smlzn1m = data.loc[kk:kk, 'Layer_bottom'].values<100.
        if data.loc[kk:kk, 'Layer_bottom'].values[-1] > 100.:   
            lastlyrCcnt = data.loc[kk:kk, 'pct_C'].values[wr_smlzn1m[-1]+1] * \
                          data.loc[kk:kk, 'BulkDensity'].values[wr_smlzn1m[-1]+1] * \
                          (100. - data.loc[kk:kk, 'Layer_top'].values[wr_smlzn1m[-1]+1])
        cumC = data.loc[kk:kk, 'cumC'].values
        cumC[wr_smlzn1m[-1]+1] = cumC[wr_smlzn1m[-1]] + lastlyrCcnt
        is_bigznD0 = data.loc[kk:kk, 'D14C_BulkLayer'].values>0.
        is_smlznD500 = data.loc[kk:kk, 'D14C_BulkLayer'].values<=-500.
        cfast = cumC[is_bigznD0 & is_smlzn1m]
        cpasv = cumC[is_smlznD500 & is_smlzn1m][-1] - \
                cumC[np.where(cumC==cumC[is_smlznD500 & is_smlzn1m][0])[-1]][0]
        cfastfrac[-1] += [cfast],
        cpasvfrac[-1] += [cpasv],

df = prep.getprofSOCD14C_interp(filename, 100)

#%% calculate # profiles that start from top and are at least 1m deep 
newdf = prep.cal_pfNumber(0, 0.)

filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
profid = data.index.unique()
Cave14C = prep.getCweightedD14C2(data, cutdep=100.)

top0 = prep.getvarxls(data, 'Layer_top', profid, 0)
plt.hist(top0,cumulative=True,bins=100)
top0bottom = prep.getvarxls(newdf, 'Layer_bottom', newdf.index.unique(), -1)
plt.hist(top0bottom,cumulative=True,bins=100)

#%% plot total SOC vs. D14C
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
sites = data.groupby(['Lon','Lat']).groups

Cave14C = prep.getCweightedD14C2(data, cutdep=100.)
subset = Cave14C[(Cave14C.Layer_top<=0) & (Cave14C.Layer_bottom ==100)]
kwargs = dict(meanprops=dict(marker='*', markeredgecolor='black', markersize=20,
                 markerfacecolor='firebrick'),
              showmeans=True)
subset.SOCave_14C.plot(kind='box',**kw  args)
Cave14C.SOCave_14C.mean()

#%% calculate 