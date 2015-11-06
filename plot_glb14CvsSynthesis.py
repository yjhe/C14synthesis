# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 21:43:00 2015

@author: Yujie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import D14Cpreprocess as prep
import mynetCDF as mync
from netCDF4 import Dataset
#%% ---------  check if data are utf-8  --------------------------
filename = 'Non_peat_data_synthesis.csv'
prep.sweepdata(filename)

#%% -------------  run get profile for 14C modeling  --------------
modeldim = (96.0, 96.0) # lat and lon dimension of the ESM
prep.getprofile4modeling(filename,*modeldim,cutdep=100.0,outfilename='sitegridid2.txt')

dum = np.loadtxt('sitegridid.txt',delimiter=',')
prep.plotDepth_cumC(filename,dum[:,2])

# write information of profiles used in modeling to csv. For table in paper
pltorisitesonly = 0
filename = 'sitegridid2.txt'
data = np.loadtxt(filename,unpack=True,delimiter=',')[:,:].T
mdlprofid = data[:,2].astype(float)
if pltorisitesonly == 0: # including extra sites
    filename = 'extrasitegridid.txt'
    data = np.loadtxt(filename,unpack=True,delimiter=',')[:,:].T
    mdlprofid = np.r_[mdlprofid, data[:,2].astype(float)]
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1', skiprows=[1])  
df = data[data['ProfileID'].isin(mdlprofid)]
df = df[['ProfileID','Author','Site','Lon','Lat','Layer_bottom','SampleYear','reference','title']]
aa = df.groupby('ProfileID').last()
# attach Cave14C
dum = np.loadtxt('tot48prof.txt', delimiter=',')
ave14C = pd.Series(dum[:,5], index=dum[:,2])
aa['ave14C'] = pd.Series(ave14C, index=aa.index)
aa.to_csv('./prof4modelinglist.csv')

# median sample year
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1', skiprows=[1])  
profid = data[data['Start_of_Profile']==1].index # index of profile start
sampleyr = prep.getvarxls(data, 'SampleYear', profid, 0)
print 'median sample year is: ', np.median(sampleyr)
print 'mean sample year is: ', np.nanmean(sampleyr)
#%% extract HWSD soc of the 48 profiles, this file needs to be updated once profile change
sawtcfn = 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\AWT_S_SOC.nc4'
tawtcfn = 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\AWT_T_SOC.nc4'
totprof = np.loadtxt('tot48prof.txt', unpack=True, delimiter=',').T
sawtc = prep.getHWSD(sawtcfn, totprof[:,0], totprof[:,1])
tawtc = prep.getHWSD(tawtcfn, totprof[:,0], totprof[:,1])
hwsdsoc = sawtc + tawtc
outf = open('hwsd48profsoc.txt',"w")
for item in hwsdsoc:
    outf.write("%.2f\n" % item)
outf.close()
#%% ---------  plot global 14C modeling D14C histogram vs synthesized data
pathh = "C:\\download\\work\\!manuscripts\\14Cboxmodel\\CMIP5_dataAnalysis\\" + \
        "twobox_modeling\\esmFixClim1"
#glbfn = "C:\\download\work\\!manuscripts\\14Cboxmodel\\CMIP5_dataAnalysis\\" + \
#        "RadioC_onebox_modeling\\Extrapolation\\D14Ctot_equrun_fixmodel.out"
#glbfn = pathh + "\\CESM\\glb_sitescalar_extraD14CSOC\\extratot.out_6.6_0.16"
glbfn = pathh + "\\CESM\\Extrapolate_D14CSOC\\D14C.out"
glbD14C = np.loadtxt(glbfn,unpack=True,delimiter=',',skiprows=1)[2,:].T
#glbD14C = np.loadtxt(glbfn,unpack=True,skiprows=0)[1,:].T
glbD14C[glbD14C>300] = np.nan
glbD14C[glbD14C<-600] = np.nan
filename = 'Non_peat_data_synthesis.csv'
Cave14C = prep.getCweightedD14C(filename,cutdep=600)[:,4]
# plot historgram
fig, axes = plt.subplots(1,1,figsize=(6,4))
binss = 30
axes.hist(glbD14C[~np.isnan(glbD14C)],binss,alpha=0.3, normed=True,label='fitted model')
axes.hist(Cave14C[~np.isnan(Cave14C)],binss,alpha=0.3, normed=True,label='All Profiles')
axes.legend()
axes.set_xlabel(r'SOC averaged $\Delta 14C$ ('+ u"\u2030)")
axes.set_ylabel(r'Fraction of grids (profiles)')
labels = [item.get_text() for item in axes.get_yticklabels()]
axes.set_yticklabels(np.array([float(x) for x in labels if x!=u''])*binss)
plt.show()
fig.savefig(pathh + '\\CESM\\Extrapolate_D14CSOC\\figure\\histogram.png')
#fig.savefig('./figures/histogram_hadleyoriginal.png')

#%% ---------  plot global 14C modeling D14C histogram vs profiles used for modeling
glbfn = "C:\\download\\work\\!manuscripts\\14Cboxmodel\\CMIP5_dataAnalysis\\" + \
        "RadioC_onebox_modeling\\glb_site_extraD14CSOC\\extratot.out.taumedscaled11.4transscaled0.126"
glbD14C = np.loadtxt(glbfn,unpack=True,skiprows=1,delimiter=',')[2,:].T
glbD14C[glbD14C>300] = np.nan
glbD14C[glbD14C<-600] = np.nan
proffn = 'sitegridid.txt'
Cave14C = np.loadtxt(proffn,unpack=True,skiprows=0,delimiter=',')[5,:].T
# plot historgram
fig, axes = plt.subplots(1,1,figsize=(6,4))
binss = 30
axes.hist(glbD14C[~np.isnan(glbD14C)],binss,alpha=0.3, normed=True,label='modeled with scalar')
axes.hist(Cave14C[~np.isnan(Cave14C)],binss,alpha=0.3, normed=True,label='Profiles used for modeling')
axes.legend()
axes.set_xlabel(r'SOC averaged $\Delta 14C$ ('+ u"\u2030)")
axes.set_ylabel(r'Fraction of grids (profiles)')
labels = [item.get_text() for item in axes.get_yticklabels()]
axes.set_yticklabels(np.array([float(x) for x in labels if x!=u''])*binss)
plt.show()
fig.savefig('./figures/histogram_prof4mdl_taumedscaled11.4transscaled0.126.png')

#%% -----------  plot global 14C modeling D14C hist vs synthesized data (top of mineral soil only)
glbfn = "C:\\download\\work\\!manuscripts\\14Cboxmodel\\CMIP5_dataAnalysis\\" + \
        "RadioC_onebox_modeling\\glb_site_extraD14CSOC\\extratot.out.taumedscaled11.4transscaled0.126"
glbD14C = np.loadtxt(glbfn,skiprows=1,delimiter=',',unpack=True)[2,:].T
glbD14C[glbD14C>300] = np.nan;glbD14C[glbD14C<-600] = np.nan;
filename = 'Non_peat_data_synthesis.csv'
Cave14C = prep.getCweightedD14C(filename)[:,3:5]
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID')  
profid_mineral = data[data['ProfileZeroDesignation']=='top of mineral soil'].index
# plot historgram
fig, axes = plt.subplots(1,1,figsize=(6,4))
Cave14C_idx = (~np.isnan(Cave14C[:,1])) & \
            [i in set(profid_mineral) for i in Cave14C[:,0]] # select profiles that starts from mineral
binss = 30
axes.hist(glbD14C,binss,alpha=0.3, normed=True,label='Modeled')
axes.hist(Cave14C[Cave14C_idx,1],binss,alpha=0.3, normed=True,label='Profiles')
axes.legend()
axes.set_xlabel(r'SOC averaged $\Delta 14C$ ('+ u"\u2030)")
axes.set_ylabel(r'Fraction of grids (profiles)')
labels = [item.get_text() for item in axes.get_yticklabels()]
axes.set_yticklabels(np.array([float(x) for x in labels])*binss)
plt.show()
#fig.savefig('./figures/histogram_topofmineral.pdf')

#%% plot depth
