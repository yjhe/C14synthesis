# -*- coding: utf-8 -*-
"""
Use Jobaggy curve to extrapolate SOC profile for sites missing SOC:
extract total 1m SOC from HWSD, use the vertical distribution of SOC reported in 
Jobaggy. 

Created on Mon Apr 20 09:44:14 2015

@author: Yujie
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import D14Cpreprocess as prep
from mpl_toolkits.basemap import Basemap, cm
import mynetCDF as mync
from netCDF4 import Dataset
import myplot as myplt
from scipy.interpolate import interp1d
import mystats as myst

jobgydepth = [20, 40, 60, 80, 100]
csvbiome = {1:[50, 25, 13, 7, 5],
            2:[50, 22, 13, 8, 7],
            3:[38, 22, 17, 13, 10],
            4:[41, 23, 15, 12, 9],
            5:[41, 23, 15,12, 9],
            6:[39, 22, 16, 13, 10],
            7:[46, 46, 46, 46, 46],
            8:[36, 23, 18, 13, 10]} # biome code in my xlsx. pctC from jobaggy

cutdep = 100.
filename = 'Non_peat_data_synthesis.csv'
Cave14C = prep.getCweightedD14C2(filename, cutdep=cutdep)
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1])
tot1mprofid = Cave14C[np.logical_and(Cave14C[:,1]==0.,Cave14C[:,2]==100.),3]
tot1mprofidlon = prep.getvarxls(data, 'Lon', tot1mprofid, 0)
tot1mprofidlat = prep.getvarxls(data, 'Lat', tot1mprofid, 0)
sitefilename = 'sitegridid2.txt'
dum = np.loadtxt(sitefilename, delimiter=',')
profid4modeling = dum[:,2]
extraprofid = list(set(tot1mprofid) - set(profid4modeling))
sawtcfn = '..\\AncillaryData\\HWSD\\Regridded_ORNLDAAC\\AWT_S_SOC.nc4'
tawtcfn = '..\\AncillaryData\\HWSD\\Regridded_ORNLDAAC\\AWT_T_SOC.nc4'
sawtc = prep.getHWSD(sawtcfn, tot1mprofidlon, tot1mprofidlat)
tawtc = prep.getHWSD(tawtcfn, tot1mprofidlon, tot1mprofidlat)
hwsdsoc = sawtc + tawtc

#%% compare jobaggy soc vs. obs soc, linear interpolation, using pctC
out = []
obss = []
depthh = []
for i in profid4modeling:
    print 'profile is :',i
    obs = 10. * data.loc[i:i,'BulkDensity'] * (data.loc[i:i,'Layer_bottom'] - \
          data.loc[i:i,'Layer_top']) * data.loc[i:i,'pct_C']/100. # kgC/m2
    jobgypctC = np.array(csvbiome[data.loc[i:i,'VegTypeCode_Local'].values[0]])/100.
    f_i = interp1d(np.r_[0,jobgydepth], np.r_[jobgypctC[0],jobgypctC])
    f_x = prep.extrap1d(f_i)
    layerbot = data.loc[i:i,'Layer_bottom'].values
    idx = layerbot > cutdep
    if cutdep in set(layerbot):
        layerbotinterp = layerbot[layerbot<=100.]
    else:
        layerbotinterp = np.hstack((layerbot[idx==False], cutdep))
    extpctC = f_x(layerbotinterp)/sum(f_x(layerbotinterp))*1.
    extrasoc = hwsdsoc[tot1mprofid==i] * np.reshape(extpctC,(f_x(layerbotinterp).shape[0],1))
    out.append(extrasoc)
    obss.append(obs)
    depthh.append(layerbotinterp)
#%% compare jobaggy soc vs. obs soc, log-fitting interpolation, using cum_pctC
from scipy.optimize import curve_fit
def func(x, K, I):
    return np.exp(K*np.log(x)+I)

out = []
obss = []
depthh = []
for i in profid4modeling:
    print 'profile is :',i
    obs = 10. * data.loc[i:i,'BulkDensity'] * (data.loc[i:i,'Layer_bottom'] - \
          data.loc[i:i,'Layer_top']) * data.loc[i:i,'pct_C']/100. # kgC/m2
    jobgypctC = np.array(csvbiome[data.loc[i:i,'VegTypeCode_Local'].values[0]])/100.
    popt, pcov = curve_fit(func, np.r_[0,jobgydepth], np.r_[0,np.cumsum(jobgypctC)])
    plt.plot(np.r_[0,jobgydepth], np.r_[0,np.cumsum(jobgypctC)]) 
    plt.plot(np.r_[0,jobgydepth], func(np.r_[0,jobgydepth], *popt))
    layerbot = data.loc[i:i,'Layer_bottom'].values
    idx = layerbot > cutdep
    if cutdep in set(layerbot):
        layerbotinterp = layerbot[layerbot<=100.]
    else:
        layerbotinterp = np.hstack((layerbot[idx==False], cutdep))
    pred = func(layerbotinterp, *popt) # cum pctC
    pred = pred - np.r_[0,pred[:-1]]
    extpctC = pred/sum(pred)*1.
    extrasoc = hwsdsoc[tot1mprofid==i] * extpctC.reshape(-1,1)
    out.append(extrasoc)
    obss.append(obs)
    depthh.append(layerbotinterp)
#%% plot
idd = 2
pro = profid4modeling[idd]
biome = {1:'Boreal Forest',2:'Temperate Forest',3:'Tropical Forest',4:'Grassland', \
         5:'Cropland',6:'Shrublands',7:'Peatland',8:'Savannas'}
print 'profid is ',pro
fig, ax = plt.subplots(1,1,figsize=(8,6))
plt.plot(obss[idd], data.loc[pro:pro,'Layer_bottom'].values, label='obs', marker='o')
plt.plot(out[idd], depthh[idd], label='extrapolated', marker='o')
ax.set_xlabel('SOC content (kgC/m2)')
ax.set_ylabel('depth')
bio = biome[data.loc[pro:pro,'VegTypeCode_Local'].values[0]]
ax.set_title('profile ID:' + str(pro) + '\n' + bio)
plt.legend(loc=4)
plt.gca().invert_yaxis()
#%% calculate rmse
rmse = []
for idd in range(len(profid4modeling)):
    pro = profid4modeling[idd]
    biome = {1:'Boreal Forest',2:'Temperate Forest',3:'Tropical Forest',4:'Grassland', \
             5:'Cropland',6:'Shrublands',7:'Peatland',8:'Savannas'}
    print 'profid is ',pro
    f_i = interp1d(data.loc[pro:pro,'Layer_bottom'].values, obss[idd])
    f_x = prep.extrap1d(f_i)
    y = f_x(depthh[idd])
    y = np.reshape(y,(y.shape[0],1))
    yhat = out[idd]
    rmse.append(myst.cal_RMSE(y, yhat))
    
#%% use jobaggy soc to extrapolate missing soc profiles, 
# log-fitting interpolation, using cum_pctC, write to extrasitegridid.txt
from scipy.optimize import curve_fit
def func(x, K, I):
    return np.exp(K*np.log(x)+I)

out = []
obss = []
depthh = []
outf = open('extrasitegridid.txt','w')
for i in extraprofid:
    print 'profile is :',i
    jobgypctC = np.array(csvbiome[data.loc[i:i,'VegTypeCode_Local'].values[0]])/100.
    popt, pcov = curve_fit(func, np.r_[0,jobgydepth], np.r_[0,np.cumsum(jobgypctC)])
    plt.plot(np.r_[0,jobgydepth], np.r_[0,np.cumsum(jobgypctC)], marker='o') 
    plt.plot(np.r_[0,jobgydepth], func(np.r_[0,jobgydepth], *popt), marker='o')
    layerbot = data.loc[i:i,'Layer_bottom'].values
    idx = layerbot > cutdep
    if cutdep in set(layerbot):
        layerbotinterp = layerbot[layerbot<=100.]
    else:
        layerbotinterp = np.hstack((layerbot[idx==False], cutdep))
    pred = func(layerbotinterp, *popt) # cum pctC
    pred = pred - np.r_[0,pred[:-1]] # convert cum pctC to pctC
    extpctC = pred/sum(pred)*1.
    extrasoc = hwsdsoc[tot1mprofid==i] * np.reshape(extpctC,(pred.shape[0],1))
    out.append(extrasoc)
    depthh.append(layerbotinterp)
    totsoc = np.nansum(extrasoc)
    # interpolate d14C for missing values if any
    d14C = data.loc[i:i,'D14C_BulkLayer'].values
    notNANs = ~np.isnan(d14C)
    f_i = interp1d(layerbot[notNANs], d14C[notNANs]); f_x = prep.extrap1d(f_i); 
    d14Cinterp = f_x(layerbotinterp)
    C14C = np.nansum(d14Cinterp * np.squeeze(extrasoc))/totsoc
    year = data.loc[i:i,'SampleYear'].values[0]
    if np.isnan(year):
        year = data.loc[i:i,'Measurement_Year'].values[0]
    outf.write("%.2f, %.2f, %d, %d, %d, %.2f, %.2f\n" % \
                (data.loc[i:i,'Lon'].values[0], data.loc[i:i,'Lat'].values[0], \
                i, 0., int(year), \
                C14C, totsoc))
outf.close()
    