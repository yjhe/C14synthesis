# -*- coding: utf-8 -*-
"""
Plot bar plot of absolute change in SOC uptake over 140yr esmFixClim1 of fitted model 
vs. 14C constrained model

Created on Wed Apr 15 14:40:51 2015

@author: Yujie
"""

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import D14Cpreprocess as prep
import mynetCDF as mync
from netCDF4 import Dataset
import scipy.io
import pylab
from matplotlib.ticker import FuncFormatter

#%% ESM get data, 2 pool RC
prefix = 'C:\\download\\work\\!manuscripts\\14Cboxmodel\\CMIP5_dataAnalysis\\'
mdl = ['CESM','GFDL','HadGEM','IPSL','MRI']
conv = 1e3/1e15*1.
esmtotsoc = np.zeros((5,4)) # first 2 cols: fitted; last 2 cols: 14C
for n,mdlname in enumerate(mdl):
    print 'model is:', mdlname
    areapath = prefix + 'esmFixClim1runs\\' + mdlname + '_FixClim1_annuvar.mat'
    areadata = scipy.io.loadmat(areapath)
    areacella = areadata['areacella']
    areacellamat = np.tile(np.reshape(areacella,(list(areacella.shape)+[1])),[1,1,140])
    landfrac = areadata['landfrac']/100.
    landfracmat = np.tile(np.reshape(landfrac,(list(landfrac.shape)+[1])),[1,1,140])
    mdlsocpath = prefix + 'twobox_modeling\\esmFixClim1\\' + \
                 mdlname + '\\Extrapolate_D14CSOC\\fwdcsoilfitted.mat'
    esm = scipy.io.loadmat(mdlsocpath)
    glbcfastfit = np.squeeze(np.nansum(np.nansum(esm['alfast'][:,:,:140] *
                          areacellamat * landfracmat * conv,axis=0), axis=0))
    glbcmedfit = np.squeeze(np.nansum(np.nansum(esm['almed'][:,:,:140] *
                          areacellamat * landfracmat * conv,axis=0),axis=0))
    C14mdlsocpath = prefix + 'twobox_modeling\\esmFixClim1\\' + mdlname + \
                    '\\glb_sitescalar_extraD14CSOC\\fwdcsoilwscalar.mat'
    esm = scipy.io.loadmat(C14mdlsocpath)
    C14glbcfastfit = np.squeeze(np.nansum(np.nansum(esm['alfast'][:,:,:140] *
                          areacellamat * landfracmat * conv,axis=0),axis=0))
    C14glbcmedfit = np.squeeze(np.nansum(np.nansum(esm['almed'][:,:,:140] *
                          areacellamat * landfracmat *conv,axis=0),axis=0))
    esmtotsoc[n,:] = np.array([glbcfastfit[-1] - glbcfastfit[0],
                               glbcmedfit[-1] - glbcmedfit[0],
                               C14glbcfastfit[-1] - C14glbcfastfit[0],
                               C14glbcmedfit[-1] - C14glbcmedfit[0]])
    
#%% plot, ESM get data, 2 pool RC
fig, ax = plt.subplots(figsize=(8,6))
N = 5; ind = np.arange(N)
width = .35
fittedmdl_fast = ax.bar(ind, esmtotsoc[:,0], width, color='white', 
                        align='center', label='fast pool RC model', hatch='//')
C14mdl_fast = ax.bar(ind+width+0.05, esmtotsoc[:,2], width, color='grey', 
                     align='center', hatch='//', label=r'fast pool $^{14}C$ model')
fittedmdl_slow = ax.bar(ind, esmtotsoc[:,1], width, bottom=esmtotsoc[:,0], 
                        color='white', align='center', label='slow pool RC model', hatch=None)
C14mdl_slow = ax.bar(ind+width+0.05, esmtotsoc[:,3], width, bottom=esmtotsoc[:,2],
                     color='grey', align='center', hatch=None, label=r'slow pool $^{14}C$ model')
mean_fitted_fast = ax.bar(max(ind)+width+1.5, np.mean(esmtotsoc[:,0]), width, color='white', 
                     align='center', hatch='//')
mean_fitted_slow = ax.bar(max(ind)+width+1.5, np.mean(esmtotsoc[:,1]), width, bottom=np.mean(esmtotsoc[:,0]),
                     color='white', align='center', hatch=None)
mean_14C_fast = ax.bar(max(ind)+2*width+1.5+.05, np.mean(esmtotsoc[:,2]), width, color='grey', 
                     align='center', hatch='//')
mean_14C_slow = ax.bar(max(ind)+2*width+1.5+.05, np.mean(esmtotsoc[:,3]), width, bottom=np.mean(esmtotsoc[:,2]),
                     color='grey', align='center', hatch=None)
plt.legend(bbox_to_anchor=(1,1),loc='upper right',ncol=1,fontsize=11,labelspacing=0.2)
ax.set_ylabel(r'$\Delta$ SOC (Pg C)')
ax.set_xlim([min(ind)-.5, max(ind)+2*width+2.5])
ax.set_xticks(list(ind+width/2.)+[max(ind)+width/2.+1.85])
ax.set_xticklabels(mdl + ['Mean'])
#%% calculate std of multimodel sink reduction
np.std(np.sum(esmtotsoc[:,0:2],axis=1)-np.sum(esmtotsoc[:,2:],axis=1))


#%% ESM get data, 3 pool RC
prefix = 'C:\\download\\work\\!manuscripts\\14Cboxmodel\\CMIP5_dataAnalysis\\'
mdl = ['CESM','GFDL','HadGEM','IPSL','MRI']
conv = 1e3/1e15*1.
esmtotsoc = np.zeros((5,4)) # first 2 cols: fitted; last 2 cols: 14C
esmtotsoc_3p = np.zeros((5,6)) # first 3 cols: fitted; last 3 cols: 14C
for n,mdlname in enumerate(mdl):
    print 'model is:', mdlname
    areapath = prefix + 'esmFixClim1runs\\' + mdlname + '_FixClim1_annuvar.mat'
    areadata = scipy.io.loadmat(areapath)
    areacella = areadata['areacella']
    areacellamat = np.tile(np.reshape(areacella,(list(areacella.shape)+[1])),[1,1,140])
    landfrac = areadata['landfrac']/100.
    landfracmat = np.tile(np.reshape(landfrac,(list(landfrac.shape)+[1])),[1,1,140])
    mdlsocpath = prefix + 'twobox_modeling\\esmFixClim1\\' + \
                 mdlname + '\\Extrapolate_D14CSOC\\fwdcsoilfitted.mat'
    esm = scipy.io.loadmat(mdlsocpath)
    glbcfastfit = np.squeeze(np.nansum(np.nansum(esm['alfast'][:,:,:140] *
                          areacellamat * landfracmat * conv,axis=0), axis=0))
    glbcmedfit = np.squeeze(np.nansum(np.nansum(esm['almed'][:,:,:140] *
                          areacellamat * landfracmat * conv,axis=0),axis=0))
    C14mdlsocpath = prefix + 'twobox_modeling\\esmFixClim1\\' + mdlname + \
                    '\\glb_sitescalar_extraD14CSOC\\fwdcsoilwscalar.mat'
    esm = scipy.io.loadmat(C14mdlsocpath)
    C14glbcfastfit = np.squeeze(np.nansum(np.nansum(esm['alfast'][:,:,:140] *
                          areacellamat * landfracmat * conv,axis=0),axis=0))
    C14glbcmedfit = np.squeeze(np.nansum(np.nansum(esm['almed'][:,:,:140] *
                          areacellamat * landfracmat *conv,axis=0),axis=0))
    esmtotsoc[n,:] = np.array([glbcfastfit[-1] - glbcfastfit[0],
                               glbcmedfit[-1] - glbcmedfit[0],
                               C14glbcfastfit[-1] - C14glbcfastfit[0],
                               C14glbcmedfit[-1] - C14glbcmedfit[0]])
    if mdlname in ['CESM','IPSL','MRI']: 
        print 'model is %s 3 pool'%mdlname
        mdlsocpath = prefix + 'twobox_modeling\\esmFixClim1\\' + \
                     mdlname + '\\Extrapolate_D14CSOC_3pool\\3poolmodel_3pooldata\\fwdcsoilfitted.mat'
        esm = scipy.io.loadmat(mdlsocpath)
        glbcfastfit = np.squeeze(np.nansum(np.nansum(esm['alfast'][:,:,:140] *
                              areacellamat * landfracmat * conv,axis=0), axis=0))
        glbcmedfit = np.squeeze(np.nansum(np.nansum(esm['almed'][:,:,:140] *
                              areacellamat * landfracmat * conv,axis=0),axis=0))
        glbcpasfit = np.squeeze(np.nansum(np.nansum(esm['alslow'][:,:,:140] *
                              areacellamat * landfracmat * conv,axis=0), axis=0))
        C14mdlsocpath = prefix + 'twobox_modeling\\esmFixClim1\\' + mdlname + \
                        '\\glb_sitescalar_extraD14C_3pool_3pooldata\\fwdcsoilwscalar.mat'
        esm = scipy.io.loadmat(C14mdlsocpath)
        C14glbcfastfit = np.squeeze(np.nansum(np.nansum(esm['alfast'][:,:,:140] *
                              areacellamat * landfracmat * conv,axis=0),axis=0))
        C14glbcmedfit = np.squeeze(np.nansum(np.nansum(esm['almed'][:,:,:140] *
                              areacellamat * landfracmat *conv,axis=0),axis=0))
        C14glbcpasfit = np.squeeze(np.nansum(np.nansum(esm['alslow'][:,:,:140] *
                              areacellamat * landfracmat *conv,axis=0),axis=0))
        esmtotsoc_3p[n,:] = np.array([glbcfastfit[-1] - glbcfastfit[0], 
                                     glbcmedfit[-1] - glbcmedfit[0],
                                     glbcpasfit[-1] - glbcpasfit[0],
                                     C14glbcfastfit[-1] - C14glbcfastfit[0],
                                     C14glbcmedfit[-1] - C14glbcmedfit[0],
                                     C14glbcpasfit[-1] - C14glbcpasfit[0]])
#is_other3mdl = esmtotsoc_3p[:,[0,1,3,4]]==0                   
#esmtotsoc_3p[is_other3mdl] = esmtotsoc[is_other3mdl[:3,:]]
#esmtotsoc_3p[4,:] = 0
esmtotsoc_bc = esmtotsoc_3p.copy() # best case
esmtotsoc_bc[[[1],[2],[4]],[0,1,3,4]] = esmtotsoc[[1,2,4],:]
#%% plot,  ESM get data, 3 pool RC
fig, ax = plt.subplots(figsize=(10,6))
N = 5; ind = np.arange(N)
width = .25
fittedmdl_fast = ax.bar(ind, esmtotsoc[:,0], width, color='white', 
                        align='center', label='fast pool RC model', hatch='//')
C14mdl_fast = ax.bar(ind+width, esmtotsoc[:,2], width, color='grey', 
                     align='center', hatch='//', label=r'fast pool 2-pool $^{14}C$ model')
C14mdl_fast_3p = ax.bar(ind+2*width, esmtotsoc_3p[:,3], width, color='lightgrey', 
                     align='center', hatch='//', label=r'fast pool 3-pool $^{14}C$ model')

fittedmdl_slow = ax.bar(ind, esmtotsoc[:,1], width, bottom=esmtotsoc[:,0], 
                        color='white', align='center', label='slow pool RC model', hatch=None)
C14mdl_slow = ax.bar(ind+width, esmtotsoc[:,3], width, bottom=esmtotsoc[:,2],
                     color='grey', align='center', hatch=None, label=r'slow pool 2-pool $^{14}C$ model')
C14mdl_slow_3p = ax.bar(ind+2*width, esmtotsoc_3p[:,4], width, bottom=esmtotsoc_3p[:,3],
                     color='lightgrey', align='center', hatch=None, label=r'slow pool 3-pool $^{14}C$ model')
C14mdl_pas_3p = ax.bar(ind+2*width, esmtotsoc_3p[:,5], width, bottom=np.sum(esmtotsoc_3p[:,3:5],axis=1),
                     color='lightgrey', align='center', hatch='O', label=r'passive pool 3-pool $^{14}C$ model')

mean_fitted_fast = ax.bar(max(ind)+width+1.5, np.mean(esmtotsoc[:,0]), width, color='white', 
                     align='center', hatch='//')
mean_fitted_slow = ax.bar(max(ind)+width+1.5, np.mean(esmtotsoc[:,1]), width, bottom=np.mean(esmtotsoc[:,0]),
                     color='white', align='center', hatch=None)
mean_14C_fast = ax.bar(max(ind)+2*width+1.5, np.mean(esmtotsoc[:,2]), width, color='grey', 
                     align='center', hatch='//')
mean_14C_slow = ax.bar(max(ind)+2*width+1.5, np.mean(esmtotsoc[:,3]), width, bottom=np.mean(esmtotsoc[:,2]),
                     color='grey', align='center', hatch=None)
#mean_14C_fast_3p = ax.bar(max(ind)+3*width+1.5, np.mean(esmtotsoc_3p[3:4,3]), width, color='lightgrey', 
#                     align='center', hatch='//')
#mean_14C_slow_3p = ax.bar(max(ind)+3*width+1.5, np.mean(esmtotsoc_3p[3:4,4]), width, 
#                          bottom=np.mean(esmtotsoc_3p[3:4,3]),
#                          color='lightgrey', align='center', hatch=None)
#mean_14C_pas_3p = ax.bar(max(ind)+3*width+1.5, np.mean(esmtotsoc_3p[3:4,5]), width, 
#                         bottom=np.mean(esmtotsoc_3p[3:4,3])+np.mean(esmtotsoc_3p[3:4,4]),
#                         color='lightgrey', align='center', hatch='O')
plt.legend(loc='upper right',ncol=1,fontsize=11,labelspacing=0.2)
ax.set_ylabel(r'$\Delta$ SOC (Pg C)')
ax.set_xlim([min(ind)-.5, max(ind)+2*width+2.5])
xticksloc = np.array(list(ind+width/2.)+[max(ind)+width*2.+1.5-width/2])
xticksloc[[0,3]] = xticksloc[[0,3]]+width/2.
ax.set_xticks(xticksloc)
ax.set_xticklabels(mdl + ['Mean'])

#%% plot,  ESM get data, 3 pool RC. show 2-pool RC mean AND best-case RC mean
fig, ax = plt.subplots(figsize=(10,6))
N = 5; ind = np.arange(N)
width = .25
fittedmdl_fast = ax.bar(ind, esmtotsoc[:,0], width, color='white', 
                        align='center', label='fast pool RC model', hatch='//')
C14mdl_fast = ax.bar(ind+width, esmtotsoc[:,2], width, color='grey', 
                     align='center', hatch='//', label=r'fast pool 2-pool $^{14}C$ model')
C14mdl_fast_3p = ax.bar(ind+2*width, esmtotsoc_3p[:,3], width, color='lightgrey', 
                     align='center', hatch='//', label=r'fast pool 3-pool $^{14}C$ model')

fittedmdl_slow = ax.bar(ind, esmtotsoc[:,1], width, bottom=esmtotsoc[:,0], 
                        color='white', align='center', label='slow pool RC model', hatch=None)
C14mdl_slow = ax.bar(ind+width, esmtotsoc[:,3], width, bottom=esmtotsoc[:,2],
                     color='grey', align='center', hatch=None, label=r'slow pool 2-pool $^{14}C$ model')
C14mdl_slow_3p = ax.bar(ind+2*width, esmtotsoc_3p[:,4], width, bottom=esmtotsoc_3p[:,3],
                     color='lightgrey', align='center', hatch=None, label=r'slow pool 3-pool $^{14}C$ model')
C14mdl_pas_3p = ax.bar(ind+2*width, esmtotsoc_3p[:,5], width, bottom=np.sum(esmtotsoc_3p[:,3:5],axis=1),
                     color='lightgrey', align='center', hatch='O', label=r'passive pool 3-pool $^{14}C$ model')

mean_fitted_fast = ax.bar(max(ind)+width+1.5, np.mean(esmtotsoc[:,0]), width, color='white', 
                     align='center', hatch='//')
mean_fitted_slow = ax.bar(max(ind)+width+1.5, np.mean(esmtotsoc[:,1]), width, bottom=np.mean(esmtotsoc[:,0]),
                     color='white', align='center', hatch=None)
mean_14C_fast = ax.bar(max(ind)+2*width+1.5, np.mean(esmtotsoc[:,2]), width, color='grey', 
                     align='center', hatch='//')
mean_14C_slow = ax.bar(max(ind)+2*width+1.5, np.mean(esmtotsoc[:,3]), width, bottom=np.mean(esmtotsoc[:,2]),
                     color='grey', align='center', hatch=None)


mean_fitted_fast = ax.bar(max(ind)+width+1.5, np.mean(esmtotsoc[:,0]), width, color='white', 
                     align='center', hatch='//')
mean_fitted_slow = ax.bar(max(ind)+width+1.5, np.mean(esmtotsoc[:,1]), width, bottom=np.mean(esmtotsoc[:,0]),
                     color='white', align='center', hatch=None)
mean_14C_fast = ax.bar(max(ind)+2*width+1.5, np.mean(esmtotsoc[:,2]), width, color='grey', 
                     align='center', hatch='//')
mean_14C_slow = ax.bar(max(ind)+2*width+1.5, np.mean(esmtotsoc[:,3]), width, bottom=np.mean(esmtotsoc[:,2]),
                     color='grey', align='center', hatch=None)                     
mean_14C_fast_3p = ax.bar(max(ind)+3*width+1.5, np.mean(np.sum(esmtotsoc_bc[:,3:],axis=1)), width, color='white', 
                     align='center', hatch='.',label='Best-case')
#mean_14C_slow_3p = ax.bar(max(ind)+3*width+1.5, np.mean(esmtotsoc_bc[:,4]), width, 
#                          bottom=np.mean(esmtotsoc_bc[:,3]),
#                          color='white', align='center', hatch='.')
#mean_14C_pas_3p = ax.bar(max(ind)+3*width+1.5, np.mean(esmtotsoc_bc[:,5]), width, 
#                         bottom=np.mean(esmtotsoc_bc[:,3])+np.mean(esmtotsoc_bc[:,4]),
#                         color='white', align='center', hatch='.',label='Best-case')
plt.legend(loc='upper right',ncol=1,fontsize=11,labelspacing=0.2)
ax.set_ylabel(r'$\Delta$ SOC (Pg C)')
ax.set_xlim([min(ind)-.5, max(ind)+2*width+2.5])
xticksloc = np.array(list(ind+width/2.)+[max(ind)+width*2.+1.5-width/2])
xticksloc[[0,3,5]] = xticksloc[[0,3,5]]+width/2.
ax.set_xticks(xticksloc)
ax.set_xticklabels(mdl + ['Mean'])

#%% plot,  ESM get data, 2pool and 3 pool RC. show best-case RC mean
fig, ax = plt.subplots(figsize=(10,6))
N = 5; ind = np.arange(N)
width = .25
fittedmdl_fast = ax.bar(ind, esmtotsoc[1:3,0], width, color='white', 
                        align='center', label='fast pool RC model', hatch='//')
C14mdl_fast = ax.bar(ind+width, esmtotsoc[1:3,2], width, color='grey', 
                     align='center', hatch='//', label=r'fast pool 2-pool $^{14}C$ model')
C14mdl_fast_3p = ax.bar(ind+2*width, esmtotsoc_3p[:,3], width, color='lightgrey', 
                     align='center', hatch='//', label=r'fast pool 3-pool $^{14}C$ model')

fittedmdl_slow = ax.bar(ind, esmtotsoc[:,1], width, bottom=esmtotsoc[:,0], 
                        color='white', align='center', label='slow pool RC model', hatch=None)
C14mdl_slow = ax.bar(ind+width, esmtotsoc[:,3], width, bottom=esmtotsoc[:,2],
                     color='grey', align='center', hatch=None, label=r'slow pool 2-pool $^{14}C$ model')
C14mdl_slow_3p = ax.bar(ind+2*width, esmtotsoc_3p[:,4], width, bottom=esmtotsoc_3p[:,3],
                     color='lightgrey', align='center', hatch=None, label=r'slow pool 3-pool $^{14}C$ model')
C14mdl_pas_3p = ax.bar(ind+2*width, esmtotsoc_3p[:,5], width, bottom=np.sum(esmtotsoc_3p[:,3:5],axis=1),
                     color='lightgrey', align='center', hatch='O', label=r'passive pool 3-pool $^{14}C$ model')

mean_fitted_fast = ax.bar(max(ind)+width+1.5, np.mean(esmtotsoc[:,0]), width, color='white', 
                     align='center', hatch='//')
mean_fitted_slow = ax.bar(max(ind)+width+1.5, np.mean(esmtotsoc[:,1]), width, bottom=np.mean(esmtotsoc[:,0]),
                     color='white', align='center', hatch=None)
mean_14C_fast = ax.bar(max(ind)+2*width+1.5, np.mean(esmtotsoc[:,2]), width, color='grey', 
                     align='center', hatch='//')
mean_14C_slow = ax.bar(max(ind)+2*width+1.5, np.mean(esmtotsoc[:,3]), width, bottom=np.mean(esmtotsoc[:,2]),
                     color='grey', align='center', hatch=None)                     
mean_14C_fast_3p = ax.bar(max(ind)+3*width+1.5, np.mean(np.sum(esmtotsoc_bc[:,3:],axis=1)), width, color='white', 
                     align='center', hatch='.',label='Best-case')
#mean_14C_slow_3p = ax.bar(max(ind)+3*width+1.5, np.mean(esmtotsoc_bc[:,4]), width, 
#                          bottom=np.mean(esmtotsoc_bc[:,3]),
#                          color='white', align='center', hatch='.')
#mean_14C_pas_3p = ax.bar(max(ind)+3*width+1.5, np.mean(esmtotsoc_bc[:,5]), width, 
#                         bottom=np.mean(esmtotsoc_bc[:,3])+np.mean(esmtotsoc_bc[:,4]),
#                         color='white', align='center', hatch='.',label='Best-case')
plt.legend(loc='upper right',ncol=1,fontsize=11,labelspacing=0.2)
ax.set_ylabel(r'$\Delta$ SOC (Pg C)')
ax.set_xlim([min(ind)-.5, max(ind)+2*width+2.5])
xticksloc = np.array(list(ind+width/2.)+[max(ind)+width*2.+1.5-width/2])
xticksloc[[0,3,5]] = xticksloc[[0,3,5]]+width/2.
ax.set_xticks(xticksloc)
ax.set_xticklabels(mdl + ['Mean'])
#%% plot,  ESM get data, 2pool and 3 pool RC. show best-case only
fig, ax = plt.subplots(figsize=(10,6))
N = 5; ind = np.arange(N)
width = .25
ind = np.array([1,2])
fittedmdl_fast = ax.bar(ind, esmtotsoc[2:4,0], width, color='white', 
                        align='center', label='Fast pool', hatch='//')
C14mdl_fast = ax.bar(ind+width, esmtotsoc[2:4,2], width, color='lightgrey', 
                     align='center', hatch='//')
ind = np.array([0,3,4])
fittedmdl_fast_3p = ax.bar(ind, esmtotsoc_3p[[0,3,4],0], width, color='white', 
                     align='center', hatch='//')
C14mdl_fast_3p = ax.bar(ind+width, esmtotsoc_3p[[0,3,4],3], width, color='lightgrey', 
                     align='center', hatch='//')

ind = np.array([1,2])
fittedmdl_slow = ax.bar(ind, esmtotsoc[2:4,1], width, bottom=esmtotsoc[2:4,0], 
                        color='white', align='center', label='Slow pool', hatch='.')
C14mdl_slow = ax.bar(ind+width, esmtotsoc[2:4,3], width, bottom=esmtotsoc[2:4,2],
                     color='lightgrey', align='center', hatch='.')
ind = np.array([0,3,4])
fittedmdl_slow_3p = ax.bar(ind, esmtotsoc_3p[[0,3,4],1], width, bottom=esmtotsoc_3p[[0,3,4],0],
                     color='white', align='center', hatch='.')
C14mdl_slow_3p = ax.bar(ind+width, esmtotsoc_3p[[0,3,4],4], width, bottom=esmtotsoc_3p[[0,3,4],3],
                     color='lightgrey', align='center', hatch='.')
fittedmdl_pas_3p = ax.bar(ind, esmtotsoc_3p[[0,3,4],2], width, bottom=np.sum(esmtotsoc_3p[[0,3,4],0:2],axis=1),
                     color='white', align='center', hatch='O', label=r'Passive pool')
C14mdl_pas_3p = ax.bar(ind+width, esmtotsoc_3p[[0,3,4],5], width, bottom=np.sum(esmtotsoc_3p[[0,3,4],3:5],axis=1),
                     color='lightgrey', align='center', hatch='O')

mean_fitted_fast = ax.bar(max(ind)+width+1.5, np.mean(np.sum(esmtotsoc_bc[:,:3],axis=1)), width, color='white', 
                     align='center', hatch=None)
#mean_fitted_slow = ax.bar(max(ind)+width+1.5, np.mean(esmtotsoc[:,1]), width, bottom=np.mean(esmtotsoc[:,0]),
#                     color='white', align='center', hatch='.')
#mean_14C_fast = ax.bar(max(ind)+2*width+1.5, np.mean(esmtotsoc[:,2]), width, color='lightgrey', 
#                     align='center', hatch='//')
#mean_14C_slow = ax.bar(max(ind)+2*width+1.5, np.mean(esmtotsoc[:,3]), width, bottom=np.mean(esmtotsoc[:,2]),
#                     color='lightgrey', align='center', hatch=None)                     
mean_14C_fast_3p = ax.bar(max(ind)+2*width+1.5, np.mean(np.sum(esmtotsoc_bc[:,3:],axis=1)), width, 
                          color='lightgrey', align='center', hatch=None)
#mean_14C_slow_3p = ax.bar(max(ind)+3*width+1.5, np.mean(esmtotsoc_bc[:,4]), width, 
#                          bottom=np.mean(esmtotsoc_bc[:,3]),
#                          color='white', align='center', hatch='.')
#mean_14C_pas_3p = ax.bar(max(ind)+3*width+1.5, np.mean(esmtotsoc_bc[:,5]), width, 
#                         bottom=np.mean(esmtotsoc_bc[:,3])+np.mean(esmtotsoc_bc[:,4]),
#                         color='white', align='center', hatch='.',label='Best-case')
plt.legend(loc='upper right',ncol=1,fontsize=11,labelspacing=0.2)
ax.set_ylabel(r'$\Delta$ SOC (Pg C)')
ax.set_xlim([min(ind)-.5, max(ind)+2*width+2.5])
xticksloc = np.array(range(5)+[5.75])+width/2
ax.set_xticks(xticksloc)
ax.set_xticklabels(mdl + ['Mean'])
#%% calculate mean sink and std of best case
a=np.sum(esmtotsoc_bc[:,:3],axis=1) - np.sum(esmtotsoc_bc[:,3:],axis=1)
np.mean(a)
np.std(a)
