# -*- coding: utf-8 -*-
"""
Plot histogram of SOC of global HWSD, all profiles, and ESMs

Created on Sat Apr 11 11:58:13 2015

@author: Yujie


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import D14Cpreprocess as prep
import mynetCDF as mync
from netCDF4 import Dataset
import scipy.io
import pylab
from matplotlib.ticker import FuncFormatter

# All profiles
fromsitegrididfile = 1
if fromsitegrididfile == 1:
    filename = 'sitegridid2.txt'
    dum = np.loadtxt(filename, delimiter=',')
    proftotSOC = dum[:,6]
    profD14C= dum[:,5]   
    proflon = dum[:,0]
    proflat = dum[:,1]
else:
    filename = 'Non_peat_data_synthesis.csv'
    cutdep = 100.
    profdata = prep.getCweightedD14C2(filename,cutdep=cutdep)
    proftotSOC = profdata[profdata[:,2]==cutdep,5]
    profD14C = profdata[profdata[:,2]==cutdep,4]
# HWSD
sawtcfn = '..\\AncillaryData\\HWSD\\Regridded_ORNLDAAC\\AWT_S_SOC.nc4'
tawtcfn = '..\\AncillaryData\\HWSD\\Regridded_ORNLDAAC\\AWT_T_SOC.nc4'
ncfid = Dataset(sawtcfn, 'r')
nc_attrs, nc_dims, nc_vars = mync.ncdump(ncfid)
sawtc = ncfid.variables['SUM_s_c_1'][:]
ncfid = Dataset(tawtcfn, 'r')
nc_attrs, nc_dims, nc_vars = mync.ncdump(ncfid)
hwsdlat = ncfid.variables['lat'][:]
hwsdlon = ncfid.variables['lon'][:]
tawtc = ncfid.variables['SUM_t_c_12'][:]
hwsdsoc = sawtc + tawtc
hwsdsoc = np.ravel(hwsdsoc)
hwsdsoc[hwsdsoc<=0.] = np.nan
#%% ESM
def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(y * 100)
    return s
        
prefix = 'C:\\download\\work\\!manuscripts\\14Cboxmodel\\CMIP5_dataAnalysis\\'
mdl = ['CESM','GFDL','HadGEM','IPSL','MRI']
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(11,8))
figord = np.array([0,6,1,7,2,8,3,9,4,10])
ax = fig.axes[5];ax.set_frame_on(False)
ax.axes.get_yaxis().set_visible(False)
ax.axes.get_xaxis().set_visible(False)
ax = fig.axes[11];ax.set_frame_on(False)
ax.axes.get_yaxis().set_visible(False)
ax.axes.get_xaxis().set_visible(False)
atprofgrids = 0

c = 0 # counter
for n,mdlname in enumerate(mdl):
    print 'model is:', mdlname
    mdlsocpath = prefix + 'esmFixClim1runs\\' + mdlname + '_FixClim1_fortwoboxmodel.mat'
    if n == 4: # MRI
        mdlsocpath = prefix + 'esmFixClim1runs\\' + mdlname + '_FixClim1_annuvar.mat'    
        esm = scipy.io.loadmat(mdlsocpath)            
        esm['annucfast'] = esm['annucLitter'];
        esm['annucmedium'] =  esm['annucSoilMedium'] + esm['annucSoilSlow'];
    else:
        esm = scipy.io.loadmat(mdlsocpath)
    esmtotsoc = esm['annucfast'][:,:,1] + esm['annucmedium'][:,:,1]
    esmtotsoc[esmtotsoc==0.] = np.nan
    esmtotsoc = np.ravel(esmtotsoc)    
    if n in [0,3,4]:    
        mdlD14Cpath = prefix + 'twobox_modeling\\esmFixClim1\\' + mdlname + \
                      '\\Extrapolate_D14CSOC_3pool\\3poolmodel_3pooldata\\D14C.out'
    else:    
        mdlD14Cpath = prefix + 'twobox_modeling\\esmFixClim1\\' + mdlname + \
                      '\\Extrapolate_D14CSOC\\D14C.out'
    esmD14C = np.loadtxt(mdlD14Cpath, unpack=True, delimiter=',', skiprows=1)[2,:].T
    esmD14Cgridid = np.loadtxt(mdlD14Cpath, unpack=True, delimiter=',', skiprows=1)[0,:].T
    esmD14C[np.logical_or(esmD14C>500,esmD14C<-800)] = np.nan
    mdlD14Cpathwscalar = prefix + 'twobox_modeling\\esmFixClim1\\' + mdlname + \
                  '\\glb_sitescalar_extraD14CSOC\\extratot.out'
    esmD14Cwscalar = np.loadtxt(mdlD14Cpathwscalar, unpack=True,
                                delimiter=',', skiprows=1)[2,:].T
    esmD14Cwscalargridid = np.loadtxt(mdlD14Cpathwscalar, unpack=True,
                                delimiter=',', skiprows=1)[0,:].T
    esmD14Cwscalar[np.logical_or(esmD14Cwscalar>500,esmD14Cwscalar<-800)] = np.nan
    
    if atprofgrids == 1: # plot only grids at profile sites
        # esm        
        esmlatdim = esm['annucfast'].shape[0]
        esmlondim = esm['annucfast'].shape[1]
        latstep = 180.0 / esmlatdim; lonstep = 360.0 / esmlondim;
        lonmax = 180.0 - lonstep/2.0;
        lonmin = -180.0 + lonstep/2.0;
        latmax = 90.0 - latstep/2.0;
        latmin = -90.0 + latstep/2.0;
        esmlat = np.arange(latmax,latmin-0.1*latstep,-latstep)
        esmlon = np.arange(lonmin,lonmax+0.1*lonstep,lonstep)
        # hwsd
        hwsdlatdim = sawtc.shape[0]
        hwsdlondim = sawtc.shape[1]
        hwsdsocatsite = []; esmtotsocatsite = []; esmD14Catsite = []; esmD14Cwscalaratsite = [];
        for i, [profloni, proflati] in enumerate(zip(proflon,proflat)):
            print 'i is :', i
            # extract esm
            esmii = np.argmin((proflati - esmlat)**2)
            esmjj = np.argmin((profloni - esmlon)**2)
            hwsdii = np.argmin((proflati - hwsdlat)**2)
            hwsdjj = np.argmin((profloni - hwsdlon)**2)
            pyesmgridid = np.ravel_multi_index((esmii,esmjj), dims=(esmlatdim,esmlondim))
            matlabesmgridid = np.ravel_multi_index((esmii,esmjj), dims=(esmlatdim,esmlondim),order='F') + 1
            pyhwsdgridid = np.ravel_multi_index((hwsdii,hwsdjj), dims=(hwsdlatdim,hwsdlondim))
            hwsdsocatsite.append(hwsdsoc[pyhwsdgridid])
            esmtotsocatsite.append(esmtotsoc[pyesmgridid])
            esmD14Catsite.append(esmD14C[esmD14Cgridid==matlabesmgridid])
            esmD14Cwscalaratsite.append(esmD14Cwscalar[esmD14Cwscalargridid==matlabesmgridid])
        plt_esmtotsoc = np.asarray(filter(None,esmtotsocatsite))
        plt_esmD14C = np.asarray(filter(None,esmD14Catsite))
        plt_esmD14Cwscalar = np.asarray(filter(None,esmD14Cwscalaratsite))
        plt_hwsdsoc = np.asarray(filter(None,hwsdsocatsite))
    else:
        plt_esmtotsoc = esmtotsoc
        plt_esmD14C = esmD14C
        plt_esmD14Cwscalar = esmD14Cwscalar
        plt_hwsdsoc = hwsdsoc
    interval = 3.
    binss = 30
    ax1 = fig.axes[figord[c]]
    data = proftotSOC[~np.isnan(proftotSOC)]
    ax1.hist(data, (data.max()-data.min())/interval, weights=np.zeros_like(data)+1./data.size,
             alpha=0.3, normed=0, label='Profiles')
    data = plt_esmtotsoc[~np.isnan(plt_esmtotsoc)]
    ax1.hist(data, (data.max()-data.min())/interval, weights=np.zeros_like(data)+1./data.size,
             alpha=0.3, normed=0, label='ESM')
    data = plt_hwsdsoc[~np.isnan(plt_hwsdsoc)]
    ax1.hist(data, (data.max()-data.min())/interval, weights=np.zeros_like(data)+1./data.size, color='r',
             alpha=0.3, normed=0, label='HWSD')
    ax1.set_xlabel(r'SOC content $(kgC$ $m^{-2})$',fontsize=10)
    ax1.set_ylabel(r'Fraction',fontsize=10)
    ax1.tick_params(labelsize=8)
    if n == 4: # MRI, add legend    
        ax1.legend(bbox_to_anchor=(1.5, .5), borderaxespad=0.,loc=6,fontsize=10)
    ax1.annotate('('+chr(97+figord[c])+') '+mdlname, xy=(1., 1.01), xycoords='axes fraction',
                 fontsize=11, horizontalalignment='right', verticalalignment='bottom')
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)      
    #ax1.set_yscale('log')
    #ax1.set_ylim([0, .8])
    ax1.set_xlim([0, 80.]) 
    ax1.tick_params(top="off")
    ax1.tick_params(right="off")
    ax1.tick_params(direction='out')
    c = c + 1
    
    interval = 20.
    binss = 30
    ax2 = fig.axes[figord[c]]
    data = profD14C[~np.isnan(profD14C)]
    ax2.hist(data, (data.max()-data.min())/interval, weights=np.zeros_like(data)+1./data.size,
             alpha=0.3, normed=0, label='Profiles')
    data = plt_esmD14C[~np.isnan(plt_esmD14C)]
    ax2.hist(data, (data.max()-data.min())/interval, weights=np.zeros_like(data)+1./data.size,
             alpha=0.3, normed=0, label='ESM')
    data = plt_esmD14Cwscalar[~np.isnan(plt_esmD14Cwscalar)]
    ax2.hist(data, (data.max()-data.min())/interval, weights=np.zeros_like(data)+1./data.size,
             alpha=0.3, normed=0, color='Gold', label=r'$^{14}C$ constrained ESM')
    ax2.set_xlabel(r"$\Delta^{14}C$ ("+ u"\u2030)",fontsize=10)
    ax2.set_ylabel(r'Fraction',fontsize=10)    
    ax2.tick_params(labelsize=8)
    ax2.set_xlim((-600, 300))
    if n == 4: # MRI, add legend    
        ax2.legend(bbox_to_anchor=(1.5, .5), borderaxespad=0.,loc=6,fontsize=10)
    ax2.annotate('('+chr(96+figord[c])+') '+mdlname, xy=(1., 1.01), xycoords='axes fraction',
                 fontsize=11, horizontalalignment='right', verticalalignment='bottom')    
    ax2.tick_params(top="off")
    ax2.tick_params(right="off")
    
    c = c + 1
    plt.rc("xtick", direction="out")
    plt.rc("ytick", direction="out")
    plt.tight_layout()          
