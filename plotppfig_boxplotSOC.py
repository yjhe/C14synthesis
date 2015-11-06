# -*- coding: utf-8 -*-
"""
Plot boxplot of SOC of global HWSD, all profiles, and ESMs

Created on Sat Apr 11 11:58:13 2015

@author: Yujie


"""
#%% define data extraction function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import D14Cpreprocess as prep
import mynetCDF as mync
from netCDF4 import Dataset
import scipy.io

# All profiles and getESM data
def getESM(atprofgrids):
    ''' extract soc, 14C, 14Cwscalar from profiles, ESMs, and HWSD
    parameters: atprofgrids, indicate to extract at profile sites only or globally
    output: pltsoc, length 7, [profile, ESMs(5), HWSD]
            plt14C, length 6, [profile, ESMs(5)]
            plt14C3p, length 3, [ESM(3)]: CESM, IPSL, MRI
            plt14Cwscalar, length 6, [profile, ESMs(5)]
            mdl: ['CESM','GFDL','HadGEM','IPSL','MRI']
    '''
    fromsitegrididfile = 1
    if fromsitegrididfile == 1:
        filename = 'tot48prof.txt'
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
    
    #% ESM  2pool RC
    print '2pool RC model 2 pool data...'
    prefix = 'C:\\download\\work\\!manuscripts\\14Cboxmodel\\CMIP5_dataAnalysis\\'
    mdl = ['CESM','GFDL','HadGEM','IPSL','MRI']
    atprofgrids = atprofgrids
    pltsoc = [proftotSOC]   
    plt14C = [profD14C]
    plt14Cwscalar = [profD14C]
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
            esmtotsocatsite = []; esmD14Catsite = []; esmD14Cwscalaratsite = [];
            for i, [profloni, proflati] in enumerate(zip(proflon,proflat)):
                print 'i is :', i
                # extract esm
                esmii = np.argmin((proflati - esmlat)**2)
                esmjj = np.argmin((profloni - esmlon)**2)
                pyesmgridid = np.ravel_multi_index((esmii,esmjj), dims=(esmlatdim,esmlondim))
                matlabesmgridid = np.ravel_multi_index((esmii,esmjj), dims=(esmlatdim,esmlondim),order='F') + 1
                esmtotsocatsite.append(esmtotsoc[pyesmgridid])
                esmD14Catsite.append(esmD14C[esmD14Cgridid==matlabesmgridid])
                esmD14Cwscalaratsite.append(esmD14Cwscalar[esmD14Cwscalargridid==matlabesmgridid])
            plt_esmtotsoc = np.asarray(filter(None,esmtotsocatsite))
            plt_esmD14C = np.asarray(filter(None,esmD14Catsite))
            plt_esmD14Cwscalar = np.asarray(filter(None,esmD14Cwscalaratsite))
            plt_esmtotsoc = plt_esmtotsoc[~np.isnan(plt_esmtotsoc)]
            plt_esmD14C = plt_esmD14C[~np.isnan(plt_esmD14C)]
            plt_esmD14Cwscalar = plt_esmD14Cwscalar[~np.isnan(plt_esmD14Cwscalar )]
            if n == 0:
                # hwsd
                hwsdlatdim = sawtc.shape[0]
                hwsdlondim = sawtc.shape[1]
                hwsdsocatsite = []; 
                for i, [profloni, proflati] in enumerate(zip(proflon,proflat)):
                    print 'i is :', i
                    hwsdii = np.argmin((proflati - hwsdlat)**2)
                    hwsdjj = np.argmin((profloni - hwsdlon)**2)
                    pyhwsdgridid = np.ravel_multi_index((hwsdii,hwsdjj), dims=(hwsdlatdim,hwsdlondim))
                    hwsdsocatsite.append(hwsdsoc[pyhwsdgridid])        
                plt_hwsdsoc = np.asarray(filter(None,hwsdsocatsite))        
                plt_hwsdsoc = plt_hwsdsoc[~np.isnan(plt_hwsdsoc)]
        else:
            plt_esmtotsoc = esmtotsoc[~np.isnan(esmtotsoc)]
            plt_esmD14C = esmD14C[~np.isnan(esmD14C)]
            plt_esmD14Cwscalar = esmD14Cwscalar[~np.isnan(esmD14Cwscalar)]
            plt_hwsdsoc = hwsdsoc[~np.isnan(hwsdsoc)]
        pltsoc.append(plt_esmtotsoc)
        plt14C.append(plt_esmD14C)
        plt14Cwscalar.append(plt_esmD14Cwscalar)
    pltsoc.append(plt_hwsdsoc) 
    
    # ESM, 3pool RC    
    print '3pool RC model 3 pool data...'
    prefix = 'C:\\download\\work\\!manuscripts\\14Cboxmodel\\CMIP5_dataAnalysis\\'
    mdl3p = ['CESM','IPSL','MRI']
    plt14C3p = []
    for n,mdlname in enumerate(mdl3p):
        print 'model is:', mdlname
        mdlsocpath = prefix + 'esmFixClim1runs\\' + mdlname + '_FixClim1_fortwoboxmodel.mat'
        if n == 2: # MRI
            mdlsocpath = prefix + 'esmFixClim1runs\\' + mdlname + '_FixClim1_annuvar.mat'    
            esm = scipy.io.loadmat(mdlsocpath)            
            esm['annucfast'] = esm['annucLitter'];
            esm['annucmedium'] =  esm['annucSoilMedium'] + esm['annucSoilSlow'];
        else:
            esm = scipy.io.loadmat(mdlsocpath)

        mdlD14Cpath = prefix + 'twobox_modeling\\esmFixClim1\\' + mdlname + \
                      '\\Extrapolate_D14CSOC_3pool\\3poolmodel_3pooldata\\D14C.out'
        esmD14C = np.loadtxt(mdlD14Cpath, unpack=True, delimiter=',', skiprows=1)[2,:].T
        esmD14Cgridid = np.loadtxt(mdlD14Cpath, unpack=True, delimiter=',', skiprows=1)[0,:].T
        esmD14C[np.logical_or(esmD14C>500,esmD14C<-800)] = np.nan
        
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
            esmD14Catsite = []; 
            for i, [profloni, proflati] in enumerate(zip(proflon,proflat)):
                print 'i is :', i
                # extract esm
                esmii = np.argmin((proflati - esmlat)**2)
                esmjj = np.argmin((profloni - esmlon)**2)
                pyesmgridid = np.ravel_multi_index((esmii,esmjj), dims=(esmlatdim,esmlondim))
                matlabesmgridid = np.ravel_multi_index((esmii,esmjj), dims=(esmlatdim,esmlondim),order='F') + 1
                esmD14Catsite.append(esmD14C[esmD14Cgridid==matlabesmgridid])
            plt_esmD14C = np.asarray(filter(None,esmD14Catsite))
            plt_esmD14C = plt_esmD14C[~np.isnan(plt_esmD14C)]
        else:
            plt_esmD14C = esmD14C[~np.isnan(esmD14C)]
        plt14C3p.append(plt_esmD14C)
    return pltsoc, plt14C, plt14C3p, plt14Cwscalar, mdl
#%%  plot on 3 panels
atprofgrids = 1
pltsoc, plt14C, plt14Cwscalar, mdl = getESM(atprofgrids)
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8,6))

ax1 = fig.axes[0]
ax1.boxplot(pltsoc, notch=0, sym='+', vert=1, whis=1.5)
ax1.set_ylabel('SOC in top 1m',fontsize=10)
ax1.set_xlim(0.5, 7+0.5)
ax1.set_xticklabels(['Profiles']+mdl+['HWSD'],fontsize=10)
ax1.annotate('(a)', xy=(.05, .85), xycoords='axes fraction',
             fontsize=13, horizontalalignment='right', verticalalignment='bottom')
ax2 = fig.axes[1]
ax2.boxplot(plt14C, notch=0, sym='+', vert=1, whis=1.5)
ax2.set_ylabel(r"$\Delta14C$ ("+ u"\u2030)"+'of fitted ESMs',fontsize=10)
ax2.set_xlim(0.5, 6+0.5)
ax2.set_xticklabels(['Profiles']+mdl,fontsize=10)
ax2.annotate('(b)', xy=(.05, .85), xycoords='axes fraction',
             fontsize=13, horizontalalignment='right', verticalalignment='bottom')

ax3 = fig.axes[2]
bp = ax3.boxplot(plt14Cwscalar, notch=0, sym='+', vert=1, whis=1.5)
ax3.set_ylabel(r"$\Delta14C$ ("+ u"\u2030)"+'of \n 14C constrained ESMs',fontsize=10)
ax3.set_xlim(0.5, 6+0.5)
ax3.set_xticklabels(['Profiles']+mdl,fontsize=10)
ax3.annotate('(c)', xy=(.05, .85), xycoords='axes fraction',
             fontsize=13, horizontalalignment='right', verticalalignment='bottom')
plt.tight_layout()          
#%%  plot on 2 panels, at profiles
atprofgrids = 1
pltsoc, plt14C, plt14Cwscalar, mdl = getESM(atprofgrids)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,6))
meanprops = dict(marker='*', markeredgecolor='black', markersize=10,
                 markerfacecolor='firebrick')
ax1 = fig.axes[0]
bp1 = ax1.boxplot([pltsoc[i] for i in [0,6,1,2,3,4,5]], notch=0, sym='', vert=1, whis=1.5, showmeans=True, meanprops=meanprops)
plt.setp(bp1['boxes'], color='black')
plt.setp(bp1['whiskers'], color='black')
#plt.setp(bp1['fliers'], color='red', marker='+')
ax1.set_ylabel('SOC in top 1m $(kgC m^{-2})$',fontsize=13)
ax1.set_xlim(0.5, 7+0.5)
ax1.set_xticklabels(['Profiles']+['HWSD']+mdl,fontsize=12)
ax1.annotate('(a)', xy=(.05, .85), xycoords='axes fraction',
             fontsize=13, horizontalalignment='right', verticalalignment='bottom')
ax2 = fig.axes[1]
bp2 = ax2.boxplot(plt14C, notch=0, sym='', vert=1, whis=1.5, showmeans=True, meanprops=meanprops)
plt.setp(bp2['boxes'], color='black')
plt.setp(bp2['whiskers'], color='black')
#plt.setp(bp2['fliers'], color='red', marker='+')
ax2.set_xlim(0.5, 6+0.5)
ax2.annotate('(b)', xy=(.05, .85), xycoords='axes fraction',
             fontsize=13, horizontalalignment='right', verticalalignment='bottom')

meanprops2 = dict(marker='*', markeredgecolor='black', markersize=10,
                  markerfacecolor='dodgerblue')
capprops = dict(linestyle='-', linewidth=1, color='blue')
bp22 = ax2.boxplot(plt14Cwscalar[1:], notch=0, sym='', vert=1, whis=1.5, positions=range(2,7),
                   showmeans=True, meanprops=meanprops2, capprops=capprops)
plt.setp(bp22['boxes'], color='blue')
plt.setp(bp22['whiskers'], color='blue')
#plt.setp(bp22['fliers'], color='blue', marker='+')
ax2.set_ylabel(r"C-averaged $\Delta14C$ ("+ u"\u2030)",fontsize=13)
ax2.set_xlim(0.5, 6+0.5)
ax2.set_xticks(range(1,7))
ax2.set_xticklabels(['Profiles']+mdl,fontsize=12)
plt.legend()
plt.tight_layout()          
#%%  plot on 2 panels, global, drop profiles
atprofgrids = 0
pltsoc, plt14C, plt14Cwscalar, mdl = getESM(atprofgrids)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,6))
meanprops = dict(marker='*', markeredgecolor='black', markersize=10,
                 markerfacecolor='firebrick')
ax1 = fig.axes[0]
bp1 = ax1.boxplot([pltsoc[i] for i in [6,1,2,3,4,5]], notch=0, sym='', vert=1, whis=1.5, showmeans=True, meanprops=meanprops)
plt.setp(bp1['boxes'], color='black')
plt.setp(bp1['whiskers'], color='black')
#plt.setp(bp1['fliers'], color='red', marker='+')
ax1.set_ylabel('SOC in top 1m $(kgC m^{-2})$',fontsize=13)
ax1.set_xlim(0.5, 6+0.5)
ax1.set_xticklabels(['HWSD']+mdl,fontsize=12)
ax1.annotate('(a)', xy=(.05, .85), xycoords='axes fraction',
             fontsize=13, horizontalalignment='right', verticalalignment='bottom')
ax2 = fig.axes[1]
bp2 = ax2.boxplot(plt14C[1:], notch=0, sym='', vert=1, whis=1.5, showmeans=True, meanprops=meanprops)
plt.setp(bp2['boxes'], color='black')
plt.setp(bp2['whiskers'], color='black')
#plt.setp(bp2['fliers'], color='red', marker='+')
ax2.set_xlim(0.5, 5+0.5)
ax2.annotate('(b)', xy=(.05, .85), xycoords='axes fraction',
             fontsize=13, horizontalalignment='right', verticalalignment='bottom')

meanprops2 = dict(marker='*', markeredgecolor='black', markersize=10,
                  markerfacecolor='dodgerblue')
capprops = dict(linestyle='-', linewidth=1, color='blue')
bp22 = ax2.boxplot(plt14Cwscalar[1:], notch=0, sym='', vert=1, whis=1.5, positions=range(1,6),
                   showmeans=True, meanprops=meanprops2, capprops=capprops)
plt.setp(bp22['boxes'], color='blue')
plt.setp(bp22['whiskers'], color='blue')
#plt.setp(bp22['fliers'], color='blue', marker='+')
ax2.set_ylabel(r"C-averaged $\Delta14C$ ("+ u"\u2030)",fontsize=13)
ax2.set_xlim(0.5, 5+0.5)
ax2.set_xticks(range(1,6))
ax2.set_xticklabels(mdl,fontsize=12)
plt.legend()
plt.tight_layout()          
#%% plot site profiles and global in one figure, 3*2
# at sites
atprofgrids = 1
pltsoc, plt14C, plt14Cwscalar, mdl = getESM(atprofgrids)
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,8))
textpos = (.07, .85)
meanprops = dict(marker='*', markeredgecolor='black', markersize=10,
                 markerfacecolor='firebrick')
meanprops2 = dict(marker='*', markeredgecolor='black', markersize=10,
                  markerfacecolor='dodgerblue')
capprops = dict(linestyle='-', linewidth=1, color='blue')
ax1 = fig.axes[0]
bp1 = ax1.boxplot([pltsoc[i] for i in [0,6,1,2,3,4,5]], notch=0, sym='+', vert=1, whis=1.5,
            showmeans=True, meanprops=meanprops)
plt.setp(bp1['boxes'][:2], color='black')
plt.setp(bp1['whiskers'][:4], color='black')
plt.setp(bp1['fliers'][:2], color='red', marker='+')
plt.setp(bp1['boxes'][2:], color='blue')
plt.setp(bp1['whiskers'][4:], color='blue')
plt.setp(bp1['fliers'][2:], color='blue', marker='+')
plt.setp(bp1['means'][2:], **meanprops2)
plt.setp(bp1['caps'][4:], color='blue')
ax1.set_ylabel('SOC $(kgC$ $m^{-2})$',fontsize=12)
ax1.set_xlim(0.5, 7+0.5)
ax1.set_xticks(range(1,8))
ax1.set_xticklabels(['Profiles']+['HWSD']+mdl,fontsize=11)
ax1.annotate('(a)', xy=textpos, xycoords='axes fraction',
             fontsize=13, horizontalalignment='right', verticalalignment='bottom')
ax1.tick_params(top="off")
ax1.tick_params(right="off")             
ax1.annotate('Sites', xy=(0.5,1.02), xycoords='axes fraction',
             fontsize=13, horizontalalignment='center', verticalalignment='bottom')

ax2 = fig.axes[2]
bp2 = ax2.boxplot(plt14C, notch=0, sym='+', vert=1, whis=1.5, showmeans=True, 
                  meanprops=meanprops, positions=[1]+range(3,8))
plt.setp(bp2['boxes'][0], color='black')
plt.setp(bp2['whiskers'][:2], color='black')
plt.setp(bp2['fliers'][0], color='red', marker='+')
plt.setp(bp2['boxes'][1:], color='blue')
plt.setp(bp2['whiskers'][2:], color='blue')
plt.setp(bp2['fliers'][1:], color='blue', marker='+')
plt.setp(bp2['means'][1:], **meanprops2)
plt.setp(bp2['caps'][2:], color='blue')
ax2.set_ylabel(r"$\Delta^{14}C$ ("+ u"\u2030) of RC model\noptimized to ESMs",fontsize=12)
ax2.set_ylim(-600, 300)
ax2.set_xlim(0.5, 7+0.5)
ax2.set_xticks([1] + range(3,8))
ax2.set_xticklabels(['Profiles']+mdl,fontsize=11)
ax2.annotate('(c)', xy=textpos, xycoords='axes fraction',
             fontsize=13, horizontalalignment='right', verticalalignment='bottom')
ax2.tick_params(top="off")
ax2.tick_params(right="off")         


ax3 = fig.axes[4]
bp3 = ax3.boxplot(plt14Cwscalar, notch=0, sym='+', vert=1, whis=1.5, 
                  showmeans=True, meanprops=meanprops, positions=[1]+range(3,8))
plt.setp(bp3['boxes'][0], color='black')
plt.setp(bp3['whiskers'][:2], color='black')
plt.setp(bp3['fliers'][0], color='red', marker='+')
plt.setp(bp3['boxes'][1:], color='blue')
plt.setp(bp3['whiskers'][2:], color='blue')
plt.setp(bp3['fliers'][1:], color='blue', marker='+')
plt.setp(bp3['means'][1:], **meanprops2)
plt.setp(bp3['caps'][2:], color='blue')
ax3.set_ylabel(r"$\Delta^{14}C$ ("+ u"\u2030) of\n$^{14}C$ constrained model",
               fontsize=12)
ax3.set_xlim(0.5, 7+0.5)
ax3.set_ylim(-600, 300)
ax3.set_xticks([1] + range(3,8))
ax3.set_xticklabels(['Profiles']+mdl,fontsize=11)
ax3.annotate('(e)', xy=textpos, xycoords='axes fraction',
             fontsize=13, horizontalalignment='right', verticalalignment='bottom')
ax3.tick_params(top="off")
ax3.tick_params(right="off")         
plt.tight_layout()          

# plot global
atprofgrids = 0
pltsoc, plt14C, plt14Cwscalar, mdl = getESM(atprofgrids)
ax4 = fig.axes[1]
bp4 = ax4.boxplot([pltsoc[i] for i in [6,1,2,3,4,5]], notch=0, sym='', vert=1, whis=1.5,
            showmeans=True, meanprops=meanprops)
plt.setp(bp4['boxes'][0], color='black')
plt.setp(bp4['whiskers'][:2], color='black')
plt.setp(bp4['boxes'][1:], color='blue')
plt.setp(bp4['whiskers'][2:], color='blue')
plt.setp(bp4['means'][1:], **meanprops2)
plt.setp(bp4['caps'][2:], color='blue')
#ax4.set_ylabel('SOC in top 1m $(kgC m^{-2})$',fontsize=12)
ax4.set_ylim(0, 70)
ax4.set_xlim(0.5, 6+0.5)
ax4.set_xticklabels(['HWSD']+mdl,fontsize=11)
ax4.set_yticklabels([''])
ax4.annotate('(b)', xy=textpos, xycoords='axes fraction',
             fontsize=13, horizontalalignment='right', verticalalignment='bottom')
ax4.tick_params(top="off")
ax4.tick_params(right="off")         
ax4.annotate('Global', xy=(0.5,1.02), xycoords='axes fraction',
             fontsize=13, horizontalalignment='center', verticalalignment='bottom')

ax5 = fig.axes[3]
bp5 = ax5.boxplot(plt14C[1:], notch=0, sym='', vert=1, whis=1.5, showmeans=True, 
                  positions=range(2,7),meanprops=meanprops2, capprops=capprops)
plt.setp(bp5['boxes'], color='blue')
plt.setp(bp5['whiskers'], color='blue')
#ax5.set_ylabel(r"$\Delta14C$ ("+ u"\u2030) of fitted model",fontsize=12)
ax5.set_xlim(0.5, 6+0.5)
ax5.set_ylim(-600, 300)
ax5.set_xticklabels(mdl,fontsize=11)
ax5.annotate('(d)', xy=textpos, xycoords='axes fraction',
             fontsize=13, horizontalalignment='right', verticalalignment='bottom')
ax5.set_yticklabels([''])
ax5.tick_params(top="off")
ax5.tick_params(right="off")         


ax6 = fig.axes[5]
bp6 = ax6.boxplot(plt14Cwscalar[1:], notch=0, sym='', vert=1, whis=1.5, positions=range(2,7),
                   showmeans=True, meanprops=meanprops2, capprops=capprops)
plt.setp(bp6['boxes'], color='blue')
plt.setp(bp6['whiskers'], color='blue')
#ax6.set_ylabel(r"$\Delta14C$ ("+ u"\u2030) of\n$^{14}C$ constrained model",fontsize=12)
ax6.set_xlim(0.5, 6+0.5)
ax6.set_ylim(-600, 300)
ax6.set_xticklabels(mdl,fontsize=11)
ax6.annotate('(f)', xy=textpos, xycoords='axes fraction',
             fontsize=13, horizontalalignment='right', verticalalignment='bottom')
ax6.set_yticklabels([''])
ax6.tick_params(top="off")
ax6.tick_params(right="off")   
plt.rc("xtick", direction="out")
plt.rc("ytick", direction="out")      
plt.tight_layout()          

#%% paired-t test
import scipy.stats as stats
mdl = ['CESM','GFDL','HadGEM','IPSL','MRI']
# at sites
atprofgrids = 1
pltsoc, plt14C, plt14C3p, plt14Cwscalar, mdl = getESM(atprofgrids)
for i in range(1,6):
    print 'SOC: test profiles vs. ESM: ',mdl[i-1]
    print stats.ttest_ind(pltsoc[0], pltsoc[i], equal_var=False)[1]
    print stats.levene(pltsoc[0], pltsoc[i],center='median')[1]
    print stats.bartlett(pltsoc[0], pltsoc[i])[1]
    print stats.mannwhitneyu(pltsoc[0], pltsoc[i])[1]
for i in range(1,6):
    print 'SOC: test HWSD vs. ESM: ',mdl[i-1]
    print stats.ttest_ind(pltsoc[6], pltsoc[i], equal_var=False)[1]
    print stats.levene(pltsoc[6], pltsoc[i],center='median')[1]
    print stats.bartlett(pltsoc[6], pltsoc[i])[1]
    print stats.mannwhitneyu(pltsoc[6], pltsoc[i])[1]
for i in range(1,6):
    print 'ESM 14C: test profiles vs. ESM: ',mdl[i-1]
    print stats.ttest_ind(plt14C[0], plt14C[i], equal_var=False)[1]
    print stats.levene(plt14C[0], plt14C[i],center='median')[1]
    print stats.bartlett(plt14C[0], plt14C[i])[1]
    print stats.mannwhitneyu(plt14C[0], plt14C[i])[1]
for i in range(1,6):
    print 'ESM 14Cwscalar: test profiles vs. ESM: ',mdl[i-1]
    print stats.ttest_ind(plt14Cwscalar[0], plt14Cwscalar[i], equal_var=False)[1]
    print stats.levene(plt14Cwscalar[0], plt14Cwscalar[i],center='median')[1]
    print stats.bartlett(plt14Cwscalar[0], plt14Cwscalar[i])[1]
    print stats.mannwhitneyu(plt14Cwscalar[0], plt14Cwscalar[i])[1]
    
# plot global
atprofgrids = 0
pltsocglb, plt14Cglb, plt14C3pglb, plt14Cwscalarglb, mdl = getESM(atprofgrids)
for i in range(1,6):
    print 'SOC: test HWSD vs. ESM: ',mdl[i-1]
    print stats.ttest_ind(pltsocglb[6], pltsocglb[i], equal_var=False)[1]
    print stats.levene(pltsocglb[6], pltsocglb[i],center='median')[1]
    print stats.bartlett(pltsocglb[6], pltsocglb[i])[1]
    print stats.mannwhitneyu(pltsocglb[6], pltsocglb[i])[1]

#%% convert 14C to turnover time
import C14tools
# at global
atprofgrids = 0
pltsoc, plt14C, plt14C3p, plt14Cwscalar, mdl = getESM(atprofgrids)
profsmpyr = np.loadtxt('tot48prof.txt', delimiter=',')[:,4]
plttau = []
for n,i in enumerate(plt14C):
    if n == 0:
        plttau.append(C14tools.cal_tau(i, profsmpyr, 1, 1))
    else:
        plttau.append(C14tools.cal_tau(i, np.ones(len(i))*1990, 1, 1))
np.save('globalESMstau.npy',plttau)

# at global, 3pool RC
atprofgrids = 0
pltsoc, plt14C, plt14C3p, plt14Cwscalar, mdl = getESM(atprofgrids)
profsmpyr = np.loadtxt('tot48prof.txt', delimiter=',')[:,4]
plttau3p = []
for n,i in enumerate(plt14C3p):
        plttau3p.append(C14tools.cal_tau(i, np.ones(len(i))*1990, 1, 1))
np.save('globalESMstau_3p.npy',plttau3p)
#%% plot (add) turnover time of site profiles and global in one figure, 4*2
# 2 pool RC results only
import C14tools
#----------- at sites
atprofgrids = 1
pltsoc, plt14C, plt14Cwscalar, mdl = getESM(atprofgrids)
profsmpyr = np.loadtxt('tot48prof.txt', delimiter=',')[:,4]
plttau = []
for n,i in enumerate(plt14C):
    if n == 0:
        plttau.append(C14tools.cal_tau(i, profsmpyr, 1, 1))
    else:
        plttau.append(C14tools.cal_tau(i, np.ones(len(i))*1990, 1, 1))
#% 
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8,8))
ylabsz = 10; xticklabsz = 10; yticklabsz = 9; annosz = 11
textpos = (.95, .85) # upper right
textpos = (.1, 1.02) # upper left outside
meanprops = dict(marker='*', markeredgecolor='black', markersize=10,
                 markerfacecolor='firebrick')
meanprops2 = dict(marker='*', markeredgecolor='black', markersize=10,
                  markerfacecolor='dodgerblue')
capprops = dict(linestyle='-', linewidth=1, color='blue')
ax1 = fig.axes[0]
bp1 = ax1.boxplot([pltsoc[i] for i in [0,6,1,2,3,4,5]], notch=0, sym='+', vert=1, whis=1.5,
            showmeans=True, meanprops=meanprops)
plt.setp(bp1['boxes'][:2], color='black')
plt.setp(bp1['whiskers'][:4], color='black')
plt.setp(bp1['fliers'][:2], color='red', marker='+')
plt.setp(bp1['boxes'][2:], color='blue')
plt.setp(bp1['whiskers'][4:], color='blue')
plt.setp(bp1['fliers'][2:], color='blue', marker='+')
plt.setp(bp1['means'][2:], **meanprops2)
plt.setp(bp1['caps'][4:], color='blue')
ax1.set_ylabel('SOC $(kgC$ $m^{-2})$',fontsize=ylabsz)
ax1.set_xlim(0.5, 7+0.5)
ax1.set_xticks(range(1,8))
ax1.set_xticklabels(['Profiles']+['HWSD']+mdl,fontsize=xticklabsz)
plt.setp(ax1.get_yticklabels(), fontsize=yticklabsz)
ax1.annotate('(a)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax1.tick_params(top="off")
ax1.tick_params(right="off")             
ax1.annotate('Sites', xy=(0.5,1.02), xycoords='axes fraction',
             fontsize=13, horizontalalignment='center', verticalalignment='bottom')

ax2 = fig.axes[2]
bp2 = ax2.boxplot(plt14C, notch=0, sym='+', vert=1, whis=1.5, showmeans=True, 
                  meanprops=meanprops, positions=[1]+range(3,8))
plt.setp(bp2['boxes'][0], color='black')
plt.setp(bp2['whiskers'][:2], color='black')
plt.setp(bp2['fliers'][0], color='red', marker='+')
plt.setp(bp2['boxes'][1:], color='blue')
plt.setp(bp2['whiskers'][2:], color='blue')
plt.setp(bp2['fliers'][1:], color='blue', marker='+')
plt.setp(bp2['means'][1:], **meanprops2)
plt.setp(bp2['caps'][2:], color='blue')
ax2.set_ylabel(r"$\Delta^{14}C$ ("+ u"\u2030) of RC model\noptimized to ESMs",fontsize=ylabsz)
ax2.set_ylim(-600, 300)
ax2.set_xlim(0.5, 7+0.5)
ax2.set_xticks([1] + range(3,8))
ax2.set_xticklabels(['Profiles']+mdl,fontsize=xticklabsz)
plt.setp(ax2.get_yticklabels(), fontsize=yticklabsz)
ax2.annotate('(c)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax2.tick_params(top="off")
ax2.tick_params(right="off")         

ax21 = fig.axes[4]
bp21 = ax21.boxplot(plttau, notch=0, sym='+', vert=1, whis=1.5, showmeans=True, 
                  meanprops=meanprops, positions=[1]+range(3,8))
plt.setp(bp21['boxes'][0], color='black')
plt.setp(bp21['whiskers'][:2], color='black')
plt.setp(bp21['fliers'][0], color='red', marker='+')
plt.setp(bp21['boxes'][1:], color='blue')
plt.setp(bp21['whiskers'][2:], color='blue')
plt.setp(bp21['fliers'][1:], color='blue', marker='+')
plt.setp(bp21['means'][1:], **meanprops2)
plt.setp(bp21['caps'][2:], color='blue')
ax21.set_ylabel("Turnover time (yr)\nof RC model\noptimized to ESMs",fontsize=ylabsz)
#ax21.set_yscale("log")
ax21.set_xlim(0.5, 7+0.5)
ax21.set_ylim(0, 5500)
ax21.set_xticks([1] + range(3,8))
ax21.set_xticklabels(['Profiles']+mdl,fontsize=xticklabsz)
plt.setp(ax21.get_yticklabels(), fontsize=yticklabsz)
ax21.annotate('(e)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax21.tick_params(top="off")
ax21.tick_params(right="off")         

ax3 = fig.axes[6]
bp3 = ax3.boxplot(plt14Cwscalar, notch=0, sym='+', vert=1, whis=1.5, 
                  showmeans=True, meanprops=meanprops, positions=[1]+range(3,8))
plt.setp(bp3['boxes'][0], color='black')
plt.setp(bp3['whiskers'][:2], color='black')
plt.setp(bp3['fliers'][0], color='red', marker='+')
plt.setp(bp3['boxes'][1:], color='blue')
plt.setp(bp3['whiskers'][2:], color='blue')
plt.setp(bp3['fliers'][1:], color='blue', marker='+')
plt.setp(bp3['means'][1:], **meanprops2)
plt.setp(bp3['caps'][2:], color='blue')
ax3.set_ylabel(r"$\Delta^{14}C$ ("+ u"\u2030) of $^{14}C$\nconstrained model",
               fontsize=ylabsz)
ax3.set_xlim(0.5, 7+0.5)
ax3.set_ylim(-600, 300)
ax3.set_xticks([1] + range(3,8))
ax3.set_xticklabels(['Profiles']+mdl,fontsize=xticklabsz)
plt.setp(ax3.get_yticklabels(), fontsize=yticklabsz)
ax3.annotate('(g)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax3.tick_params(top="off")
ax3.tick_params(right="off")         
plt.tight_layout()          

#%----------------- plot global
atprofgrids = 0
pltsoc, plt14C, plt14Cwscalar, mdl = getESM(atprofgrids)
plttau = np.load('globalESMstau.npy')
ax4 = fig.axes[1]
bp4 = ax4.boxplot([pltsoc[i] for i in [6,1,2,3,4,5]], notch=0, sym='', vert=1, whis=1.5,
            showmeans=True, meanprops=meanprops)
plt.setp(bp4['boxes'][0], color='black')
plt.setp(bp4['whiskers'][:2], color='black')
plt.setp(bp4['boxes'][1:], color='blue')
plt.setp(bp4['whiskers'][2:], color='blue')
plt.setp(bp4['means'][1:], **meanprops2)
plt.setp(bp4['caps'][2:], color='blue')
#ax4.set_ylabel('SOC in top 1m $(kgC m^{-2})$',fontsize=12)
ax4.set_ylim(0, 70)
ax4.set_xlim(0.5, 6+0.5)
ax4.set_xticklabels(['HWSD']+mdl,fontsize=xticklabsz)
ax4.set_yticklabels([''], fontsize=yticklabsz)
ax4.annotate('(b)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax4.tick_params(top="off")
ax4.tick_params(right="off")         
ax4.annotate('Global', xy=(0.5,1.02), xycoords='axes fraction',
             fontsize=13, horizontalalignment='center', verticalalignment='bottom')

ax5 = fig.axes[3]
bp5 = ax5.boxplot(plt14C[1:], notch=0, sym='', vert=1, whis=1.5, showmeans=True, 
                  positions=range(2,7),meanprops=meanprops2, capprops=capprops)
plt.setp(bp5['boxes'], color='blue')
plt.setp(bp5['whiskers'], color='blue')
#ax5.set_ylabel(r"$\Delta14C$ ("+ u"\u2030) of fitted model",fontsize=12)
ax5.set_xlim(0.5, 6+0.5)
ax5.set_ylim(-600, 300)
ax5.set_xticklabels(mdl,fontsize=xticklabsz)
ax5.annotate('(d)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax5.set_yticklabels([''], fontsize=yticklabsz)
ax5.tick_params(top="off")
ax5.tick_params(right="off")         

ax51 = fig.axes[5]
bp51 = ax51.boxplot(plttau[1:], notch=0, sym='', vert=1, whis=1.5, showmeans=True, 
                  positions=range(2,7),meanprops=meanprops2, capprops=capprops)
plt.setp(bp51['boxes'], color='blue')
plt.setp(bp51['whiskers'], color='blue')
#ax51.set_yscale("log", nonposy='clip')
ax51.set_xlim(0.5, 6+0.5)
ax51.set_ylim(0, 5500)
ax51.set_xticklabels(mdl,fontsize=xticklabsz)
ax51.annotate('(f)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax51.set_yticklabels([''], fontsize=yticklabsz)
ax51.tick_params(top="off")
ax51.tick_params(right="off")         

ax6 = fig.axes[7]
bp6 = ax6.boxplot(plt14Cwscalar[1:], notch=0, sym='', vert=1, whis=1.5, positions=range(2,7),
                   showmeans=True, meanprops=meanprops2, capprops=capprops)
plt.setp(bp6['boxes'], color='blue')
plt.setp(bp6['whiskers'], color='blue')
#ax6.set_ylabel(r"$\Delta14C$ ("+ u"\u2030) of\n$^{14}C$ constrained model",fontsize=12)
ax6.set_xlim(0.5, 6+0.5)
ax6.set_ylim(-600, 300)
ax6.set_xticklabels(mdl,fontsize=xticklabsz)
ax6.annotate('(h)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax6.set_yticklabels([''], fontsize=yticklabsz)
ax6.tick_params(top="off")
ax6.tick_params(right="off")   
plt.rc("xtick", direction="out")
plt.rc("ytick", direction="out")      
plt.tight_layout()   


#%% plot (add) turnover time of site profiles and global in one figure, 4*2
# 2 pool RC results + 3 pool RC results
import C14tools
#----------- at sites --------------
atprofgrids = 1
pltsoc, plt14C, plt14C3p, plt14Cwscalar, mdl = getESM(atprofgrids)
profsmpyr = np.loadtxt('tot48prof.txt', delimiter=',')[:,4]
plttau = []
for n,i in enumerate(plt14C):
    if n == 0:
        plttau.append(C14tools.cal_tau(i, profsmpyr, 1, 1))
    else:
        plttau.append(C14tools.cal_tau(i, np.ones(len(i))*1990, 1, 1))
plttau3p = []
for n,i in enumerate(plt14C3p):
    plttau3p.append(C14tools.cal_tau(i, np.ones(len(i))*1990, 1, 1))
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8,8))
ylabsz = 10; xticklabsz = 10; yticklabsz = 9; annosz = 11
textpos = (.95, .85) # upper right
textpos = (.1, 1.02) # upper left outside
meanprops = dict(marker='*', markeredgecolor='black', markersize=10,
                 markerfacecolor='firebrick')
meanprops2 = dict(marker='*', markeredgecolor='black', markersize=10,
                  markerfacecolor='dodgerblue')
meanprops3 = dict(marker='*', markeredgecolor='black', markersize=10,
                  markerfacecolor='green') # for 3pool results
capprops = dict(linestyle='-', linewidth=1, color='blue')
ax1 = fig.axes[0]
bp1 = ax1.boxplot([pltsoc[i] for i in [0,6,1,2,3,4,5]], notch=0, sym='+', vert=1, whis=1.5,
            showmeans=True, meanprops=meanprops)
plt.setp(bp1['boxes'][:2], color='black')
plt.setp(bp1['whiskers'][:4], color='black')
plt.setp(bp1['fliers'][:2], color='red', marker='+')
plt.setp(bp1['boxes'][2:], color='blue')
plt.setp(bp1['whiskers'][4:], color='blue')
plt.setp(bp1['fliers'][2:], color='blue', marker='+')
plt.setp(bp1['means'][2:], **meanprops2)
plt.setp(bp1['caps'][4:], color='blue')
ax1.set_ylabel('SOC $(kgC$ $m^{-2})$',fontsize=ylabsz)
ax1.set_xlim(0.5, 7+0.5)
ax1.set_xticks(range(1,8))
ax1.set_xticklabels(['Profiles']+['HWSD']+mdl,fontsize=xticklabsz)
plt.setp(ax1.get_yticklabels(), fontsize=yticklabsz)
ax1.annotate('(a)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax1.tick_params(top="off")
ax1.tick_params(right="off")             
ax1.annotate('Sites', xy=(0.5,1.02), xycoords='axes fraction',
             fontsize=13, horizontalalignment='center', verticalalignment='bottom')
# plot 2pool RC
ax2 = fig.axes[2]
bp2 = ax2.boxplot(plt14C, notch=0, sym='+', vert=1, whis=1.5, showmeans=True, 
                  meanprops=meanprops, positions=[1]+range(3,8))
plt.setp(bp2['boxes'][0], color='black')
plt.setp(bp2['whiskers'][:2], color='black')
plt.setp(bp2['fliers'][0], color='red', marker='+')
plt.setp(bp2['boxes'][1:], color='blue')
plt.setp(bp2['whiskers'][2:], color='blue')
plt.setp(bp2['fliers'][1:], color='blue', marker='+')
plt.setp(bp2['means'][1:], **meanprops2)
plt.setp(bp2['caps'][2:], color='blue')
ax2.set_ylabel(r"$\Delta^{14}C$ ("+ u"\u2030) of RC model\noptimized to ESMs",fontsize=ylabsz)
ax2.set_ylim(-600, 300)
ax2.set_xlim(0.5, 7+0.5)
ax2.set_xticks([1] + range(3,8))
ax2.set_xticklabels(['Profiles']+mdl,fontsize=xticklabsz)
# plot 3pool RC
bp20 = ax2.boxplot(plt14C3p, notch=0, sym='+', vert=1, whis=1.5, showmeans=True, 
                  meanprops=meanprops, positions=[3,6,7])
plt.setp(bp20['boxes'], color='green')
plt.setp(bp20['whiskers'], color='green')
plt.setp(bp20['fliers'], color='green', marker='+')
plt.setp(bp20['means'], **meanprops3)
plt.setp(bp20['caps'], color='green')
plt.setp(ax2.get_yticklabels(), fontsize=yticklabsz)
ax2.set_ylim(-600, 300)
ax2.set_xlim(0.5, 7+0.5)
ax2.set_xticks([1] + range(3,8))
ax2.set_xticklabels(['Profiles']+mdl,fontsize=xticklabsz)
ax2.annotate('(c)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax2.tick_params(top="off")
ax2.tick_params(right="off")         

# plot 2 pool RC
ax21 = fig.axes[4]
bp21 = ax21.boxplot(plttau, notch=0, sym='+', vert=1, whis=1.5, showmeans=True, 
                  meanprops=meanprops, positions=[1]+range(3,8))
plt.setp(bp21['boxes'][0], color='black')
plt.setp(bp21['whiskers'][:2], color='black')
plt.setp(bp21['fliers'][0], color='red', marker='+')
plt.setp(bp21['boxes'][1:], color='blue')
plt.setp(bp21['whiskers'][2:], color='blue')
plt.setp(bp21['fliers'][1:], color='blue', marker='+')
plt.setp(bp21['means'][1:], **meanprops2)
plt.setp(bp21['caps'][2:], color='blue')
ax21.set_ylabel("Turnover time (yr)\nof RC model\noptimized to ESMs",fontsize=ylabsz)
ax21.set_xlim(0.5, 7+0.5)
ax21.set_ylim(0, 5500)
ax21.set_xticks([1] + range(3,8))
ax21.set_xticklabels(['Profiles']+mdl,fontsize=xticklabsz)
# plot 3pool RC
bp210 = ax21.boxplot(plttau3p, notch=0, sym='+', vert=1, whis=1.5, showmeans=True, 
                  meanprops=meanprops, positions=[3,6,7])
plt.setp(bp210['boxes'], color='green')
plt.setp(bp210['whiskers'], color='green')
plt.setp(bp210['fliers'], color='green', marker='+')
plt.setp(bp210['means'], **meanprops3)
plt.setp(bp210['caps'], color='green')
ax21.set_ylabel("Turnover time (yr)\nof RC model\noptimized to ESMs",fontsize=ylabsz)
ax21.set_xlim(0.5, 7+0.5)
ax21.set_ylim(0, 5500)
ax21.set_xticks([1] + range(3,8))
ax21.set_xticklabels(['Profiles']+mdl,fontsize=xticklabsz)
plt.setp(ax21.get_yticklabels(), fontsize=yticklabsz)
ax21.annotate('(e)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax21.tick_params(top="off")
ax21.tick_params(right="off")         

ax3 = fig.axes[6]
bp3 = ax3.boxplot(plt14Cwscalar, notch=0, sym='+', vert=1, whis=1.5, 
                  showmeans=True, meanprops=meanprops, positions=[1]+range(3,8))
plt.setp(bp3['boxes'][0], color='black')
plt.setp(bp3['whiskers'][:2], color='black')
plt.setp(bp3['fliers'][0], color='red', marker='+')
plt.setp(bp3['boxes'][1:], color='blue')
plt.setp(bp3['whiskers'][2:], color='blue')
plt.setp(bp3['fliers'][1:], color='blue', marker='+')
plt.setp(bp3['means'][1:], **meanprops2)
plt.setp(bp3['caps'][2:], color='blue')
ax3.set_ylabel(r"$\Delta^{14}C$ ("+ u"\u2030) of $^{14}C$\nconstrained model",
               fontsize=ylabsz)
ax3.set_xlim(0.5, 7+0.5)
ax3.set_ylim(-600, 300)
ax3.set_xticks([1] + range(3,8))
ax3.set_xticklabels(['Profiles']+mdl,fontsize=xticklabsz)
plt.setp(ax3.get_yticklabels(), fontsize=yticklabsz)
ax3.annotate('(g)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax3.tick_params(top="off")
ax3.tick_params(right="off")         
plt.tight_layout()          

#%----------------- plot global ----------------
atprofgrids = 0
pltsoc, plt14C, plt14C3p, plt14Cwscalar, mdl = getESM(atprofgrids)
plttau = np.load('globalESMstau.npy')
plttau3p = np.load('globalESMstau_3p.npy')
ax4 = fig.axes[1]
bp4 = ax4.boxplot([pltsoc[i] for i in [6,1,2,3,4,5]], notch=0, sym='', vert=1, whis=1.5,
            showmeans=True, meanprops=meanprops)
plt.setp(bp4['boxes'][0], color='black')
plt.setp(bp4['whiskers'][:2], color='black')
plt.setp(bp4['boxes'][1:], color='blue')
plt.setp(bp4['whiskers'][2:], color='blue')
plt.setp(bp4['means'][1:], **meanprops2)
plt.setp(bp4['caps'][2:], color='blue')
#ax4.set_ylabel('SOC in top 1m $(kgC m^{-2})$',fontsize=12)
ax4.set_ylim(0, 70)
ax4.set_xlim(0.5, 6+0.5)
ax4.set_xticklabels(['HWSD']+mdl,fontsize=xticklabsz)
ax4.set_yticklabels([''], fontsize=yticklabsz)
ax4.annotate('(b)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax4.tick_params(top="off")
ax4.tick_params(right="off")         
ax4.annotate('Global', xy=(0.5,1.02), xycoords='axes fraction',
             fontsize=13, horizontalalignment='center', verticalalignment='bottom')

ax5 = fig.axes[3]
# plot 2 pool RC
bp5 = ax5.boxplot(plt14C[1:], notch=0, sym='', vert=1, whis=1.5, showmeans=True, 
                  positions=range(2,7),meanprops=meanprops2, capprops=capprops)
plt.setp(bp5['boxes'], color='blue')
plt.setp(bp5['whiskers'], color='blue')
ax5.set_xlim(0.5, 6+0.5)
ax5.set_xticks(range(2,7))
ax5.set_ylim(-600, 300)
ax5.set_xticklabels(mdl,fontsize=xticklabsz)
# plot 3 pool RC
bp50 = ax5.boxplot(plt14C3p, notch=0, sym='', vert=1, whis=1.5, showmeans=True, 
                  positions=[2,5,6],meanprops=meanprops3, capprops=capprops)
plt.setp(bp50['boxes'], color='green')
plt.setp(bp50['whiskers'], color='green')
#ax5.set_ylabel(r"$\Delta14C$ ("+ u"\u2030) of fitted model",fontsize=12)
ax5.set_xlim(0.5, 6+0.5)
ax5.set_xticks(range(2,7))
ax5.set_ylim(-600, 300)
ax5.set_xticklabels(mdl,fontsize=xticklabsz)
ax5.annotate('(d)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax5.set_yticklabels([''], fontsize=yticklabsz)
ax5.tick_params(top="off")
ax5.tick_params(right="off")         

ax51 = fig.axes[5]
# 2pool RC
bp51 = ax51.boxplot(plttau[1:], notch=0, sym='', vert=1, whis=1.5, showmeans=True, 
                  positions=range(2,7),meanprops=meanprops2, capprops=capprops)
plt.setp(bp51['boxes'], color='blue')
plt.setp(bp51['whiskers'], color='blue')
ax51.set_xlim(0.5, 6+0.5)
ax51.set_xticks(range(2,7))
ax51.set_ylim(0, 5500)
ax51.set_xticklabels(mdl,fontsize=xticklabsz)
# plot 3 pool RC
bp510 = ax51.boxplot(plttau3p, notch=0, sym='', vert=1, whis=1.5, showmeans=True, 
                  positions=[2,5,6],meanprops=meanprops3, capprops=capprops)
plt.setp(bp510['boxes'], color='green')
plt.setp(bp510['whiskers'], color='green')
#ax51.set_yscale("log", nonposy='clip')
ax51.set_xlim(0.5, 6+0.5)
ax51.set_xticks(range(2,7))
ax51.set_ylim(0, 5500)
ax51.set_xticklabels(mdl,fontsize=xticklabsz)
ax51.annotate('(f)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax51.set_yticklabels([''], fontsize=yticklabsz)
ax51.tick_params(top="off")
ax51.tick_params(right="off")         

ax6 = fig.axes[7]
bp6 = ax6.boxplot(plt14Cwscalar[1:], notch=0, sym='', vert=1, whis=1.5, positions=range(2,7),
                   showmeans=True, meanprops=meanprops2, capprops=capprops)
plt.setp(bp6['boxes'], color='blue')
plt.setp(bp6['whiskers'], color='blue')
#ax6.set_ylabel(r"$\Delta14C$ ("+ u"\u2030) of\n$^{14}C$ constrained model",fontsize=12)
ax6.set_xlim(0.5, 6+0.5)
ax6.set_ylim(-600, 300)
ax6.set_xticklabels(mdl,fontsize=xticklabsz)
ax6.annotate('(h)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax6.set_yticklabels([''], fontsize=yticklabsz)
ax6.tick_params(top="off")
ax6.tick_params(right="off")   
plt.rc("xtick", direction="out")
plt.rc("ytick", direction="out")      
plt.tight_layout()   
#%% plot (add) turnover time of site profiles and global in one figure, 4*2
# best case
import C14tools
#----------- at sites --------------
atprofgrids = 1
pltsoc, plt14C, plt14C3p, plt14Cwscalar, mdl = getESM(atprofgrids)
profsmpyr = np.loadtxt('tot48prof.txt', delimiter=',')[:,4]
plttau = []
for n,i in enumerate(plt14C):
    if n == 0:
        plttau.append(C14tools.cal_tau(i, profsmpyr, 1, 1))
    else:
        plttau.append(C14tools.cal_tau(i, np.ones(len(i))*1990, 1, 1))
plttau3p = []
for n,i in enumerate(plt14C3p):
    plttau3p.append(C14tools.cal_tau(i, np.ones(len(i))*1990, 1, 1))
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8,8))
ylabsz = 10; xticklabsz = 10; yticklabsz = 9; annosz = 11
textpos = (.95, .85) # upper right
textpos = (.1, 1.02) # upper left outside
meanprops = dict(marker='*', markeredgecolor='black', markersize=10,
                 markerfacecolor='firebrick')
meanprops2 = dict(marker='*', markeredgecolor='black', markersize=10,
                  markerfacecolor='dodgerblue')
meanprops3 = dict(marker='*', markeredgecolor='black', markersize=10,
                  markerfacecolor='dodgerblue') # for 3pool results
capprops = dict(linestyle='-', linewidth=1, color='blue')
ax1 = fig.axes[0]
bp1 = ax1.boxplot([pltsoc[i] for i in [0,6,1,2,3,4,5]], notch=0, sym='+', vert=1, whis=1.5,
            showmeans=True, meanprops=meanprops)
plt.setp(bp1['boxes'][:2], color='black')
plt.setp(bp1['whiskers'][:4], color='black')
plt.setp(bp1['fliers'][:2], color='red', marker='+')
plt.setp(bp1['boxes'][2:], color='blue')
plt.setp(bp1['whiskers'][4:], color='blue')
plt.setp(bp1['fliers'][2:], color='blue', marker='+')
plt.setp(bp1['means'][2:], **meanprops2)
plt.setp(bp1['caps'][4:], color='blue')
ax1.set_ylabel('SOC $(kgC$ $m^{-2})$',fontsize=ylabsz)
ax1.set_xlim(0.5, 7+0.5)
ax1.set_xticks(range(1,8))
ax1.set_xticklabels(['Profiles']+['HWSD']+mdl,fontsize=xticklabsz)
plt.setp(ax1.get_yticklabels(), fontsize=yticklabsz)
ax1.annotate('(a)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax1.tick_params(top="off")
ax1.tick_params(right="off")             
ax1.annotate('Sites', xy=(0.5,1.02), xycoords='axes fraction',
             fontsize=13, horizontalalignment='center', verticalalignment='bottom')
# plot 2pool RC
ax2 = fig.axes[2]
bp2 = ax2.boxplot([plt14C[i] for i in [0,2,3]], notch=0, sym='+', vert=1, whis=1.5, showmeans=True, 
                  meanprops=meanprops, positions=[1,4,5])
plt.setp(bp2['boxes'][0], color='black')
plt.setp(bp2['whiskers'][:2], color='black')
plt.setp(bp2['fliers'][0], color='red', marker='+')
plt.setp(bp2['boxes'][1:], color='blue')
plt.setp(bp2['whiskers'][2:], color='blue')
plt.setp(bp2['fliers'][1:], color='blue', marker='+')
plt.setp(bp2['means'][1:], **meanprops2)
plt.setp(bp2['caps'][2:], color='blue')
ax2.set_ylabel(r"$\Delta^{14}C$ ("+ u"\u2030) of RC model\noptimized to ESMs",fontsize=ylabsz)
ax2.set_ylim(-600, 300)
ax2.set_xlim(0.5, 7+0.5)
ax2.set_xticks([1] + range(3,8))
ax2.set_xticklabels(['Profiles']+mdl,fontsize=xticklabsz)
# plot 3pool RC
bp20 = ax2.boxplot(plt14C3p, notch=0, sym='+', vert=1, whis=1.5, showmeans=True, 
                  meanprops=meanprops, positions=[3,6,7])
plt.setp(bp20['boxes'], color='blue')
plt.setp(bp20['whiskers'], color='blue')
plt.setp(bp20['fliers'], color='blue', marker='+')
plt.setp(bp20['means'], **meanprops3)
plt.setp(bp20['caps'], color='blue')
plt.setp(ax2.get_yticklabels(), fontsize=yticklabsz)
ax2.set_ylim(-600, 300)
ax2.set_xlim(0.5, 7+0.5)
ax2.set_xticks([1] + range(3,8))
ax2.set_xticklabels(['Profiles']+mdl,fontsize=xticklabsz)
ax2.annotate('(c)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax2.tick_params(top="off")
ax2.tick_params(right="off")         

# plot 2 pool RC
ax21 = fig.axes[4]
bp21 = ax21.boxplot([plttau[i] for i in [0,2,3]], notch=0, sym='+', vert=1, whis=1.5, showmeans=True, 
                  meanprops=meanprops, positions=[1,4,5])
plt.setp(bp21['boxes'][0], color='black')
plt.setp(bp21['whiskers'][:2], color='black')
plt.setp(bp21['fliers'][0], color='red', marker='+')
plt.setp(bp21['boxes'][1:], color='blue')
plt.setp(bp21['whiskers'][2:], color='blue')
plt.setp(bp21['fliers'][1:], color='blue', marker='+')
plt.setp(bp21['means'][1:], **meanprops2)
plt.setp(bp21['caps'][2:], color='blue')
ax21.set_ylabel("Turnover time (yr)\nof RC model\noptimized to ESMs",fontsize=ylabsz)
ax21.set_xlim(0.5, 7+0.5)
ax21.set_ylim(0, 5500)
ax21.set_xticks([1] + range(3,8))
ax21.set_xticklabels(['Profiles']+mdl,fontsize=xticklabsz)
# plot 3pool RC
bp210 = ax21.boxplot(plttau3p, notch=0, sym='+', vert=1, whis=1.5, showmeans=True, 
                  meanprops=meanprops, positions=[3,6,7])
plt.setp(bp210['boxes'], color='blue')
plt.setp(bp210['whiskers'], color='blue')
plt.setp(bp210['fliers'], color='blue', marker='+')
plt.setp(bp210['means'], **meanprops3)
plt.setp(bp210['caps'], color='blue')
ax21.set_ylabel("Turnover time (yr)\nof RC model\noptimized to ESMs",fontsize=ylabsz)
ax21.set_xlim(0.5, 7+0.5)
ax21.set_ylim(0, 5500)
ax21.set_xticks([1] + range(3,8))
ax21.set_xticklabels(['Profiles']+mdl,fontsize=xticklabsz)
plt.setp(ax21.get_yticklabels(), fontsize=yticklabsz)
ax21.annotate('(e)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax21.tick_params(top="off")
ax21.tick_params(right="off")         

ax3 = fig.axes[6]
bp3 = ax3.boxplot(plt14Cwscalar, notch=0, sym='+', vert=1, whis=1.5, 
                  showmeans=True, meanprops=meanprops, positions=[1]+range(3,8))
plt.setp(bp3['boxes'][0], color='black')
plt.setp(bp3['whiskers'][:2], color='black')
plt.setp(bp3['fliers'][0], color='red', marker='+')
plt.setp(bp3['boxes'][1:], color='blue')
plt.setp(bp3['whiskers'][2:], color='blue')
plt.setp(bp3['fliers'][1:], color='blue', marker='+')
plt.setp(bp3['means'][1:], **meanprops2)
plt.setp(bp3['caps'][2:], color='blue')
ax3.set_ylabel(r"$\Delta^{14}C$ ("+ u"\u2030) of $^{14}C$\nconstrained model",
               fontsize=ylabsz)
ax3.set_xlim(0.5, 7+0.5)
ax3.set_ylim(-600, 300)
ax3.set_xticks([1] + range(3,8))
ax3.set_xticklabels(['Profiles']+mdl,fontsize=xticklabsz)
plt.setp(ax3.get_yticklabels(), fontsize=yticklabsz)
ax3.annotate('(g)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax3.tick_params(top="off")
ax3.tick_params(right="off")         
plt.tight_layout()          

#%----------------- plot global ----------------
atprofgrids = 0
pltsoc, plt14C, plt14C3p, plt14Cwscalar, mdl = getESM(atprofgrids)
plttau = np.load('globalESMstau.npy')
plttau3p = np.load('globalESMstau_3p.npy')
ax4 = fig.axes[1]
bp4 = ax4.boxplot([pltsoc[i] for i in [6,1,2,3,4,5]], notch=0, sym='', vert=1, whis=1.5,
            showmeans=True, meanprops=meanprops)
plt.setp(bp4['boxes'][0], color='black')
plt.setp(bp4['whiskers'][:2], color='black')
plt.setp(bp4['boxes'][1:], color='blue')
plt.setp(bp4['whiskers'][2:], color='blue')
plt.setp(bp4['means'][1:], **meanprops2)
plt.setp(bp4['caps'][2:], color='blue')
#ax4.set_ylabel('SOC in top 1m $(kgC m^{-2})$',fontsize=12)
ax4.set_ylim(0, 70)
ax4.set_xlim(0.5, 6+0.5)
ax4.set_xticklabels(['HWSD']+mdl,fontsize=xticklabsz)
ax4.set_yticklabels([''], fontsize=yticklabsz)
ax4.annotate('(b)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax4.tick_params(top="off")
ax4.tick_params(right="off")         
ax4.annotate('Global', xy=(0.5,1.02), xycoords='axes fraction',
             fontsize=13, horizontalalignment='center', verticalalignment='bottom')

ax5 = fig.axes[3]

# plot 3 pool RC
bp50 = ax5.boxplot(plt14C3p, notch=0, sym='', vert=1, whis=1.5, showmeans=True, 
                  positions=[2,5,6],meanprops=meanprops2, capprops=capprops)
plt.setp(bp50['boxes'], color='blue')
plt.setp(bp50['whiskers'], color='blue')
# plot 2 pool RC
bp5 = ax5.boxplot([plt14C[i] for i in [2,3]], widths=0.5,notch=0, sym='', vert=1, whis=1.5, showmeans=True, 
                  positions=[3,4],meanprops=meanprops2, capprops=capprops)
plt.setp(bp5['boxes'], color='blue')
plt.setp(bp5['whiskers'], color='blue')
ax5.set_xlim(0.5, 6+0.5)
ax5.set_xticks(range(2,7))
ax5.set_ylim(-600, 300)
ax5.set_xticklabels(mdl,fontsize=xticklabsz)

ax5.set_xlim(0.5, 6+0.5)
ax5.set_xticks(range(2,7))
ax5.set_ylim(-600, 300)
ax5.set_xticklabels(mdl,fontsize=xticklabsz)
ax5.annotate('(d)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax5.set_yticklabels([''], fontsize=yticklabsz)
ax5.tick_params(top="off")
ax5.tick_params(right="off")         

ax51 = fig.axes[5]

# plot 3 pool RC
bp510 = ax51.boxplot(plttau3p, notch=0, sym='', vert=1, whis=1.5, showmeans=True, 
                  positions=[2,5,6],meanprops=meanprops2, capprops=capprops)
plt.setp(bp510['boxes'], color='blue')
plt.setp(bp510['whiskers'], color='blue')
ax51.set_xlim(0.5, 6+0.5)
ax51.set_xticks(range(2,7))
ax51.set_ylim(0, 5500)
# 2pool RC
bp51 = ax51.boxplot([plttau[i] for i in [2,3]], widths=0.5,notch=0, sym='', vert=1, whis=1.5, showmeans=True, 
                  positions=[3,4],meanprops=meanprops2, capprops=capprops)
plt.setp(bp51['boxes'], color='blue')
plt.setp(bp51['whiskers'], color='blue')
ax51.set_xlim(0.5, 6+0.5)
ax51.set_xticks(range(2,7))
ax51.set_ylim(0, 5500)
ax51.set_xticklabels(mdl,fontsize=xticklabsz)

ax51.set_xticklabels(mdl,fontsize=xticklabsz)
ax51.annotate('(f)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax51.set_yticklabels([''], fontsize=yticklabsz)
ax51.tick_params(top="off")
ax51.tick_params(right="off")         

ax6 = fig.axes[7]
bp6 = ax6.boxplot(plt14Cwscalar[1:], notch=0, sym='', vert=1, whis=1.5, positions=range(2,7),
                   showmeans=True, meanprops=meanprops2, capprops=capprops)
plt.setp(bp6['boxes'], color='blue')
plt.setp(bp6['whiskers'], color='blue')
#ax6.set_ylabel(r"$\Delta14C$ ("+ u"\u2030) of\n$^{14}C$ constrained model",fontsize=12)
ax6.set_xlim(0.5, 6+0.5)
ax6.set_ylim(-600, 300)
ax6.set_xticklabels(mdl,fontsize=xticklabsz)
ax6.annotate('(h)', xy=textpos, xycoords='axes fraction',
             fontsize=annosz, horizontalalignment='right', verticalalignment='bottom')
ax6.set_yticklabels([''], fontsize=yticklabsz)
ax6.tick_params(top="off")
ax6.tick_params(right="off")   
plt.rc("xtick", direction="out")
plt.rc("ytick", direction="out")      
plt.tight_layout()  