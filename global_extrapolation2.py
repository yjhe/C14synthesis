# -*- coding: utf-8 -*-
"""
Extrapolation file: use ML models to predict global D14C

Created on Thu Jun 18 02:05:17 2015

@author: happysk8er
"""
#%%
import pandas as pd
import numpy as np
import D14Cpreprocess as prep
import sklearn as sk
from sklearn import svm
from sklearn import preprocessing
import mystats as mysm
import C14tools
import scipy.io
import matplotlib.pyplot as plt
import myplot as myplt
from netCDF4 import Dataset
import mynetCDF as mync
#% extrapolation. SVM regression
def d14C_svr(useorilyr=1,cutdep=9999,ddelta=0,vartau=0,clayon=0,nppon=0,Xextra=None):
    ''' construct svr learner. 
    param: 
        useorilyr : whether use original layer or incremented layer data
        cutdep    : indicate the depth above which to used for training        
        indicate what is y (default: d14C; or ddelta, or vartau)
        indicate what additional input to use (clay or npp)
        Xextra    : X used for extrapolation. if is None, construct scalar from obs only; 
                    otherwise, construct scalar using obs + Xextra
    return:
        svrtrained : trained svr model, can use to predict
        scaler     : scalar, need to be used to scale Xextra
    '''
    filename = 'Non_peat_data_synthesis.csv'
    data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1])  
    profid = data[data['Start_of_Profile']==1].index # index of profile start
    if useorilyr == 1:    
        d14C = prep.getvarxls(data,'D14C_BulkLayer', profid, ':')
        sampleyr = prep.getvarxls(data, 'SampleYear', profid, ':')
        dd14C = prep.getDDelta14C(sampleyr, d14C)
        #tau, cost = C14tools.cal_tau(d14C, sampleyr, 1, 1)
        #data['tau'] = pd.Series(tau[:,0], index=data.index)
        mat = prep.getvarxls(data,'MAT', profid, ':')
        mapp = prep.getvarxls(data,'MAP', profid, ':')
        layerbot = prep.getvarxls(data, 'Layer_bottom_norm', profid, ':')
        vegid = prep.getvarxls(data, 'VegTypeCode_Local', profid, ':')
        vegiduniq = np.unique(vegid[~np.isnan(vegid)])
        soilorder = prep.getvarxls(data, 'LEN Order', profid, ':')
        soilorder = np.array([str(i) for i in soilorder])
        soilorderuniq = np.unique(soilorder[soilorder != 'nan'])
        lon = prep.getvarxls(data,'Lon',profid,':')
        lat = prep.getvarxls(data,'Lat',profid,':')
        clayon = 0
        nppon = 0    
        if clayon == 1:
            sclayfn = 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_CLAY.nc4'
            tclayfn = 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_CLAY.nc4'
            scecclayfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_CEC_CLAY.nc4'
            tcecclayfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_CEC_CLAY.nc4'
            tbulkdenfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\T_BULK_DEN.nc4'
            sbulkdenfn= 'AncillaryData\\HWSD\\Regridded_ORNLDAAC\\S_BULK_DEN.nc4'
            nppfn = 'AncillaryData\\NPP\\2000_2012meannpp_gCm2yr.nc'
            sclay = prep.getHWSD(sclayfn, lon, lat) # % weight
            tclay = prep.getHWSD(tclayfn, lon, lat)  # % weight
            scecclay = prep.getHWSD(scecclayfn, lon, lat)  # cmol/kg clay
            tcecclay = prep.getHWSD(tcecclayfn, lon, lat)  # cmol/kg clay
            sbulkden = prep.getHWSD(sbulkdenfn, lon, lat)  # g/cm3
            tbulkden = prep.getHWSD(tbulkdenfn, lon, lat)  # g/cm3
            clay = tbulkden*tclay*0.3 + sbulkden*sclay*0.7
            cecclay =  tbulkden*tclay*tcecclay*0.3 + sbulkden*sclay*scecclay*0.7
        if nppon == 1:
            npp = prep.getnpp(nppfn, lon, lat) # gC/m2/yr
        dummyveg = (vegid[:, None] == vegiduniq).astype(float)  # dummy[:,1:]
        dummysord = (soilorder[:, None] == soilorderuniq).astype(float)
        
        # construct X and y
        x = np.c_[layerbot, mat, mapp]
        if ddelta == 1:
            y = dd14C
        if vartau == 1:
            y = tau[:,0]
        else:
            y = d14C
    else:
        colname = ['Layer_top_norm','Layer_bottom_norm','D14C_BulkLayer','VegTypeCode_Local',
                   'MAT','MAP','LEN Order','SampleYear']
        newdf = data[colname]
        incre_data = prep.prep_increment(newdf, colname)
        dd14C = prep.getDDelta14C(incre_data['SampleYear'].values.astype(float),
                                  incre_data['D14C_BulkLayer'].values.astype(float))    
        vegid = incre_data['VegTypeCode_Local'].values.astype(int)
        vegiduniq = np.unique(vegid[~np.isnan(vegid)])
        soilorder = incre_data['LEN Order'].values.astype(str)
        soilorderuniq = np.unique(soilorder[soilorder != 'nan'])
        
        # construct X and y
        dummyveg = (vegid[:, None] == vegiduniq).astype(float)  # dummy[:,1:]
        dummysord = (soilorder[:, None] == soilorderuniq).astype(float)
        x = incre_data[['cont_depth','MAT','MAP']].values.astype(float)
        if ddelta == 1:
            y = dd14C
        else:
            y = incre_data['D14C_BulkLayer'].values.astype(float)
        if vartau == 1:
            y = tau[:,0]
    
    notNaNs = ~np.any(np.isnan(x),1) & ~np.isnan(y) 
    X = x[notNaNs,:]
    y = y[notNaNs]
    X = np.c_[X, dummyveg[notNaNs,1:]]
    idx = X[:,0] < cutdep
    X = X[idx,:]; y = y[idx]
    if Xextra is None:
        scalar = sk.preprocessing.StandardScaler().fit(X)
        print 'in d14Csvr, scalar is None, constructed scalar.mean is ', scalar.mean_
        X_scaled = scalar.transform(X)
    else:
        scalar = sk.preprocessing.StandardScaler().fit(np.r_[X,Xextra])
        print 'in d14Csvr, Xextra is given, the constructed scalar.mean for whole dataset is ', scalar.mean_
        X_scaled = scalar.transform(X)
    # train SVM and predict
    #svr = sk.svm.SVR(C=50, kernel='poly', degree=3, coef0=2)
    C = 1e4;
    svr = sk.svm.SVR(C=C, kernel='rbf', gamma=0.4)
    svrtrained = svr.fit(X_scaled, y)
    yhat = svr.fit(X_scaled, y).predict(X_scaled)
    print "R2 is: %.3f, R2adj is: %.3f" %(mysm.cal_R2(y,yhat),0)
    print "RMSE is: %.3f" %(mysm.cal_RMSE(y,yhat))
    return svrtrained, scalar
    
def changelccodeto14Csyn(lc):
    '''change the code from original landcover data to the 8 categories
    used in 14C synthesis. lc contains 1-12, 14.
    parameters: lc is the readed in .mat matrix
    output: np.array same with lc dimension, but with the 8 14C categories
    '''
    idx = np.zeros((17,lc.shape[0],lc.shape[1]))
    for i in range(17):
        idx[i,:,:] = lc==i
    lonts, lats = construct_lonlat()
    mapping = {1:[1,2,3,4,5,8],
               2:[1,2,3,4,5], 
               3:[1,2,3,4,5],
               4:[10],
               5:[12,14],
               6:[6,7],
               7:[11],
               8:[9],
               9:[6,7],
              10:[16]}

    for i in mapping.keys():
        tmpidx = np.zeros((lc.shape))
        for j in mapping[i]:
            tmpidx = tmpidx + idx[j,:,:]
        if i == 1: # boreal forest
            tmpidx[lats<50,:] = 0
        if i == 2: # teperate forest
            tmpidx[(lats>=50) | ((lats<23) & (lats>-23)) | (lats<=-50), :] = 0
        if i == 3: # tropical forest
            tmpidx[(lats>=23) | (lats<=-23),:] = 0     
        lc[tmpidx>=1] = i
        if i == 9: # high latitude tundra, assigned from shrubland
            lc[(tmpidx>0) & np.tile(lats<60, (720,1)).T] = 6  
    return lc

def svrpred(svr, scalar, depth, x, dummy):
    ''' use trained svr to predict d14C at a particular depth
    x must not be the scaled, will construct final x appending depth and 
    dummy variable to it in this func, then scale. no NANs are allowed.
    parameters:
        svr: the trained svr learner
        scalar: if exist(scaler=1), use and predict, return yhat; 
                otherwise(scaler=0), return X
        x: mat, mapp or other continuous var, total m var. (N*m)
        depth: scalar 
        dummy: 1d vector indicating veg type or soil order. (N*1)
    output:
        yhat: predicted y, depends on what svr is trained on. (N*1)
        X   : Xextra, original, not scaled
    '''    
    N = x.shape[0]
    dummyuniq = np.unique(dummy)
    dummyvar = (dummy[:,None] == dummyuniq).astype(float)
    X = np.c_[np.tile(depth,(N,1)), x, dummyvar[:,1:]]
    if scalar != 0:
        X_scaled = scalar.transform(X)
        yhat = svr.predict(X_scaled)
        return yhat
    else:
        scalarr = sk.preprocessing.StandardScaler().fit(X)
        print 'in svnpred, scalar.mean is ', scalarr.mean_
        return X
        
def construct_lonlat():
    latstep = 0.5; lonstep = 0.5;
    lonmax = 180.0 - lonstep/2.0;
    lonmin = -180.0 + lonstep/2.0;
    latmax = 90.0 - latstep/2.0;
    latmin = -90.0 + latstep/2.0;
    lats = np.arange(latmax,latmin-0.1*latstep,-latstep)
    lons = np.arange(lonmin,lonmax+0.1*lonstep,lonstep)
    return lons, lats

def cal_binedtauC(tau, C, bins):
    '''
    calculate the binned tau and total C, used for plotting histogram
    parameters:
        tau  : turnover time, 2-D
        C    : carbon content (kgC/m2)
        bins : number of bins you want to group tau
    return: 
        xtau : 1-D mean tau for each bin [bins,]
        yC   : 1-D total C for each bin [bins,]
    '''
    areaa = mysm.cal_earthgridarea(0.5)
    hist, bin_eg = np.histogram(tau[~np.isnan(tau)], bins)
#    bin_eg = np.arange(0, np.log(30000), np.log(30000)/(bins+1))  
#    bin_eg = np.exp(bin_eg)
    xtau = np.zeros(bins)
    yC   = np.zeros(bins)
    for n in range(bins):
        idx = np.where((tau > bin_eg[n]) & (tau < bin_eg[n+1]))
        xtau[n] = np.nansum(tau[idx] * areaa[idx])/np.nansum(areaa[idx])
        yC[n]   = np.nansum(C[idx] * areaa[idx])
    return xtau, yC, bin_eg

# preprocess input data for extrapolation
def prepdata(plotorilc=0, plotnewlc=0):
    mat = np.load('..\\AncillaryData\\CRUdata\\cruannualtmp.npy')
    mat = np.nanmean(mat[-30:-20,:,:],axis=0)/10.   # top-down 90S-90N
    maprep = np.load('..\\AncillaryData\\CRUdata\\cruannualpre.npy')
    maprep = np.nanmean(maprep[-30:-20,:,:],axis=0)*12./10.-300.
    lc = scipy.io.loadmat('..\\AncillaryData\\landcoverdata\\lc2011_05x05.mat') #1-12, 14, 16
    lc = lc['lc05x05']   # top-down 90N-90S
    if plotorilc == 1:
        lons,lats = construct_lonlat()
        vegtype = ['EG needleleaf','EG broadleaf','Deci Needleleaf','Deci broadleaf',
                   'Mixed forest','closed shrublands','Open shrublands','Woody savannas',
                   'savannas','grasslands','permanent wetlands','croplands',
                   'Cropland/natural veg mosaic']
        myplt.geoshow(np.flipud(lc),lons,np.flipud(lats),cbar='h',ticks=range(16),cbarticklab=vegtype,
                      rotation=45)
    lc = changelccodeto14Csyn(lc)
    lc = np.flipud(lc) 
    # ravel matrix to column vector
    mat_1d = np.ravel(mat)
    maprep_1d = np.ravel(maprep)
    lc_1d = np.ravel(lc)
    
    # plot lc
    if plotnewlc == 1:
        lons,lats = construct_lonlat()
        vegtype = ['boreal forest','temperate forest','tropical forest','grassland',
                   'cropland','shrublands','peatland','savannas','tundra','desert']
        myplt.geoshow(lc,lons,np.flipud(lats),cbar='h',ticks=np.arange(0,11,1),cbarticklab=vegtype,
                      rotation=45, levels=10, extend='neither')
    return mat_1d, maprep_1d, lc_1d
    
def getbiomed14Cprof(depth, mdl, siteinfo=None):
    '''
    parameters:
        mdl      : 'svr', 'gbrt'
        siteinfo : lon, lat, lc at sites, []
    return:
        biome_14C_mean : [len(depth),nbiomes]
        biome_14C_std  : [len(depth),nbiomes] 
    '''
    mat_1d, maprep_1d, lc_1d = prepdata()
    x = np.c_[mat_1d, maprep_1d]
    lc = np.reshape(lc_1d,(360,720)); lc = np.flipud(lc) # 90N - 90S
    notNaNs = ~np.any(np.isnan(x),1) & ~np.isnan(lc_1d) 
    #print 'total grid notNaNs: ', sum(notNaNs)    
    biomes    = np.unique(lc_1d)[:10]  
    biome_14C_mean = np.zeros((len(depth),len(biomes)))
    biome_14C_std = np.zeros((len(depth),len(biomes)))
    if siteinfo is not None:
        mask = np.zeros((len(biomes), lc.shape[0], lc.shape[1]))
        lons, lats = construct_lonlat()
        for nb, bio in enumerate(biomes):
            dum = np.array(siteinfo)
            cursiteinfo = list(dum[dum[:,2]==bio,:])
            for lon, lat, sitelc in cursiteinfo:  
                if ~np.isnan(lon) and ~np.isnan(lat):
                    ii,jj = np.where(lats<lat)[0][0], np.where(lons>lon)[0][0]
                    if sitelc == lc[ii,jj]:
                        print 'sitelc is consistent with lc map...'
                    else:
                        print 'Not consistent lc ...'
                        print '     site lon, lat is %.2f, %.2f, lc lon,lat is %.2f, %.2f'%(lon, lat, lons[jj], lats[ii])
                        print '     sitelc is %.1f, lc is %.1f, ...'%(sitelc, lc[ii,jj])
                    mask[nb,ii,jj] = 1    
    for nd, d in enumerate(depth):
        print 'depth is ', d
        yhat = np.empty((x.shape[0])); yhat[:] = np.NAN        
        if mdl == 'svr':
            svr, scalar = d14C_svr()
            ypred = svrpred(svr, scalar, d, x[notNaNs,:], lc_1d[notNaNs])
        if mdl == 'gbrt':
            print "Not implemented yet"
        yhat[notNaNs] = ypred
        yhat = np.reshape(yhat,(360,720))
        yhat = np.flipud(yhat) # now 90N-90S
       
        # get biome profile
        if siteinfo is not None:
            for nb, bio in enumerate(biomes):
                tmp = yhat[mask[nb,:,:]==1]
                biome_14C_mean[nd, nb] = np.nanmean(tmp)
                biome_14C_std[nd, nb]  = np.nanstd(tmp)
        else:
            for nb, bio in enumerate(biomes):
                tmp = yhat[lc==bio]
                biome_14C_mean[nd, nb] = np.nanmean(tmp)
                biome_14C_std[nd, nb]  = np.nanstd(tmp)
    return biome_14C_mean, biome_14C_std
        
#%% train svr and predict (extrapolate). calculate tau, and save
# use only observations to contruct scaler 
mat_1d, maprep_1d, lc_1d = prepdata()
svr, scalar = d14C_svr(useorilyr=1)
depth = 100.
x = np.c_[mat_1d, maprep_1d]
dummy = lc_1d
notNaNs = ~np.any(np.isnan(x),1) & ~np.isnan(dummy) 
#print 'total grid notNaNs: ', sum(notNaNs)
yhat = np.empty((x.shape[0])); yhat[:] = np.NAN
ypred = svrpred(svr, scalar, depth, x[notNaNs,:], lc_1d[notNaNs])
yhat[notNaNs] = ypred
yhat = np.reshape(yhat,(360,720))

# need to use the whole dataset to construct scaler.
depth = 100.
x = np.c_[mat_1d, maprep_1d]
dummy = lc_1d
notNaNs = ~np.any(np.isnan(x),1) & ~np.isnan(dummy) 
Xextra = svrpred(1, 0, depth, x[notNaNs,:], lc_1d[notNaNs])
svr, scalar = d14C_svr(Xextra=Xextra)
yhat = np.empty((x.shape[0])); yhat[:] = np.NAN
ypred = svrpred(svr, scalar, depth, x[notNaNs,:], lc_1d[notNaNs])
yhat[notNaNs] = ypred
yhat = np.reshape(yhat,(360,720))


# plot
plt.figure()
lons,lats = construct_lonlat()
im = myplt.geoshow(yhat,lons,np.flipud(lats))
plt.title('depth ' +  str(depth) + 'cm')

plt.figure()
myplt.geoshow(mat,lons,np.flipud(lats))
myplt.geoshow(maprep,lons,np.flipud(lats))

#%% read in HWSD and resample to 0.5 X 0.5, save data to npz. 
from scipy.interpolate import griddata
from scipy.misc import imresize
pathh = 'C:\\download\work\\!manuscripts\\C14_synthesis\\AncillaryData\\HWSD\Regridded_ORNLDAAC\\'
orifn = 'AWT_S_SOC.nc4'
orifn = 'AWT_T_SOC.nc4'
ncfid = Dataset(pathh + orifn, 'r')
nc_attrs, nc_dims, nc_vars = mync.ncdump(ncfid)
lats = ncfid.variables['lat'][:]
lons = ncfid.variables['lon'][:]
var = ncfid.variables[nc_vars[-1]][:]

awt_s_0505 = imresize(var, (360, 720))
lons,lats = construct_lonlat()
awt_s_0505[awt_s_0505 == 0] = np.nan
plt.figure()
myplt.geoshow(awt_s_0505,lons,np.flipud(lats))
outlats = np.arange()

awt_t_0505 = mysm.rebin(var,[360, 720])
myplt.geoshow(awt_t_0505,lons,np.flipud(lats))
awt_t_0505.fill_value = np.nan
awt_s_0505.fill_value = np.nan
t = awt_t_0505.filled()
s = awt_s_0505.filled()
np.savez('awt_soc_0505',awt_s_0505=s,awt_t_0505=t)
#%%  plot histogram of turnover time and its associated C stock
# get turnover time
import C14tools
d14c50cm = yhat
tau30cm = C14tools.cal_tau(ypred,np.repeat(2000,ypred.shape[0]),1,1)
tau = np.empty(yhat.shape[0]*yhat.shape[1]); tau[:] = np.NAN
tau[notNaNs] = tau30cm
tau = np.reshape(tau, (360, 720))
im = myplt.geoshow(tau,lons,np.flipud(lats))
tau30cm = tau
np.save('extra_tau_30cm',tau30cm)

# read in HWSD
pathh = '.\\AncillaryData\\HWSD\\Regridded_ORNLDAAC\\'
hwsd = np.load(pathh + 'awt_soc_0505.npz')

tau = np.load('extra_tau_90cm.npy')
C     = hwsd['awt_s_0505']
bins  = 40
x, y, bin_eg  = cal_binedtauC(tau, C, bins)
#plt.hist(tau[~np.isnan(tau)],bins=bins,normed=True)

# plot barplot
fig, ax = plt.subplots(figsize=(8,6))
ax.bar(bin_eg[1:], y/1e12, width=np.diff(bin_eg)[0])
ax.set_yscale('log')
ax.set_ylabel('SOC in 30-100cm (TgC)')
ax.set_xlabel('tau (yr)')
#%% get biome mean profile from global extrapolation
# get biome from global extrapolation
depth = [0, 30, 70, 100, 150, 200]
meann, std = getbiomed14Cprof(depth, 'svr')
vegtype = ['boreal forest','temperate forest','tropical forest','grassland',
           'cropland','shrublands','peatland','savannas','tundra','desert']

# get biome from global extrapolation at obs sites
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
pid = data[data['Start_of_Profile']==1].index # index of profile start
d14C = prep.getvarxls(data,'D14C_BulkLayer', pid, ':')
sitelon = prep.getvarxls(data,'Lon', pid, 0)
sitelat = prep.getvarxls(data,'Lat', pid, 0)
sitelc = prep.getvarxls(data,'VegTypeCode_Local', pid, 0)
siteinfo = zip(sitelon, sitelat, sitelc)
meann, std = getbiomed14Cprof(depth, 'svr', siteinfo=siteinfo)
vegtype = ['boreal forest','temperate forest','tropical forest','grassland',
           'cropland','shrublands','peatland','savannas','tundra','desert']

# plot
nrows, ncols = 2, 5
fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(15,8),sharey=True)
for n,row in enumerate(axes):
    for m,col in enumerate(row):
        idx = ncols*n+m
#        if idx == 8:
#            break         
        col.errorbar(meann[:,idx], depth, xerr=std[:,idx], fmt='-o')    
        col.locator_params(nbins=4)
        col.set_ylabel('depth(cm)')
        col.set_xlabel(r'$\Delta^{14}C$')
        col.set_title(vegtype[idx])
        col.set_xlim([-1000, 500])
        col.set_ylim([-10, 250])
col.invert_yaxis()
plt.tight_layout()

#%% use increment approach to process original csv data, plot biome averaged profile
filename = 'Non_peat_data_synthesis.csv'
data = pd.read_csv(filename,encoding='iso-8859-1',index_col='ProfileID', skiprows=[1]) 
pid = data[data['Start_of_Profile']==1].index # index of profile start
colname = ['Layer_top_norm','Layer_bottom_norm','D14C_BulkLayer','VegTypeCode_Local']
newdf = data[colname]
incre_data = prep.prep_increment(newdf, colname)

# plot observed biome-averaged profile. errorbar
vegtype = ['boreal forest','temperate forest','tropical forest','grassland',
           'cropland','shrublands','peatland','savannas','tundra','desert']
nrows, ncols = 2, 5
fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(15,8),sharey=True)
for n,row in enumerate(axes):
    for m,ax in enumerate(row):
        idx = ncols*n+m
#        if idx == 8:
#            break     
        dum = prep.get_biome_ave(idx+1, incre_data, 'D14C_BulkLayer') # biome index (1-10)
        ax.errorbar(dum['mean'], dum['cont_depth'], xerr=dum['std'], fmt='-')    
        ax.locator_params(nbins=6)
        ax.set_ylabel('depth(cm)')
        ax.set_xlabel(r'$\Delta^{14}C$')
        ax.set_title(vegtype[idx])
        ax.set_xlim([-1000, 500])
        ax.set_ylim([-50, 200])
ax.invert_yaxis()
plt.tight_layout()

# plot observed biome-averaged profile. shaded errorbar
vegtype = ['boreal forest','temperate forest','tropical forest','grassland',
           'cropland','shrublands','peatland','savannas','tundra','desert']
nrows, ncols = 2, 5
fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(15,8),sharey=True)
for n,row in enumerate(axes):
    for m,ax in enumerate(row):
        idx = ncols*n+m
#        if idx == 8:
#            break     
        dum = prep.get_biome_ave(idx+1, incre_data, 'D14C_BulkLayer') # biome index (1-10)
        x = dum['mean']; y = dum['cont_depth']; err = dum['std']
        ax.plot(x, y, 'k-')    
        ax.locator_params(axis='y',tight=True,nbins=8)
        ax.locator_params(axis='x',tight=True,nbins=6)
        ax.fill_betweenx(y, x-err, x+err, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848',
                        linewidth=0, linestyle='-')
        ax.set_ylabel('depth(cm)')
        ax.set_xlabel(r'$\Delta^{14}C$')
        ax.set_title(vegtype[idx])
        ax.set_xlim([-1000, 500])
        ax.set_ylim([-10, 250])
ax.invert_yaxis()
plt.tight_layout()